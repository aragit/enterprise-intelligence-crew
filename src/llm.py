from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import httpx

from crewai.llms.base_llm import BaseLLM, llm_call_context

from src.config import settings

if TYPE_CHECKING:
    from crewai.agents.agent_builder.base_agent import BaseAgent
    from crewai.task import Task
    from crewai.tools.base_tool import BaseTool
    from crewai.utilities.types import LLMMessage


logger = logging.getLogger(__name__)

OLLAMA_DEFAULT_TIMEOUT = 300.0


# ---------------------------------------------------------------------------
# High-level provider abstraction (for direct use outside CrewAI pipeline)
# ---------------------------------------------------------------------------


@dataclass
class LLMResponse:
    content: str
    raw: dict[str, Any] | None = None


class BaseLLMProvider(ABC):
    @abstractmethod
    def generate(self, prompt: str, **kwargs: Any) -> LLMResponse:
        ...

    @abstractmethod
    def chat(
        self, messages: list[dict[str, str]], **kwargs: Any
    ) -> LLMResponse:
        ...


class OllamaProvider(BaseLLMProvider):
    def __init__(self) -> None:
        self.base_url = settings.ollama_base_url.rstrip("/")
        self.model = settings.ollama_model
        self._client = httpx.Client(timeout=120.0)

    def generate(self, prompt: str, **kwargs: Any) -> LLMResponse:
        resp = self._client.post(
            f"{self.base_url}/api/generate",
            json={"model": self.model, "prompt": prompt, **kwargs},
        )
        resp.raise_for_status()
        lines = resp.text.strip().split("\n")
        content = "".join(json.loads(line)["response"] for line in lines if line)
        return LLMResponse(content=content, raw={"model": self.model})

    def chat(
        self, messages: list[dict[str, str]], **kwargs: Any
    ) -> LLMResponse:
        resp = self._client.post(
            f"{self.base_url}/api/chat",
            json={"model": self.model, "messages": messages, **kwargs},
        )
        resp.raise_for_status()
        lines = resp.text.strip().split("\n")
        content = "".join(
            json.loads(line)["message"]["content"] for line in lines if line
        )
        return LLMResponse(content=content, raw={"model": self.model})


class OpenAIProvider(BaseLLMProvider):
    def __init__(self) -> None:
        from openai import OpenAI

        self._client = OpenAI(api_key=settings.openai_api_key)
        self.model = settings.openai_model

    def generate(self, prompt: str, **kwargs: Any) -> LLMResponse:
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )
        return LLMResponse(
            content=resp.choices[0].message.content or "",
            raw=resp.model_dump(),
        )

    def chat(
        self, messages: list[dict[str, str]], **kwargs: Any
    ) -> LLMResponse:
        resp = self._client.chat.completions.create(
            model=self.model, messages=messages, **kwargs
        )
        return LLMResponse(
            content=resp.choices[0].message.content or "",
            raw=resp.model_dump(),
        )


class MockProvider(BaseLLMProvider):
    def __init__(self) -> None:
        self.model = "mock"

    def generate(self, prompt: str, **kwargs: Any) -> LLMResponse:
        return LLMResponse(
            content=f"[mock response to: {prompt[:60]}]",
            raw={"model": self.model, "prompt": prompt},
        )

    def chat(
        self, messages: list[dict[str, str]], **kwargs: Any
    ) -> LLMResponse:
        last = messages[-1]["content"] if messages else ""
        return LLMResponse(
            content=f"[mock response to: {last[:60]}]",
            raw={"model": self.model, "messages": messages},
        )


def get_llm_provider() -> BaseLLMProvider:
    provider_map = {
        "ollama": OllamaProvider,
        "openai": OpenAIProvider,
        "mock": MockProvider,
    }
    cls = provider_map.get(settings.llm_provider)
    if cls is None:
        raise ValueError(f"Unknown LLM provider: {settings.llm_provider}")
    return cls()


# ---------------------------------------------------------------------------
# CrewAI-compatible Ollama adapter (used by the agent pipeline)
# ---------------------------------------------------------------------------


class OllamaNativeLLM(BaseLLM):
    llm_type: str = "ollama_native"
    timeout: float = OLLAMA_DEFAULT_TIMEOUT

    def _build_url(self) -> str:
        base = (self.base_url or "http://localhost:11434").rstrip("/")
        return f"{base}/api/chat"

    def _prepare_payload(
        self,
        messages: list[LLMMessage],
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
        }

        options: dict[str, Any] = {}
        if self.temperature is not None:
            options["temperature"] = self.temperature
        if self.top_p is not None:
            options["top_p"] = self.top_p
        if self.max_tokens is not None:
            options["num_predict"] = int(self.max_tokens)
        if self.seed is not None:
            options["seed"] = self.seed
        if self.stop:
            options["stop"] = self.stop

        if options:
            payload["options"] = options

        if tools:
            payload["tools"] = self._format_tools(tools)

        return payload

    @staticmethod
    def _format_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        formatted = []
        for tool in tools:
            fn = tool.get("function", {})
            formatted.append({
                "type": "function",
                "function": {
                    "name": fn.get("name", ""),
                    "description": fn.get("description", ""),
                    "parameters": fn.get("parameters", {}),
                },
            })
        return formatted

    def _extract_content(self, data: dict[str, Any]) -> str:
        return data.get("message", {}).get("content", "")

    def _extract_tool_calls(self, data: dict[str, Any]) -> list[dict[str, Any]]:
        message = data.get("message", {})
        raw = message.get("tool_calls", [])
        if not raw:
            return []
        result = []
        for tc in raw:
            fn = tc.get("function", {})
            result.append({
                "id": tc.get("id", ""),
                "type": "function",
                "function": {
                    "name": fn.get("name", ""),
                    "arguments": json.dumps(fn.get("arguments", {})),
                },
            })
        return result

    def _extract_usage(self, data: dict[str, Any]) -> dict[str, int]:
        return {
            "prompt_tokens": data.get("prompt_eval_count", 0),
            "completion_tokens": data.get("eval_count", 0),
            "total_tokens": (data.get("prompt_eval_count", 0)
                             + data.get("eval_count", 0)),
        }

    def call(
        self,
        messages: str | list[LLMMessage],
        tools: list[dict[str, BaseTool]] | None = None,
        callbacks: list[Any] | None = None,
        available_functions: dict[str, Any] | None = None,
        from_task: Task | None = None,
        from_agent: BaseAgent | None = None,
        response_model: type[BaseModel] | None = None,  # noqa: F821
    ) -> str | Any:
        from pydantic import BaseModel as PydanticBaseModel

        from crewai.events.types.llm_events import LLMCallType

        with llm_call_context():
            self._emit_call_started_event(
                messages=messages,
                tools=tools,
                callbacks=callbacks,
                available_functions=available_functions,
                from_task=from_task,
                from_agent=from_agent,
            )

            try:
                formatted_messages = self._format_messages(messages)

                if not self._invoke_before_llm_call_hooks(
                    formatted_messages, from_agent
                ):
                    raise ValueError("LLM call blocked by before_llm_call hook")

                payload = self._prepare_payload(formatted_messages, tools)
                url = self._build_url()

                resp = httpx.post(
                    url,
                    json=payload,
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                data = resp.json()

                content = self._extract_content(data)
                content = self._apply_stop_words(content)
                tool_calls = self._extract_tool_calls(data)
                usage = self._extract_usage(data)

                if tool_calls and available_functions:
                    return self._handle_tool_call(
                        tool_calls,
                        available_functions,
                        from_task,
                        from_agent,
                    )

                if tool_calls and not available_functions:
                    return tool_calls

                if response_model is not None:
                    try:
                        parsed = json.loads(content)
                        result = response_model.model_validate(parsed)
                        json_result = result.model_dump_json()
                        self._emit_call_completed_event(
                            response=json_result,
                            call_type=LLMCallType.LLM_CALL,
                            from_task=from_task,
                            from_agent=from_agent,
                            messages=formatted_messages,
                            usage=usage,
                        )
                        return json_result
                    except (json.JSONDecodeError, ValueError):
                        pass

                self._emit_call_completed_event(
                    response=content,
                    call_type=LLMCallType.LLM_CALL,
                    from_task=from_task,
                    from_agent=from_agent,
                    messages=formatted_messages,
                    usage=usage,
                )

                result: str | Any = content
                if isinstance(result, str):
                    result = self._invoke_after_llm_call_hooks(
                        formatted_messages, result, from_agent
                    )
                return result

            except httpx.HTTPStatusError as e:
                detail = self._parse_error(e.response)
                msg = f"Ollama API error ({e.response.status_code}): {detail}"
                logger.error(msg)
                self._emit_call_failed_event(
                    error=msg, from_task=from_task, from_agent=from_agent
                )
                raise RuntimeError(msg) from e
            except httpx.TimeoutException as e:
                msg = f"Ollama request timed out after {self.timeout}s"
                logger.error(msg)
                self._emit_call_failed_event(
                    error=msg, from_task=from_task, from_agent=from_agent
                )
                raise TimeoutError(msg) from e
            except httpx.ConnectError as e:
                msg = (
                    f"Cannot connect to Ollama at {self.base_url or 'http://localhost:11434'}. "
                    f"Ensure Ollama is running (ollama serve)."
                )
                logger.error(msg)
                self._emit_call_failed_event(
                    error=msg, from_task=from_task, from_agent=from_agent
                )
                raise ConnectionError(msg) from e
            except Exception as e:
                msg = f"Ollama call failed: {e!s}"
                logger.error(msg)
                self._emit_call_failed_event(
                    error=msg, from_task=from_task, from_agent=from_agent
                )
                raise RuntimeError(msg) from e

    async def acall(
        self,
        messages: str | list[LLMMessage],
        tools: list[dict[str, BaseTool]] | None = None,
        callbacks: list[Any] | None = None,
        available_functions: dict[str, Any] | None = None,
        from_task: Task | None = None,
        from_agent: BaseAgent | None = None,
        response_model: type[BaseModel] | None = None,  # noqa: F821
    ) -> str | Any:
        from pydantic import BaseModel as PydanticBaseModel

        from crewai.events.types.llm_events import LLMCallType

        with llm_call_context():
            self._emit_call_started_event(
                messages=messages,
                tools=tools,
                callbacks=callbacks,
                available_functions=available_functions,
                from_task=from_task,
                from_agent=from_agent,
            )

            try:
                formatted_messages = self._format_messages(messages)

                if not self._invoke_before_llm_call_hooks(
                    formatted_messages, from_agent
                ):
                    raise ValueError("LLM call blocked by before_llm_call hook")

                payload = self._prepare_payload(formatted_messages, tools)
                url = self._build_url()

                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    resp = await client.post(url, json=payload)
                    resp.raise_for_status()
                    data = resp.json()

                content = self._extract_content(data)
                content = self._apply_stop_words(content)
                tool_calls = self._extract_tool_calls(data)
                usage = self._extract_usage(data)

                if tool_calls and available_functions:
                    return self._handle_tool_call(
                        tool_calls,
                        available_functions,
                        from_task,
                        from_agent,
                    )

                if tool_calls and not available_functions:
                    return tool_calls

                if response_model is not None:
                    try:
                        parsed = json.loads(content)
                        result = response_model.model_validate(parsed)
                        json_result = result.model_dump_json()
                        self._emit_call_completed_event(
                            response=json_result,
                            call_type=LLMCallType.LLM_CALL,
                            from_task=from_task,
                            from_agent=from_agent,
                            messages=formatted_messages,
                            usage=usage,
                        )
                        return json_result
                    except (json.JSONDecodeError, ValueError):
                        pass

                self._emit_call_completed_event(
                    response=content,
                    call_type=LLMCallType.LLM_CALL,
                    from_task=from_task,
                    from_agent=from_agent,
                    messages=formatted_messages,
                    usage=usage,
                )

                result: str | Any = content
                if isinstance(result, str):
                    result = self._invoke_after_llm_call_hooks(
                        formatted_messages, result, from_agent
                    )
                return result

            except httpx.HTTPStatusError as e:
                detail = self._parse_error(e.response)
                msg = f"Ollama API error ({e.response.status_code}): {detail}"
                logger.error(msg)
                self._emit_call_failed_event(
                    error=msg, from_task=from_task, from_agent=from_agent
                )
                raise RuntimeError(msg) from e
            except httpx.TimeoutException as e:
                msg = f"Ollama request timed out after {self.timeout}s"
                logger.error(msg)
                self._emit_call_failed_event(
                    error=msg, from_task=from_task, from_agent=from_agent
                )
                raise TimeoutError(msg) from e
            except httpx.ConnectError as e:
                msg = (
                    f"Cannot connect to Ollama at {self.base_url or 'http://localhost:11434'}. "
                    f"Ensure Ollama is running (ollama serve)."
                )
                logger.error(msg)
                self._emit_call_failed_event(
                    error=msg, from_task=from_task, from_agent=from_agent
                )
                raise ConnectionError(msg) from e
            except Exception as e:
                msg = f"Ollama call failed: {e!s}"
                logger.error(msg)
                self._emit_call_failed_event(
                    error=msg, from_task=from_task, from_agent=from_agent
                )
                raise RuntimeError(msg) from e

    @staticmethod
    def _parse_error(response: httpx.Response) -> str:
        try:
            body = response.json()
            return body.get("error", {}).get("message", str(body))
        except Exception:
            return response.text[:500]

    def supports_function_calling(self) -> bool:
        return True

    def get_context_window_size(self) -> int:
        return 131072

    @classmethod
    def list_available_models(cls, base_url: str = "http://localhost:11434") -> list[str]:
        try:
            resp = httpx.get(f"{base_url.rstrip('/')}/api/tags", timeout=5.0)
            resp.raise_for_status()
            data = resp.json()
            return [m["name"] for m in data.get("models", [])]
        except Exception:
            return []

    @classmethod
    def check_model_exists(
        cls, model: str, base_url: str = "http://localhost:11434"
    ) -> bool:
        available = cls.list_available_models(base_url)
        return model in available


class MockNativeLLM(BaseLLM):
    llm_type: str = "mock_native"
    model: str = "mock"
    timeout: float = 10.0

    def call(
        self,
        messages: str | list[LLMMessage],
        tools: list[dict[str, Any]] | None = None,
        callbacks: list[Any] | None = None,
        available_functions: dict[str, Any] | None = None,
        from_task: Any = None,
        from_agent: Any = None,
        response_model: type[BaseModel] | None = None,
    ) -> str | Any:
        if response_model is not None:
            fake = self._make_fake(response_model)
            return fake.model_dump(mode="json")
        content = str(messages) if isinstance(messages, str) else (
            messages[-1].get("content", "") if messages else ""
        )
        return json.dumps({
            "trend_name": "Mock Trend",
            "momentum_score": 0.5,
            "extracted_metrics": {"repos": 100},
            "verified_sources": ["https://example.com/mock"],
            "is_safe": True,
            "risk_score": 0.1,
            "flagged_keywords": [],
            "required_revisions": [],
            "headline": "Mock Headline",
            "body_content": "Mock body content for testing purposes.",
            "metadata_tags": ["mock", "test"],
        })

    async def acall(
        self,
        messages: str | list[LLMMessage],
        **kwargs: Any,
    ) -> str | Any:
        return self.call(messages, **kwargs)

    @staticmethod
    def _make_fake(model_class: type[BaseModel]) -> BaseModel:
        import copy
        fake_kwargs = {}
        for name, field in model_class.model_fields.items():
            if isinstance(field.default_factory, type) and field.default_factory is list:
                fake_kwargs[name] = ["mock_item"]
            elif isinstance(field.default, list):
                fake_kwargs[name] = copy.copy(field.default)
            elif name == "risk_score":
                fake_kwargs[name] = 0.1
            elif name == "momentum_score":
                fake_kwargs[name] = 0.5
            elif name == "headline":
                fake_kwargs[name] = "Mock Headline"
            elif name == "trend_name":
                fake_kwargs[name] = "Mock Trend"
            elif name == "body_content":
                fake_kwargs[name] = "Mock body content for testing."
            elif name == "verified_sources":
                fake_kwargs[name] = ["https://example.com"]
            elif name == "metadata_tags":
                fake_kwargs[name] = ["mock", "test"]
            elif name == "is_safe":
                fake_kwargs[name] = True
            elif name == "extracted_metrics":
                fake_kwargs[name] = {"mock_key": "mock_value"}
            elif field.annotation is str:
                fake_kwargs[name] = "mock"
            elif field.annotation is int:
                fake_kwargs[name] = 0
            elif field.annotation is float:
                fake_kwargs[name] = 0.0
            elif field.annotation is bool:
                fake_kwargs[name] = True
            elif hasattr(field.annotation, "__origin__") and field.annotation.__origin__ is dict:
                fake_kwargs[name] = {}
            else:
                fake_kwargs[name] = None
        return model_class(**fake_kwargs)

    def supports_function_calling(self) -> bool:
        return False

    def get_context_window_size(self) -> int:
        return 4096
