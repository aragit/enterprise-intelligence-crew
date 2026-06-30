from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

import httpx

from crewai.llms.base_llm import BaseLLM, llm_call_context

if TYPE_CHECKING:
    from crewai.agents.agent_builder.base_agent import BaseAgent
    from crewai.task import Task
    from crewai.tools.base_tool import BaseTool
    from crewai.utilities.types import LLMMessage


logger = logging.getLogger(__name__)

OLLAMA_DEFAULT_TIMEOUT = 300.0


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
