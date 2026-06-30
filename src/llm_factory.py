from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from src.config import settings


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
        import httpx

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
        content = "".join(json.loads(line)["message"]["content"] for line in lines if line)
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
