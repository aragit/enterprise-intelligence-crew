from __future__ import annotations

import pytest

from src.llm_factory import MockProvider, OllamaProvider, OpenAIProvider, get_llm_provider
from src.config import settings


class TestMockProvider:
    def test_generate_returns_content(self):
        provider = MockProvider()
        resp = provider.generate("hello")
        assert isinstance(resp.content, str)
        assert "hello" in resp.content

    def test_chat_returns_content(self):
        provider = MockProvider()
        resp = provider.chat([{"role": "user", "content": "hi"}])
        assert isinstance(resp.content, str)
        assert "hi" in resp.content

    def test_empty_messages_chat(self):
        provider = MockProvider()
        resp = provider.chat([])
        assert resp.content == "[mock response to: ]"


class TestGetProvider:
    def test_mock_provider(self):
        settings.llm_provider = "mock"
        provider = get_llm_provider()
        assert isinstance(provider, MockProvider)

    def test_ollama_provider(self):
        settings.llm_provider = "ollama"
        provider = get_llm_provider()
        assert isinstance(provider, OllamaProvider)

    def test_openai_provider_no_key(self):
        settings.llm_provider = "openai"
        settings.openai_api_key = None
        with pytest.raises(Exception):
            get_llm_provider()

    def test_openai_provider_with_key(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        settings.llm_provider = "openai"
        settings.openai_api_key = "sk-test"
        provider = get_llm_provider()
        assert isinstance(provider, OpenAIProvider)

    def test_invalid_provider(self):
        settings.llm_provider = "invalid"
        with pytest.raises(ValueError):
            get_llm_provider()
