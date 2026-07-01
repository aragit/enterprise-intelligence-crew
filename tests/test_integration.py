from __future__ import annotations

import pytest

from src.agents.intelligence_crew import EnterpriseIntelligenceCrew
from src.config import settings
from src.llm import MockProvider
from src.schemas.payloads import TrendPayload, RiskPayload, ContentPayload
from src.orchestration.risk_gate import RiskGate


class TestWebSearchTool:
    def test_web_search_returns_results(self):
        import os
        if os.environ.get("CI") or not os.environ.get("NETWORK_TEST"):
            pytest.skip("Skipping network-dependent test")
        from src.tools.web_search import web_search, WebSearchInput
        result = web_search(WebSearchInput(query="python programming", max_results=2))
        assert result.success is True


class TestSentimentTool:
    def test_sentiment_on_trend_text(self):
        from src.tools.sentiment_analyzer import sentiment_analyzer, SentimentInput
        result = sentiment_analyzer(SentimentInput(text="AI is transforming the manufacturing industry with predictive maintenance."))
        assert result.success is True
        assert result.data["label"] in ("positive", "neutral")


class TestBiasTool:
    def test_bias_on_content(self):
        from src.tools.bias_detector import bias_detector, BiasInput
        result = bias_detector(BiasInput(text="This groundbreaking revolutionary product is guaranteed to change everything."))
        assert result.success is True
        assert result.data["verdict"] == "high bias"


class TestSEOOnContent:
    def test_seo_on_sample_content(self, sample_content_payload):
        from src.tools.seo_optimizer import seo_optimizer, SEOInput
        result = seo_optimizer(SEOInput(
            text=sample_content_payload["body_content"],
            target_keywords=sample_content_payload["metadata_tags"],
        ))
        assert result.success is True
        assert isinstance(result.data["readability_score"], float)


class TestRiskGateIntegration:
    def test_low_risk_approves(self):
        trend = TrendPayload(
            trend_name="AI Test",
            momentum_score=0.5,
            extracted_metrics={"a": 1},
            verified_sources=["https://example.com"],
        )
        risk = RiskPayload(is_safe=True, risk_score=0.1)
        gate = RiskGate(threshold=0.7)
        decision, feedback = gate.run(trend, risk)
        assert decision == "approve"
