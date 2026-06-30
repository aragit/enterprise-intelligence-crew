from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.schemas.payloads import TrendPayload, RiskPayload, ContentPayload


class TestTrendPayload:
    def test_valid_payload(self, sample_trend_payload):
        p = TrendPayload(**sample_trend_payload)
        assert p.trend_name == "AI in Manufacturing"
        assert p.momentum_score == 0.85

    def test_momentum_score_clamping(self):
        p = TrendPayload(
            trend_name="test",
            momentum_score=1.5,
            extracted_metrics={"a": 1},
            verified_sources=["https://example.com"],
        )
        assert p.momentum_score == 1.5

    def test_missing_field_raises(self):
        with pytest.raises(ValidationError):
            TrendPayload(trend_name="test")

    def test_invalid_url_raises(self):
        with pytest.raises(ValidationError):
            TrendPayload(
                trend_name="test",
                momentum_score=0.5,
                extracted_metrics={"a": 1},
                verified_sources=["not-a-url"],
            )

    def test_empty_verified_sources_allowed(self):
        p = TrendPayload(
            trend_name="test",
            momentum_score=0.5,
            extracted_metrics={"a": 1},
            verified_sources=[],
        )
        assert p.verified_sources == []


class TestRiskPayload:
    def test_valid_payload(self, sample_risk_payload):
        p = RiskPayload(**sample_risk_payload)
        assert p.is_safe is True
        assert p.risk_score == 0.2

    def test_default_lists(self):
        p = RiskPayload(is_safe=True, risk_score=0.0)
        assert p.flagged_keywords == []
        assert p.required_revisions == []

    def test_negative_risk_score(self):
        p = RiskPayload(is_safe=True, risk_score=-0.5)
        assert p.risk_score == -0.5


class TestContentPayload:
    def test_valid_payload(self, sample_content_payload):
        p = ContentPayload(**sample_content_payload)
        assert p.headline == "AI Transforms Manufacturing"

    def test_missing_metadata_tags(self):
        with pytest.raises(ValidationError):
            ContentPayload(headline="H", body_content="B")

    def test_minimal_valid(self):
        p = ContentPayload(headline="H", body_content="B", metadata_tags=["tag"])
        assert p.headline == "H"
