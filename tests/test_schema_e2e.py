from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.schemas.payloads import TrendPayload, ContentPayload


class TestSchemaValidationE2E:

    def test_trend_payload_url_sanitization(self):
        payload = TrendPayload(
            trend_name="Test",
            momentum_score=0.5,
            extracted_metrics={},
            verified_sources=[
                "See https://example.com/article for details",
                "not-a-url",
                "https://valid.com"
            ]
        )
        sources = payload.verified_sources
        assert "https://valid.com" in sources
        assert "https://example.com/article" in sources
        assert "not-a-url" not in sources

    def test_missing_required_field_raises(self):
        with pytest.raises(ValidationError):
            ContentPayload(headline="H", body_content="B")
