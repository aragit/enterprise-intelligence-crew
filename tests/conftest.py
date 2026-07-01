from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

os.environ["CREWAI_TELEMETRY_OPT_OUT"] = "true"


@pytest.fixture(autouse=True)
def reset_settings():
    from src.config import settings

    settings.llm_provider = "mock"
    settings.openai_api_key = None
    os.environ["OPENAI_API_KEY"] = ""
    yield


@pytest.fixture
def sample_trend_payload() -> dict[str, Any]:
    return {
        "trend_name": "AI in Manufacturing",
        "momentum_score": 0.85,
        "extracted_metrics": {"github_repos": 1200, "patents": 450},
        "verified_sources": ["https://reuters.com/ai-manufacturing"],
    }


@pytest.fixture
def sample_risk_payload() -> dict[str, Any]:
    return {
        "is_safe": True,
        "risk_score": 0.2,
        "flagged_keywords": [],
        "required_revisions": [],
    }


@pytest.fixture
def sample_content_payload() -> dict[str, Any]:
    return {
        "headline": "AI Transforms Manufacturing",
        "body_content": "Artificial intelligence is revolutionizing the manufacturing sector with predictive maintenance and quality control.",
        "metadata_tags": ["AI", "Manufacturing", "Industry 4.0"],
    }


@pytest.fixture
def mock_intelligence_crew():
    from src.agents.intelligence_crew import EnterpriseIntelligenceCrew
    return EnterpriseIntelligenceCrew()
