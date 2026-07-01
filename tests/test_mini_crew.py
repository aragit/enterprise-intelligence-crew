from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from crewai import Agent

from src.agents.intelligence_crew import _extract_payload
from src.schemas.payloads import TrendPayload


class TestMiniCrewExecution:

    def test_mini_crew_runs_single_agent(self, mock_intelligence_crew):
        crew = mock_intelligence_crew
        payload = TrendPayload(
            trend_name="Test", momentum_score=0.5,
            extracted_metrics={}, verified_sources=["https://example.com"]
        )

        class FakeTaskOutput:
            pydantic = payload
            json_dict = None
            raw = None

        class FakeCrewOutput:
            tasks_output = [FakeTaskOutput()]
            raw = ''

        with patch("src.agents.intelligence_crew.Crew") as mock_crew_cls:
            mock_crew_instance = MagicMock()
            mock_crew_cls.return_value = mock_crew_instance
            mock_crew_instance.kickoff.return_value = FakeCrewOutput()
            result = crew._run_agent("trend_investigator", {})
            assert result is not None
            assert result.get("trend_name") == "Test"

    def test_mini_crew_preserves_tools(self, mock_intelligence_crew):
        from src.tools.crew_tools import TOOL_REGISTRY
        crew = mock_intelligence_crew
        tools = TOOL_REGISTRY.get("trend_investigator", [])
        assert len(tools) > 0
        agents = crew.build_agents()
        agent = agents["trend_investigator"]
        assert len(agent.tools) == len(tools)

    def test_mini_crew_preserves_output_json(self, mock_intelligence_crew):
        from src.agents.intelligence_crew import _AGENT_TASK_MAP, _TASK_OUTPUT_MAP
        task_key = _AGENT_TASK_MAP["trend_investigator"]
        assert _TASK_OUTPUT_MAP[task_key] == TrendPayload

    def test_three_mini_crews_in_sequence(self, mock_intelligence_crew):
        crew = mock_intelligence_crew
        execution_order = []

        def mock_run_agent(agent_id, _inputs):
            execution_order.append(agent_id)
            if agent_id == "trend_investigator":
                return {"trend_name": "T", "momentum_score": 0.5, "extracted_metrics": {"m": 1}, "verified_sources": ["https://x.com"]}
            if agent_id == "risk_analyst":
                return {"is_safe": True, "risk_score": 0.1}
            return {"headline": "H", "body_content": "B", "metadata_tags": ["t"]}

        with patch.object(crew, "_run_agent", side_effect=mock_run_agent):
            crew.run_pipeline("test query")

        assert execution_order == ["trend_investigator", "risk_analyst", "copywriter"]

    def test_extract_payload_fallback_limit(self):
        result = _extract_payload(None, TrendPayload)
        assert result == {}
