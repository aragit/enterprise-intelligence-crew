from __future__ import annotations

from unittest.mock import patch


class TestFullPipelineIntegration:

    def test_pipeline_happy_path(self, mock_intelligence_crew):
        crew = mock_intelligence_crew
        result = crew.run_pipeline("Edge AI in healthcare")
        assert "trend" in result
        assert "risk" in result
        assert "content" in result
        assert result["trend"].get("trend_name")
        assert result["risk"].get("is_safe") is not None
        assert result["content"].get("headline")

    def test_pipeline_gate1_rejects_and_reruns(self, mock_intelligence_crew):
        crew = mock_intelligence_crew
        call_count = [0]

        def mock_run_agent(agent_id, _inputs):
            call_count[0] += 1
            if agent_id != "trend_investigator":
                return {"is_safe": True, "risk_score": 0.1}
            if call_count[0] == 1:
                return {"trend_name": "", "momentum_score": 0.5, "extracted_metrics": {}, "verified_sources": []}
            return {"trend_name": "Edge AI", "momentum_score": 0.5, "extracted_metrics": {"x": 1}, "verified_sources": ["https://example.com"]}

        with patch.object(crew, "_run_agent", side_effect=mock_run_agent):
            result = crew.run_pipeline("test", max_iterations=3)
        assert call_count[0] >= 2
        assert result["trend"]["trend_name"] == "Edge AI"

    def test_pipeline_gate2_rejects_and_reruns(self, mock_intelligence_crew):
        crew = mock_intelligence_crew
        agent_call_count = [0]
        gate_call_count = [0]

        def mock_run_agent(agent_id, _inputs):
            agent_call_count[0] += 1
            if agent_id == "trend_investigator":
                return {"trend_name": "Edge AI", "momentum_score": 0.5, "extracted_metrics": {"x": 1}, "verified_sources": ["https://example.com"]}
            if agent_id == "risk_analyst":
                return {"is_safe": False, "risk_score": 0.9, "flagged_keywords": ["bad"], "required_revisions": ["fix"]}
            return {"headline": "H", "body_content": "B", "metadata_tags": ["t"]}

        def mock_risk_gate(trend, risk, threshold=0.7, max_iterations=3):
            gate_call_count[0] += 1
            if gate_call_count[0] == 1:
                return ("reject", ["risk_score exceeds threshold"])
            return ("approve", [])

        with patch.object(crew, "_run_agent", side_effect=mock_run_agent):
            with patch("src.agents.intelligence_crew.run_risk_gate", side_effect=mock_risk_gate):
                result = crew.run_pipeline("test", max_iterations=3)
        assert agent_call_count[0] == 4  # trend + risk (×2 for re-run) + copywriter
        assert result["risk"]["is_safe"] is False

    def test_pipeline_circuit_breaker(self, mock_intelligence_crew):
        crew = mock_intelligence_crew
        gate_count = [0]

        def always_bad(agent_id, _inputs):
            if agent_id == "trend_investigator":
                return {"trend_name": "", "momentum_score": 0.5, "extracted_metrics": {}, "verified_sources": []}
            return {"is_safe": True, "risk_score": 0.1}

        def rejecting_gate(*args, **kwargs):
            gate_count[0] += 1
            return ("reject", ["always rejects"])

        with patch.object(crew, "_run_agent", side_effect=always_bad):
            with patch("src.agents.intelligence_crew.run_trend_gate", side_effect=rejecting_gate):
                result = crew.run_pipeline("test", max_iterations=2)
        assert "Circuit breaker" in str(result.get("gate_feedback", []))
