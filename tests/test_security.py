from __future__ import annotations

import pytest

from src.orchestration.risk_gate import run_trend_gate, run_risk_gate


class TestPromptInjectionResilience:

    def test_feedback_block_escapes_newlines(self, mock_intelligence_crew):
        crew = mock_intelligence_crew
        malicious = ["foo\nbar\nbaz", "normal feedback"]
        block = crew._build_feedback_block(malicious)
        assert "- foo" in block
        assert "bar" in block
        assert "baz" in block
        assert "- normal feedback" in block
        assert block.startswith("\n\nPREVIOUS ATTEMPT FEEDBACK:\n")

    def test_feedback_block_empty_input(self, mock_intelligence_crew):
        crew = mock_intelligence_crew
        assert crew._build_feedback_block([]) == ""
        assert crew._build_feedback_block(None) == ""


class TestBoundedResourceUsage:

    def test_gate1_handles_malformed_payload(self):
        bad = {"trend_name": None, "momentum_score": "abc", "extracted_metrics": None, "verified_sources": None}
        decision, feedback = run_trend_gate(bad)
        assert decision == "reject"
        assert len(feedback) > 0

    def test_gate2_handles_malformed_payload(self):
        decision, feedback = run_risk_gate(trend=None, risk=None)
        assert decision in ("approve", "reject")
        assert isinstance(feedback, list)
