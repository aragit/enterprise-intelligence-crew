from __future__ import annotations

import pytest

from src.orchestration.risk_gate import RiskGate, run_risk_gate
from src.schemas.payloads import TrendPayload, RiskPayload


class TestRiskGate:
    @pytest.fixture
    def gate(self):
        return RiskGate(threshold=0.7, max_iterations=3)

    @pytest.fixture
    def low_risk(self):
        return RiskPayload(is_safe=True, risk_score=0.2, flagged_keywords=[], required_revisions=[])

    @pytest.fixture
    def high_risk(self):
        return RiskPayload(
            is_safe=False,
            risk_score=0.85,
            flagged_keywords=["hallucination"],
            required_revisions=["Verify source data"],
        )

    @pytest.fixture
    def trend(self):
        return TrendPayload(
            trend_name="test",
            momentum_score=0.5,
            extracted_metrics={"a": 1},
            verified_sources=["https://example.com"],
        )

    def test_approve_low_risk(self, gate, trend, low_risk):
        decision, feedback = gate.run(trend, low_risk)
        assert decision == "approve"
        assert len(feedback) == 0

    def test_high_risk_loops_then_force_approve(self, gate, trend, high_risk):
        decision, feedback = gate.run(trend, high_risk)
        assert decision == "approve"
        assert len(feedback) >= 3
        assert all("exceeds threshold" in f for f in feedback if "Iteration" in f)

    def test_max_iterations_respected(self, trend, high_risk):
        gate = RiskGate(threshold=0.7, max_iterations=1)
        decision, feedback = gate.run(trend, high_risk)
        assert decision == "approve"
        assert len(feedback) >= 1

    def test_reject_when_no_risk(self, gate, trend):
        decision, feedback = gate.run(trend, None)
        assert decision == "reject"
        assert len(feedback) == 1

    def test_run_with_dicts(self, gate):
        trend_dict = {"trend_name": "test", "momentum_score": 0.5, "extracted_metrics": {}, "verified_sources": []}
        risk_dict = {"is_safe": True, "risk_score": 0.9, "flagged_keywords": [], "required_revisions": ["fix"]}
        decision, feedback = gate.run(trend_dict, risk_dict)
        assert decision == "approve"
        assert len(feedback) >= 3

    def test_run_function_standalone(self, trend, low_risk):
        decision, feedback = run_risk_gate(trend, low_risk)
        assert decision == "approve"

    def test_custom_threshold_lower(self, trend):
        low = RiskPayload(is_safe=True, risk_score=0.5)
        gate = RiskGate(threshold=0.3, max_iterations=2)
        decision, feedback = gate.run(trend, low)
        assert decision == "approve"
        assert len(feedback) >= 2

    def test_custom_threshold_higher(self, trend):
        moderate = RiskPayload(is_safe=False, risk_score=0.6)
        gate = RiskGate(threshold=0.7)
        decision, _ = gate.run(trend, moderate)
        assert decision == "approve"

    def test_standalone_run_without_class(self, trend, low_risk):
        decision, feedback = run_risk_gate(trend, low_risk)
        assert decision == "approve"

    def test_standalone_reject_no_risk(self, trend):
        decision, feedback = run_risk_gate(trend, None)
        assert decision == "reject"
