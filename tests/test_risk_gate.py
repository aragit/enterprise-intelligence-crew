from __future__ import annotations

import pytest

from src.orchestration.risk_gate import RiskGate, RiskGateState
from src.schemas.payloads import TrendPayload, RiskPayload


class TestRiskGate:
    @pytest.fixture
    def gate(self):
        return RiskGate(threshold=0.7)

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

    def test_reject_high_risk(self, gate, trend, high_risk):
        decision, feedback = gate.run(trend, high_risk)
        assert decision == "reject"
        assert len(feedback) > 0

    def test_max_iterations_forces_approve(self, trend, high_risk):
        gate = RiskGate(threshold=0.7)
        state = RiskGateState(trend=trend, risk=high_risk, iteration=3, max_iterations=3)
        decision = gate.evaluate(state)
        assert decision == "approve"

    def test_evaluate_returns_reject_when_no_risk(self, gate, trend):
        state = RiskGateState(trend=trend, risk=None)
        decision = gate.evaluate(state)
        assert decision == "reject"

    def test_step_analyze_to_evaluate(self, gate, trend, low_risk):
        state = RiskGateState(trend=trend, risk=low_risk, phase="analyze")
        state = gate.step(state)
        assert state.phase == "evaluate_risk"

    def test_step_evaluate_to_approve(self, gate, trend, low_risk):
        state = RiskGateState(trend=trend, risk=low_risk, phase="evaluate_risk")
        state = gate.step(state)
        assert state.phase == "approve"

    def test_step_evaluate_to_reject(self, gate, trend, high_risk):
        state = RiskGateState(trend=trend, risk=high_risk, phase="evaluate_risk")
        state = gate.step(state)
        assert state.phase == "reject"

    def test_step_approve_to_generate(self, gate, trend, low_risk):
        state = RiskGateState(trend=trend, risk=low_risk, phase="approve")
        state = gate.step(state)
        assert state.phase == "generate"
