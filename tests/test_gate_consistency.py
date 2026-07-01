from __future__ import annotations

import pytest

from src.orchestration.risk_gate import run_trend_gate, run_risk_gate
from src.schemas.payloads import TrendPayload, RiskPayload


class TestCrossGateConsistency:

    def test_both_gates_agree_on_safe_payload(self):
        trend = TrendPayload(
            trend_name="Safe Trend",
            momentum_score=0.5,
            extracted_metrics={"mentions": 100},
            verified_sources=["https://trusted.com"]
        )
        risk = RiskPayload(is_safe=True, risk_score=0.1)
        d1, _ = run_trend_gate(trend)
        d2, _ = run_risk_gate(trend=trend, risk=risk)
        assert d1 == d2, f"Gate drift: trend_gate={d1}, risk_gate={d2}"

    def test_both_gates_agree_on_unsafe_payload(self):
        trend = TrendPayload(
            trend_name="",
            momentum_score=1.5,
            extracted_metrics={},
            verified_sources=[]
        )
        d1, _ = run_trend_gate(trend)
        d2, _ = run_risk_gate(trend=trend, risk=None)
        assert d1 == d2, f"Gate drift: trend_gate={d1}, risk_gate={d2}"
