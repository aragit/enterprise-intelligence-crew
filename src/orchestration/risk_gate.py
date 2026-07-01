from __future__ import annotations

import logging
from typing import Any, Literal
from typing_extensions import TypedDict

from langgraph.graph import END, StateGraph

from src.schemas.payloads import (
    ContentPayload,
    RiskPayload,
    TrendPayload,
)

logger = logging.getLogger(__name__)

GATE_THRESHOLD = 0.7
MAX_ITERATIONS = 3


class _GateState(TypedDict):
    phase: str
    trend: dict[str, Any] | None
    risk: dict[str, Any] | None
    content: dict[str, Any] | None
    feedback: list[str]
    iteration: int
    decision: str | None


def _to_dict(obj: Any) -> dict[str, Any]:
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if isinstance(obj, dict):
        return obj
    return {}


def _analyze(state: _GateState) -> _GateState:
    return {**state, "phase": "evaluate_risk"}


def _evaluate(
    state: _GateState, threshold: float, max_iterations: int
) -> _GateState:
    risk = state.get("risk")
    if risk is None:
        return {**state, "decision": "reject", "phase": "reject"}

    score = risk.get("risk_score", 0)
    iteration = state.get("iteration", 0)

    logger.info(
        "RiskGate: risk_score=%.2f threshold=%.2f iteration=%d/%d",
        score,
        threshold,
        iteration,
        max_iterations,
    )

    if iteration >= max_iterations:
        logger.warning(
            "RiskGate: max iterations (%d) reached, forcing approve",
            max_iterations,
        )
        return {**state, "decision": "approve", "phase": "approve"}

    if score > threshold:
        fb = list(state.get("feedback", []))
        fb.append(
            f"Iteration {iteration + 1}: "
            f"risk_score {score} exceeds threshold {threshold}"
        )
        for rev in risk.get("required_revisions", []):
            fb.append(rev)
        return {
            **state,
            "decision": "reject",
            "phase": "reject",
            "feedback": fb,
            "iteration": iteration + 1,
        }

    return {**state, "decision": "approve", "phase": "approve"}


def _approve_node(state: _GateState) -> _GateState:
    return {**state, "phase": "generate"}


def _reject_node(state: _GateState) -> _GateState:
    return {**state, "phase": "analyze"}


def _generate(state: _GateState) -> _GateState:
    return {**state, "phase": "done"}


def _route_decision(state: _GateState) -> Literal["approve", "reject"]:
    return state.get("decision", "reject")


def create_risk_graph(
    threshold: float = GATE_THRESHOLD,
    max_iterations: int = MAX_ITERATIONS,
):
    builder = StateGraph(_GateState)

    builder.add_node("analyze", _analyze)
    builder.add_node(
        "evaluate_risk",
        lambda s: _evaluate(s, threshold, max_iterations),
    )
    builder.add_node("approve", _approve_node)
    builder.add_node("reject", _reject_node)
    builder.add_node("generate", _generate)

    builder.set_entry_point("analyze")
    builder.add_edge("analyze", "evaluate_risk")
    builder.add_conditional_edges(
        "evaluate_risk",
        _route_decision,
        {"approve": "approve", "reject": "reject"},
    )
    builder.add_edge("approve", "generate")
    builder.add_edge("reject", "analyze")
    builder.add_edge("generate", END)

    return builder.compile()


_compiled_graphs: dict[tuple[float, int], Any] = {}


def _get_graph(
    threshold: float = GATE_THRESHOLD,
    max_iterations: int = MAX_ITERATIONS,
):
    key = (threshold, max_iterations)
    if key not in _compiled_graphs:
        _compiled_graphs[key] = create_risk_graph(threshold, max_iterations)
    return _compiled_graphs[key]


def run_risk_gate(
    trend: TrendPayload | RiskPayload | dict[str, Any],
    risk: RiskPayload | dict[str, Any] | None,
    threshold: float = GATE_THRESHOLD,
    max_iterations: int = MAX_ITERATIONS,
) -> tuple[str, list[str]]:
    if risk is None:
        return ("reject", ["No risk assessment available"])

    graph = _get_graph(threshold, max_iterations)
    initial: dict[str, Any] = {
        "phase": "analyze",
        "trend": _to_dict(trend),
        "risk": _to_dict(risk),
        "content": None,
        "feedback": [],
        "iteration": 0,
        "decision": None,
    }
    result = graph.invoke(initial)
    return (result.get("decision", "reject"), result.get("feedback", []))


def run_trend_gate(
    trend: TrendPayload | dict[str, Any] | None,
    threshold: float = GATE_THRESHOLD,
) -> tuple[str, list[str]]:
    """Evaluate a trend payload before it reaches RiskAnalyst.
    
    Checks:
    - momentum_score is in valid range (0.0–1.0)
    - verified_sources is non-empty
    - extracted_metrics is non-empty
    """
    if trend is None:
        return ("reject", ["No trend data to evaluate"])

    data = _to_dict(trend)
    feedback: list[str] = []

    score = data.get("momentum_score", 0)
    if not isinstance(score, (int, float)) or score < 0 or score > 1:
        feedback.append(f"Invalid momentum_score: {score} (must be 0.0–1.0)")

    sources = data.get("verified_sources", [])
    if not sources:
        feedback.append("No verified sources provided")

    metrics = data.get("extracted_metrics", {})
    if not metrics:
        feedback.append("No extracted metrics provided")

    trend_name = data.get("trend_name", "")
    if not trend_name or not trend_name.strip():
        feedback.append("Trend name is empty")

    if feedback:
        return ("reject", feedback)
    return ("approve", [])


class RiskGate:
    def __init__(
        self,
        threshold: float = GATE_THRESHOLD,
        max_iterations: int = MAX_ITERATIONS,
    ):
        self.threshold = threshold
        self.max_iterations = max_iterations

    def run(
        self,
        trend: TrendPayload | dict[str, Any],
        risk: RiskPayload | dict[str, Any],
    ) -> tuple[str, list[str]]:
        return run_risk_gate(
            trend, risk, threshold=self.threshold, max_iterations=self.max_iterations
        )
