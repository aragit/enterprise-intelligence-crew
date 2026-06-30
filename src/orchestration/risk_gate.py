from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

from src.schemas.payloads import TrendPayload, RiskPayload, ContentPayload

logger = logging.getLogger(__name__)

State = Literal["analyze", "evaluate_risk", "approve", "reject", "generate", "done"]
Decision = Literal["approve", "reject"]


@dataclass
class RiskGateState:
    phase: State = "analyze"
    trend: TrendPayload | None = None
    risk: RiskPayload | None = None
    content: ContentPayload | None = None
    feedback: list[str] = field(default_factory=list)
    iteration: int = 0
    max_iterations: int = 3


class RiskGate:
    def __init__(self, threshold: float = 0.7):
        self.threshold = threshold

    def evaluate(self, state: RiskGateState) -> Decision:
        if state.risk is None:
            return "reject"

        score = state.risk.risk_score
        logger.info("RiskGate: risk_score=%.2f threshold=%.2f", score, self.threshold)

        if state.iteration >= state.max_iterations:
            logger.warning("RiskGate: max iterations (%d) reached, forcing approve", state.max_iterations)
            return "approve"

        if score > self.threshold:
            state.feedback.append(
                f"Iteration {state.iteration + 1}: risk_score {score} exceeds threshold {self.threshold}"
            )
            if state.risk.required_revisions:
                state.feedback.extend(state.risk.required_revisions)
            state.iteration += 1
            return "reject"

        return "approve"

    def step(self, state: RiskGateState) -> RiskGateState:
        if state.phase == "analyze":
            state.phase = "evaluate_risk"
        elif state.phase == "evaluate_risk":
            decision = self.evaluate(state)
            if decision == "approve":
                state.phase = "approve"
            else:
                state.phase = "reject"
        elif state.phase == "approve":
            state.phase = "generate"
        elif state.phase == "reject":
            state.phase = "analyze"
        elif state.phase == "generate":
            state.phase = "done"

        return state

    def run(self, trend: TrendPayload, risk: RiskPayload) -> tuple[Decision, list[str]]:
        state = RiskGateState(trend=trend, risk=risk)
        last_decision: Decision = "reject"

        while state.phase != "done":
            if state.phase == "evaluate_risk":
                last_decision = self.evaluate(state)
            state = self.step(state)

        return (last_decision, state.feedback)
