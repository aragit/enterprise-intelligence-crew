from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from crewai import Agent, Crew, LLM, Process, Task
from pydantic import BaseModel, ValidationError

from src.config import settings
from src.llm import OllamaNativeLLM, MockNativeLLM
from src.schemas.payloads import ContentPayload, RiskPayload, TrendPayload
from src.orchestration.risk_gate import (
    run_risk_gate,
    run_trend_gate,
    GATE_THRESHOLD,
    MAX_ITERATIONS,
)
from src.tools.crew_tools import TOOL_REGISTRY
from src.memory.crew_memory import CrewMemory

os.environ["CREWAI_TELEMETRY_OPT_OUT"] = "true"
os.environ["OTEL_SDK_DISABLED"] = "true"

logger = logging.getLogger(__name__)

_AGENT_TASK_MAP: dict[str, str] = {
    "trend_investigator": "investigate_trend",
    "risk_analyst": "analyze_compliance",
    "copywriter": "generate_content",
}

_TASK_OUTPUT_MAP: dict[str, type[BaseModel]] = {
    "investigate_trend": TrendPayload,
    "analyze_compliance": RiskPayload,
    "generate_content": ContentPayload,
}


@dataclass
class PipelineState:
    """Mutable state accumulated through the pipeline steps."""
    query_context: str
    feedback: list[str] = field(default_factory=list)
    iteration: int = 0
    trend_data: dict[str, Any] | None = None
    risk_data: dict[str, Any] | None = None
    content_data: dict[str, Any] | None = None


def _make_llm() -> LLM:
    if settings.llm_provider == "mock":
        return MockNativeLLM(model="mock")
    if settings.llm_provider == "ollama":
        return OllamaNativeLLM(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            temperature=0.7,
        )
    if settings.llm_provider == "openai":
        kwargs: dict[str, Any] = {"model": settings.openai_model}
        if settings.openai_api_key:
            kwargs["api_key"] = settings.openai_api_key
        return LLM(**kwargs)
    return LLM(model="gpt-4o-mini", api_key="sk-mock-placeholder")


def check_llm_health() -> str | None:
    if settings.llm_provider == "mock":
        return None
    if settings.llm_provider == "ollama":
        available = OllamaNativeLLM.list_available_models(settings.ollama_base_url)
        if not available:
            try:
                import httpx
                r = httpx.get(f"{settings.ollama_base_url}/api/tags", timeout=5.0)
                r.raise_for_status()
            except httpx.ConnectError:
                return (
                    f"Cannot connect to Ollama at {settings.ollama_base_url}. "
                    f"Ensure Ollama is running: ollama serve"
                )
            except Exception as e:
                return f"Ollama health check failed: {e}"
            return (
                f"Ollama is reachable at {settings.ollama_base_url} "
                f"but no models are installed. Run: ollama pull <model>"
            )
        model = settings.ollama_model
        if model not in available:
            return (
                f"Configured model '{model}' not found in Ollama. "
                f"Available models: {', '.join(sorted(available))}. "
                f"Run: ollama pull {model}"
            )
        return None
    if settings.llm_provider == "openai":
        if not settings.openai_api_key:
            return "OPENAI_API_KEY not set"
        return None
    return f"Unknown LLM provider: {settings.llm_provider}"


def _payload_to_dict(payload: Any) -> dict[str, Any]:
    if hasattr(payload, "model_dump"):
        return payload.model_dump()
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, str):
        try:
            return json.loads(payload)
        except (json.JSONDecodeError, TypeError):
            pass
    return {"raw": str(payload)}


def _extract_payload(output: Any, expected_model: type[BaseModel]) -> dict[str, Any]:
    """Schema-aware extraction from CrewAI task output.

    Handles:
    - Direct Pydantic model instance
    - CrewOutput with tasks_output (each TaskOutput has .pydantic, .json_dict, .raw)
    - Raw CrewOutput with .raw attribute
    - Plain dict input
    - JSON string fallback
    """

    def _validate(raw_data: Any) -> dict[str, Any] | None:
        if isinstance(raw_data, expected_model):
            return raw_data.model_dump()
        if isinstance(raw_data, dict):
            try:
                return expected_model.model_validate(raw_data).model_dump()
            except ValidationError:
                pass
        if isinstance(raw_data, str):
            try:
                parsed = json.loads(raw_data)
                if isinstance(parsed, dict):
                    return expected_model.model_validate(parsed).model_dump()
            except (json.JSONDecodeError, ValidationError):
                pass
        return None

    # Try 1: Direct Pydantic or dict
    result = _validate(output)
    if result is not None:
        return result

    # Try 2: CrewOutput — check tasks_output
    tasks_output = getattr(output, "tasks_output", None)
    if tasks_output:
        for t in tasks_output:
            for attr in ("pydantic", "json_dict", "raw"):
                val = getattr(t, attr, None)
                if val is not None:
                    result = _validate(val)
                    if result is not None:
                        return result

    # Try 3: output.raw (CrewOutput raw attribute)
    raw_str = getattr(output, "raw", None)
    if isinstance(raw_str, str):
        result = _validate(raw_str)
        if result is not None:
            return result

    # Try 4: str(output) as last resort (may produce repr, try json.loads)
    try:
        result = _validate(str(output))
        if result is not None:
            return result
    except Exception:
        pass

    logger.error(
        "Could not extract %s from output: %s",
        expected_model.__name__,
        str(output)[:200],
    )
    return {}


class EnterpriseIntelligenceCrew:
    def __init__(self, tool_registry: dict[str, list] | None = None):
        config_path = (
            Path(__file__).parent.parent.parent / "configs" / "crew_config.yaml"
        )
        if not config_path.exists():
            raise FileNotFoundError(f"Crew configuration file missing at: {config_path}")

        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.agents_config = self.config["agents"]
        self.tasks_config = self.config["tasks"]
        self._tool_registry = tool_registry or TOOL_REGISTRY

    def build_agents(self) -> dict[str, Agent]:
        agents: dict[str, Agent] = {}
        for agent_id, cfg in self.agents_config.items():
            tools = self._tool_registry.get(agent_id, [])
            agents[agent_id] = Agent(
                role=cfg["role"].strip(),
                goal=cfg["goal"].strip(),
                backstory=cfg["backstory"].strip(),
                verbose=True,
                memory=False,
                allow_delegation=False,
                llm=_make_llm(),
                tools=tools,
            )
        return agents

    def build_crew(self) -> Crew:
        agents = self.build_agents()
        tasks: list[Task] = []
        for agent_id, task_key in _AGENT_TASK_MAP.items():
            output_model = _TASK_OUTPUT_MAP[task_key]
            tasks.append(Task(
                description=self.tasks_config[task_key]["description"].strip(),
                expected_output=self.tasks_config[task_key][
                    "expected_output"
                ].strip(),
                agent=agents[agent_id],
                output_json=output_model,
            ))
        return Crew(
            agents=list(agents.values()),
            tasks=tasks,
            process=Process.sequential,
            verbose=True,
        )

    def _build_feedback_block(self, feedback: list[str]) -> str:
        if not feedback:
            return ""
        fb_lines = "\n".join(f"- {f}" for f in feedback)
        return f"\n\nPREVIOUS ATTEMPT FEEDBACK:\n{fb_lines}"

    def _build_agent_inputs(self, state: PipelineState) -> dict[str, str]:
        inputs: dict[str, str] = {
            "query_context": state.query_context,
            "feedback": "",
            "trend_context": "",
            "risk_context": "",
        }
        if state.feedback:
            inputs["feedback"] = self._build_feedback_block(state.feedback)
        if state.trend_data:
            inputs["trend_context"] = json.dumps(
                state.trend_data, indent=2, default=str
            )
        if state.risk_data:
            inputs["risk_context"] = json.dumps(
                state.risk_data, indent=2, default=str
            )
        return inputs

    def _run_agent(
        self, agent_id: str, inputs: dict[str, str]
    ) -> dict[str, Any]:
        task_key = _AGENT_TASK_MAP[agent_id]
        output_model = _TASK_OUTPUT_MAP[task_key]
        agents = self.build_agents()
        task = Task(
            description=self.tasks_config[task_key]["description"].strip(),
            expected_output=self.tasks_config[task_key][
                "expected_output"
            ].strip(),
            agent=agents[agent_id],
            output_json=output_model,
        )
        crew = Crew(
            agents=[agents[agent_id]],
            tasks=[task],
            process=Process.sequential,
            verbose=False,
        )
        result = crew.kickoff(inputs=inputs)
        return _extract_payload(result, output_model)

    def run_pipeline(
        self, query_context: str, max_iterations: int = 3
    ) -> dict[str, Any]:
        state = PipelineState(query_context=query_context)

        # =====================================================
        # Agent 1: TrendInvestigator
        # =====================================================
        trend_data = self._run_agent(
            "trend_investigator", self._build_agent_inputs(state)
        )
        if not trend_data:
            return {
                "output": "",
                "error": "Trend investigation produced no output",
            }

        # Gate 1: Evaluate trend payload before passing to RiskAnalyst
        trend_iter = 0
        while trend_iter < max_iterations:
            g1_decision, g1_feedback = run_trend_gate(trend_data)
            if g1_decision == "approve":
                logger.info("Gate 1 (trend) approved")
                break
            trend_iter += 1
            if trend_iter >= max_iterations:
                logger.warning(
                    "Gate 1 max iterations (%d) reached, force-approving trend",
                    max_iterations,
                )
                state.feedback.append("Circuit breaker: max iterations reached for trend gate")
                break
            state.feedback.extend(g1_feedback)
            state.iteration = trend_iter
            logger.info(
                "Gate 1 rejected (iter %d/%d): %s",
                trend_iter,
                max_iterations,
                g1_feedback,
            )
            trend_data = self._run_agent(
                "trend_investigator", self._build_agent_inputs(state)
            )
            if not trend_data:
                return {
                    "output": "",
                    "error": "Trend investigation failed after gate rejection",
                }

        state.trend_data = trend_data

        # =====================================================
        # Agent 2: RiskAnalyst
        # =====================================================
        risk_data = self._run_agent(
            "risk_analyst", self._build_agent_inputs(state)
        )
        if not risk_data:
            return {
                "output": "",
                "trend": trend_data,
                "error": "Risk analysis produced no output",
            }

        # Gate 2: Evaluate risk payload before passing to Copywriter
        risk_iter = 0
        while risk_iter < max_iterations:
            g2_decision, g2_feedback = run_risk_gate(
                trend_data,
                risk_data,
                threshold=GATE_THRESHOLD,
                max_iterations=MAX_ITERATIONS,
            )
            if g2_decision == "approve":
                logger.info("Gate 2 (risk) approved")
                break
            risk_iter += 1
            if risk_iter >= max_iterations:
                logger.warning(
                    "Gate 2 max iterations (%d) reached, force-approving risk",
                    max_iterations,
                )
                state.feedback.append("Circuit breaker: max iterations reached for risk gate")
                break
            state.feedback.extend(g2_feedback)
            state.iteration += risk_iter
            logger.info(
                "Gate 2 rejected (iter %d/%d): %s",
                risk_iter,
                max_iterations,
                g2_feedback,
            )
            risk_data = self._run_agent(
                "risk_analyst", self._build_agent_inputs(state)
            )
            if not risk_data:
                return {
                    "output": "",
                    "trend": trend_data,
                    "error": "Risk analysis failed after gate rejection",
                }

        state.risk_data = risk_data

        # =====================================================
        # Agent 3: Copywriter
        # =====================================================
        content_data = self._run_agent(
            "copywriter", self._build_agent_inputs(state)
        )
        state.content_data = content_data

        # =====================================================
        # Store in memory
        # =====================================================
        try:
            mem = CrewMemory()
            mem.add_research(
                topic=query_context,
                trend_payload=trend_data,
                risk_payload=risk_data,
                content_payload=content_data,
            )
        except Exception as e:
            logger.warning("CrewMemory store failed: %s", e)

        return {
            "output": str(content_data or risk_data or trend_data),
            "trend": trend_data,
            "risk": risk_data,
            "content": content_data,
            "gate_decision": "approve",
            "gate_feedback": state.feedback,
        }
