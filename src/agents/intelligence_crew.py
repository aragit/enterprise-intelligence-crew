from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import yaml
from crewai import Agent, Crew, LLM, Process, Task

from src.config import settings
from src.llm import OllamaNativeLLM, MockNativeLLM
from src.schemas.payloads import ContentPayload, RiskPayload, TrendPayload
from src.orchestration.risk_gate import run_risk_gate, GATE_THRESHOLD, MAX_ITERATIONS
from src.tools.crew_tools import TOOL_REGISTRY
from src.memory.crew_memory import CrewMemory

os.environ["CREWAI_TELEMETRY_OPT_OUT"] = "true"
os.environ["OTEL_SDK_DISABLED"] = "true"

logger = logging.getLogger(__name__)


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


def _parse_crew_output(output: Any) -> dict[str, Any]:
    raw = str(output)
    trend: dict[str, Any] = {}
    risk: dict[str, Any] = {}
    content: dict[str, Any] = {}

    tasks_output = getattr(output, "tasks_output", None)
    if tasks_output:
        for t in tasks_output:
            val = getattr(t, "exported_output", t.raw) if hasattr(t, "exported_output") else str(t)
            if isinstance(val, str):
                try:
                    val = json.loads(val)
                except (json.JSONDecodeError, TypeError):
                    pass
            if isinstance(val, dict):
                keys = set(val.keys())
                if {"trend_name", "momentum_score"} & keys:
                    trend = val
                elif {"is_safe", "risk_score"} & keys:
                    risk = val
                elif {"headline", "body_content"} & keys:
                    content = val

    if not trend and not risk and not content:
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return {"output": raw, **parsed}
        except (json.JSONDecodeError, TypeError):
            pass

    return {"trend": trend, "risk": risk, "content": content, "output": raw}


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

        investigate_task = Task(
            description=self.tasks_config["investigate_trend"]["description"].strip(),
            expected_output=self.tasks_config["investigate_trend"][
                "expected_output"
            ].strip(),
            agent=agents["trend_investigator"],
            output_json=TrendPayload,
        )

        risk_task = Task(
            description=self.tasks_config["analyze_compliance"]["description"].strip(),
            expected_output=self.tasks_config["analyze_compliance"][
                "expected_output"
            ].strip(),
            agent=agents["risk_analyst"],
            output_json=RiskPayload,
        )

        content_task = Task(
            description=self.tasks_config["generate_content"]["description"].strip(),
            expected_output=self.tasks_config["generate_content"][
                "expected_output"
            ].strip(),
            agent=agents["copywriter"],
            output_json=ContentPayload,
        )

        return Crew(
            agents=list(agents.values()),
            tasks=[investigate_task, risk_task, content_task],
            process=Process.sequential,
            verbose=True,
        )

    def run_pipeline(self, query_context: str) -> dict[str, Any]:
        crew = self.build_crew()
        result = crew.kickoff(inputs={"query_context": query_context})

        parsed = _parse_crew_output(result)

        trend_data = parsed.get("trend", {})
        risk_data = parsed.get("risk", {})
        content_data = parsed.get("content", {})

        decision = "approve"
        feedback: list[str] = []
        if trend_data and risk_data:
            decision, feedback = run_risk_gate(
                trend_data,
                risk_data,
                threshold=GATE_THRESHOLD,
                max_iterations=MAX_ITERATIONS,
            )
            logger.info(
                "RiskGate decision=%s on %d feedback items",
                decision,
                len(feedback),
            )

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
            "output": parsed.get("output", str(result)),
            "trend": trend_data,
            "risk": risk_data,
            "content": content_data,
            "gate_decision": decision,
            "gate_feedback": feedback,
        }
