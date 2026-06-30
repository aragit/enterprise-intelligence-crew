import os
import yaml
from pathlib import Path
from crewai import Agent, Task, Crew, Process, LLM
from src.config import settings
from src.llm import OllamaNativeLLM
from src.schemas.payloads import TrendPayload, RiskPayload, ContentPayload

# Disable CrewAI telemetry to avoid timeout delays
os.environ["CREWAI_TELEMETRY_OPT_OUT"] = "true"
os.environ["OTEL_SDK_DISABLED"] = "true"


def _make_llm() -> LLM:
    if settings.llm_provider == "mock":
        return LLM(model="gpt-4o-mini", api_key="sk-mock-placeholder")
    if settings.llm_provider == "ollama":
        return OllamaNativeLLM(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            temperature=0.7,
        )
    if settings.llm_provider == "openai":
        kwargs = {"model": settings.openai_model}
        if settings.openai_api_key:
            kwargs["api_key"] = settings.openai_api_key
        return LLM(**kwargs)
    return LLM(model="gpt-4o-mini", api_key="sk-mock-placeholder")


def check_llm_health() -> str | None:
    """Returns None if healthy, or an error message string if unhealthy."""
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


class EnterpriseIntelligenceCrew:
    def __init__(self, tool_registry: dict | None = None):
        config_path = Path(__file__).parent.parent.parent / "configs" / "crew_config.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"Crew configuration file missing at: {config_path}")

        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.agents_config = self.config["agents"]
        self.tasks_config = self.config["tasks"]
        self._tool_registry = tool_registry or {}

    def build_agents(self) -> dict:
        agents = {}
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
            expected_output=self.tasks_config["investigate_trend"]["expected_output"].strip(),
            agent=agents["trend_investigator"],
            output_json=TrendPayload,
        )

        risk_task = Task(
            description=self.tasks_config["analyze_compliance"]["description"].strip(),
            expected_output=self.tasks_config["analyze_compliance"]["expected_output"].strip(),
            agent=agents["risk_analyst"],
            output_json=RiskPayload,
        )

        content_task = Task(
            description=self.tasks_config["generate_content"]["description"].strip(),
            expected_output=self.tasks_config["generate_content"]["expected_output"].strip(),
            agent=agents["copywriter"],
            output_json=ContentPayload,
        )

        return Crew(
            agents=list(agents.values()),
            tasks=[investigate_task, risk_task, content_task],
            process=Process.sequential,
            verbose=True,
        )
