import sys
import os
from dotenv import load_dotenv
from src.config import settings
from src.agents.intelligence_crew import EnterpriseIntelligenceCrew, check_llm_health

# Disable CrewAI telemetry to avoid timeout delays
os.environ["CREWAI_TELEMETRY_OPT_OUT"] = "true"

load_dotenv()


def run_pipeline(query_context: str | None = None):
    print("=" * 60)
    print("Enterprise Multi-Agent Intelligence Crew Engine")
    print(f"  LLM Provider: {settings.llm_provider}")
    if settings.llm_provider == "ollama":
        print(f"  Ollama URL:   {settings.ollama_base_url}")
        print(f"  Model:        {settings.ollama_model}")
    print("=" * 60)

    health_msg = check_llm_health()
    if health_msg:
        print(f"\n[CRITICAL] {health_msg}")
        print("\nOptions:")
        print(f"  1. Pull the model: ollama pull {settings.ollama_model}")
        print(f"  2. Use a different model: OLLAMA_MODEL=qwen2.5:1.5b python3 main.py")
        print(f"  3. Use mock:      LLM_PROVIDER=mock python3 main.py")
        print(f"  4. Use OpenAI:    LLM_PROVIDER=openai OPENAI_API_KEY=sk-... python3 main.py")
        sys.exit(1)

    try:
        crew_engine = EnterpriseIntelligenceCrew().build_crew()

        inputs = {
            "query_context": query_context
            or "Next-generation neuro-symbolic agentic AI system designs for clinical multi-drug validation pipelines."
        }

        print(f"\nLaunching autonomous agent pipeline...")
        print(f"Context: {inputs['query_context']}\n")

        pipeline_output = crew_engine.kickoff(inputs=inputs)

        print("\n" + "=" * 60)
        print("PIPELINE COMPLETE")
        print("=" * 60)
        print(pipeline_output)

    except Exception as e:
        print(f"\n[CRITICAL] Pipeline execution failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    run_pipeline()
