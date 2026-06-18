import os
import sys
from dotenv import load_dotenv
from src.agents.intelligence_crew import EnterpriseIntelligenceCrew

# Load local telemetry and API key bounds
load_dotenv()

def run_pipeline():
    """Execution driver for the Enterprise Intelligence Crew."""
    if not os.getenv("OPENAI_API_KEY"):
        print("[CRITICAL] RUNTIME ERROR: 'OPENAI_API_KEY' is not set in the environment variables.")
        print("Please create a '.env' file or export your credentials to proceed.")
        sys.exit(1)

    print("=" * 60)
    print("Initializing Enterprise Multi-Agent Intelligence Crew Engine...")
    print("=" * 60)
    
    try:
        # Initialize factory and compile multi-agent pipeline
        crew_engine = EnterpriseIntelligenceCrew().build_crew()
        
        # Injection input context parameters
        inputs = {
            "query_context": "Next-generation neuro-symbolic agentic AI system designs for clinical multi-drug validation pipelines."
        }
        
        print(f"\n[EXECUTION] Launching autonomous agent matrix...")
        print(f"[CONTEXT] Target Domain Vector: {inputs['query_context']}\n")
        
        # Trigger the pipeline execution run
        pipeline_output = crew_engine.kickoff(inputs=inputs)
        
        print("\n" + "=" * 60)
        print("EXECUTION PIPELINE COMPLETE - TARGET SCHEMAS GENNED")
        print("=" * 60)
        print(pipeline_output)
        
    except Exception as e:
        print(f"\n[CRITICAL] Runtime execution failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    run_pipeline()
