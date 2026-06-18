import os
import yaml
from pathlib import Path
from crewai import Agent, Task, Crew, Process
from src.schemas.payloads import TrendPayload, RiskPayload, ContentPayload

class EnterpriseIntelligenceCrew:
    def __init__(self):
        # Locate config file relative to this module path
        config_path = Path(__file__).parent.parent.parent / "configs" / "crew_config.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"Crew configuration file missing at: {config_path}")
            
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        
        self.agents_config = self.config["agents"]
        self.tasks_config = self.config["tasks"]

    def build_agents(self) -> dict:
        """Instantiate underlying operational agents from yaml profiles."""
        agents = {}
        for agent_id, cfg in self.agents_config.items():
            agents[agent_id] = Agent(
                role=cfg["role"].strip(),
                goal=cfg["goal"].strip(),
                backstory=cfg["backstory"].strip(),
                verbose=True,
                memory=True
            )
        return agents

    def build_crew(self) -> Crew:
        """Construct tasks and bind them cleanly into a unified execution crew."""
        agents = self.build_agents()
        
        # Task 1: Unstructured Data Extraction mapped to TrendPayload schema
        investigate_task = Task(
            description=self.tasks_config["investigate_trend"]["description"].strip(),
            expected_output=self.tasks_config["investigate_trend"]["expected_output"].strip(),
            agent=agents["trend_investigator"],
            output_json=TrendPayload
        )
        
        # Task 2: Safety/Compliance Assessment mapped to RiskPayload schema
        risk_task = Task(
            description=self.tasks_config["analyze_compliance"]["description"].strip(),
            expected_output=self.tasks_config["analyze_compliance"]["expected_output"].strip(),
            agent=agents["risk_analyst"],
            output_json=RiskPayload
        )
        
        # Task 3: Content Synthesis mapped to final ContentPayload schema
        content_task = Task(
            description=self.tasks_config["generate_content"]["description"].strip(),
            expected_output=self.tasks_config["generate_content"]["expected_output"].strip(),
            agent=agents["copywriter"],
            output_json=ContentPayload
        )

        return Crew(
            agents=list(agents.values()),
            tasks=[investigate_task, risk_task, content_task],
            process=Process.sequential,
            verbose=True
        )
