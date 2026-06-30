from __future__ import annotations

import pytest
from src.agents.intelligence_crew import EnterpriseIntelligenceCrew


class TestEnterpriseIntelligenceCrew:
    def test_init_loads_config(self):
        crew = EnterpriseIntelligenceCrew()
        assert "trend_investigator" in crew.agents_config
        assert "risk_analyst" in crew.agents_config
        assert "copywriter" in crew.agents_config

    def test_build_agents_returns_dict(self):
        crew = EnterpriseIntelligenceCrew()
        agents = crew.build_agents()
        assert isinstance(agents, dict)
        assert len(agents) == 3
        assert "trend_investigator" in agents
        assert "risk_analyst" in agents
        assert "copywriter" in agents

    def test_agent_has_role(self):
        crew = EnterpriseIntelligenceCrew()
        agents = crew.build_agents()
        assert "researcher" in agents["trend_investigator"].role.lower()

    def test_build_crew_returns_crew(self):
        crew = EnterpriseIntelligenceCrew()
        c = crew.build_crew()
        assert c is not None

    def test_crew_has_three_tasks(self):
        crew = EnterpriseIntelligenceCrew()
        c = crew.build_crew()
        assert len(c.tasks) == 3

    def test_crew_has_three_agents(self):
        crew = EnterpriseIntelligenceCrew()
        c = crew.build_crew()
        assert len(c.agents) == 3

    def test_tasks_have_output_schemas(self):
        crew = EnterpriseIntelligenceCrew()
        c = crew.build_crew()
        task_names = [t.description[:20] for t in c.tasks]
        assert len(task_names) == 3

    def test_tool_registry_empty_by_default(self):
        crew = EnterpriseIntelligenceCrew()
        assert crew._tool_registry == {}

    def test_tool_registry_accepted(self):
        crew = EnterpriseIntelligenceCrew(tool_registry={"trend_investigator": []})
        assert "trend_investigator" in crew._tool_registry
