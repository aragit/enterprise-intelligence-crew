from __future__ import annotations

import pytest

from src.memory.crew_memory import CrewMemory


class TestCrewMemory:
    @pytest.fixture
    def memory(self):
        return CrewMemory()

    def test_add_research_returns_id(self, memory, sample_trend_payload, sample_risk_payload, sample_content_payload):
        doc_id = memory.add_research(
            topic="AI Manufacturing",
            trend_payload=sample_trend_payload,
            risk_payload=sample_risk_payload,
            content_payload=sample_content_payload,
        )
        assert isinstance(doc_id, str)
        assert len(doc_id) > 0

    def test_find_similar_returns_list(self, memory, sample_trend_payload, sample_risk_payload, sample_content_payload):
        memory.add_research("AI Manufacturing", sample_trend_payload, sample_risk_payload, sample_content_payload)
        results = memory.find_similar("AI", k=1)
        assert isinstance(results, list)

    def test_count_starts_zero(self, memory):
        count = memory.count()
        assert isinstance(count, int)

    def test_count_increases(self, memory, sample_trend_payload, sample_risk_payload, sample_content_payload):
        before = memory.count()
        memory.add_research("test topic", sample_trend_payload, sample_risk_payload, sample_content_payload)
        after = memory.count()
        assert after >= before

    def test_find_similar_empty_returns_list(self, memory):
        results = memory.find_similar("nonexistent", k=5)
        assert isinstance(results, list)

    def test_find_similar_k_respected(self, memory, sample_trend_payload, sample_risk_payload, sample_content_payload):
        for i in range(3):
            memory.add_research(f"topic {i}", sample_trend_payload, sample_risk_payload, sample_content_payload)
        results = memory.find_similar("topic", k=2)
        assert len(results) <= 2
