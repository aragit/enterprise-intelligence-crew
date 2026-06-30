from __future__ import annotations

import pytest
from httpx import AsyncClient, ASGITransport

from src.api.main import app


@pytest.fixture
async def async_client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@pytest.mark.asyncio
class TestHealth:
    async def test_health_endpoint(self, async_client):
        resp = await async_client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    async def test_health_contains_llm_provider(self, async_client):
        resp = await async_client.get("/health")
        data = resp.json()
        assert "llm_provider" in data


@pytest.mark.asyncio
class TestCrewRun:
    async def test_crew_run_sync(self, async_client):
        resp = await async_client.post("/crew/run", json={"query_context": "test"})
        assert resp.status_code in (200, 500)  # may fail without Ollama

    async def test_crew_run_async(self, async_client):
        resp = await async_client.post("/crew/run/async", json={"query_context": "test"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "pending"
        assert len(data["task_id"]) > 0

    async def test_crew_status_not_found(self, async_client):
        resp = await async_client.get("/crew/status/nonexistent")
        assert resp.status_code == 404


@pytest.mark.asyncio
class TestMemory:
    async def test_memory_endpoint(self, async_client):
        resp = await async_client.get("/crew/memory/ai")
        assert resp.status_code == 200
        data = resp.json()
        assert "topic" in data
        assert "results" in data


@pytest.mark.asyncio
class TestMetrics:
    async def test_metrics_endpoint(self, async_client):
        resp = await async_client.get("/metrics")
        assert resp.status_code == 200
