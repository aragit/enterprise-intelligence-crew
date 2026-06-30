from __future__ import annotations

import asyncio
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException
from loguru import logger
from pydantic import BaseModel, Field

from src.agents.intelligence_crew import EnterpriseIntelligenceCrew, check_llm_health
from src.memory.crew_memory import CrewMemory
from src.config import settings

router = APIRouter()

_task_store: dict[str, dict[str, Any]] = {}
_memory: CrewMemory | None = None


def _get_memory() -> CrewMemory:
    global _memory
    if _memory is None:
        _memory = CrewMemory()
    return _memory


class CrewRunRequest(BaseModel):
    query_context: str = Field(
        default="Next-generation neuro-symbolic agentic AI system designs for clinical multi-drug validation pipelines.",
        min_length=1,
    )


class CrewRunResponse(BaseModel):
    task_id: str
    status: str
    result: dict | None = None


@router.get("/health")
async def health():
    llm_health = check_llm_health()
    return {
        "status": "ok" if llm_health is None else "degraded",
        "llm_provider": settings.llm_provider,
        "llm_model": settings.ollama_model if settings.llm_provider == "ollama" else settings.openai_model,
        "llm_health": llm_health or "healthy",
    }


@router.post("/crew/run", response_model=CrewRunResponse)
async def crew_run(request: CrewRunRequest):
    task_id = str(uuid.uuid4())
    _task_store[task_id] = {"status": "running", "result": None}

    health_msg = check_llm_health()
    if health_msg:
        _task_store[task_id] = {"status": "failed", "result": None}
        raise HTTPException(status_code=503, detail=health_msg)

    try:
        loop = asyncio.get_event_loop()
        logger.info("Crew run sync: task_id=%s query_context='%s'", task_id, request.query_context[:60])
        result = await loop.run_in_executor(None, _execute_crew, request.query_context)
        _task_store[task_id] = {"status": "completed", "result": result}
        logger.info("Crew run sync completed: task_id=%s", task_id)
        return CrewRunResponse(task_id=task_id, status="completed", result=result)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Crew run sync failed: task_id=%s error=%s", task_id, e)
        _task_store[task_id] = {"status": "failed", "result": None}
        raise HTTPException(status_code=500, detail=f"Crew execution failed: {e}")


@router.post("/crew/run/async", response_model=CrewRunResponse)
async def crew_run_async(request: CrewRunRequest):
    task_id = str(uuid.uuid4())
    _task_store[task_id] = {"status": "pending", "result": None}

    health_msg = check_llm_health()
    if health_msg:
        _task_store[task_id] = {"status": "failed", "result": None}
        raise HTTPException(status_code=503, detail=health_msg)

    asyncio.create_task(_background_run(task_id, request.query_context))
    return CrewRunResponse(task_id=task_id, status="pending")


@router.get("/crew/status/{task_id}", response_model=CrewRunResponse)
async def crew_status(task_id: str):
    entry = _task_store.get(task_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    return CrewRunResponse(task_id=task_id, status=entry["status"], result=entry.get("result"))


@router.get("/crew/memory/{topic}")
async def crew_memory(topic: str, k: int = 3):
    mem = _get_memory()
    results = mem.find_similar(topic, k=k)
    return {"topic": topic, "results": results}


@router.get("/metrics")
async def metrics():
    import prometheus_client

    return prometheus_client.generate_latest()


def _execute_crew(query_context: str) -> dict:
    if settings.llm_provider == "mock":
        return {
            "output": f"[mock] Crew run complete for: {query_context[:60]}",
            "trend": {"trend_name": "Mock Trend", "momentum_score": 0.5},
            "risk": {"is_safe": True, "risk_score": 0.1},
            "content": {"headline": "Mock Headline", "body_content": "Mock body.", "metadata_tags": ["mock"]},
        }
    crew = EnterpriseIntelligenceCrew().build_crew()
    output = crew.kickoff(inputs={"query_context": query_context})
    return {"output": str(output)}


async def _background_run(task_id: str, query_context: str):
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, _execute_crew, query_context)
        _task_store[task_id] = {"status": "completed", "result": result}
    except Exception as e:
        _task_store[task_id] = {"status": "failed", "result": {"error": str(e)}}
