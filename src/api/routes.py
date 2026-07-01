from __future__ import annotations

import asyncio
import fcntl
import json
import os
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from loguru import logger
from pydantic import BaseModel, Field

from src.agents.intelligence_crew import EnterpriseIntelligenceCrew, check_llm_health
from src.memory.crew_memory import CrewMemory
from src.config import settings

router = APIRouter()

_memory: CrewMemory | None = None
_TASK_STORE_PATH = Path("data/task_store.json")


def _load_task_store() -> dict[str, dict[str, Any]]:
    if _TASK_STORE_PATH.exists():
        try:
            with open(_TASK_STORE_PATH) as f:
                fcntl.flock(f, fcntl.LOCK_SH)
                data = json.load(f)
                fcntl.flock(f, fcntl.LOCK_UN)
                return data
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _save_task_store(store: dict[str, dict[str, Any]]):
    os.makedirs(str(_TASK_STORE_PATH.parent), exist_ok=True)
    tmp = _TASK_STORE_PATH.with_suffix(".tmp")
    with open(tmp, "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        json.dump(store, f, indent=2, default=str)
        f.flush()
        os.fsync(f.fileno())
        fcntl.flock(f, fcntl.LOCK_UN)
    tmp.replace(_TASK_STORE_PATH)


_task_store: dict[str, dict[str, Any]] = _load_task_store()


def _flush_store():
    _save_task_store(_task_store)


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
        "llm_model": (
            settings.ollama_model
            if settings.llm_provider == "ollama"
            else settings.openai_model
        ),
        "llm_health": llm_health or "healthy",
    }


@router.post("/crew/run", response_model=CrewRunResponse)
async def crew_run(request: CrewRunRequest):
    task_id = str(uuid.uuid4())
    _task_store[task_id] = {"status": "running", "result": None}
    _flush_store()

    health_msg = check_llm_health()
    if health_msg:
        _task_store[task_id] = {"status": "failed", "result": None}
        _flush_store()
        raise HTTPException(status_code=503, detail=health_msg)

    try:
        loop = asyncio.get_event_loop()
        logger.info(
            "Crew run sync: task_id=%s query='%s'",
            task_id,
            request.query_context[:60],
        )
        result = await loop.run_in_executor(
            None, _execute_crew, request.query_context
        )
        _task_store[task_id] = {"status": "completed", "result": result}
        _flush_store()
        logger.info("Crew run sync completed: task_id=%s", task_id)
        return CrewRunResponse(
            task_id=task_id, status="completed", result=result
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Crew run sync failed: task_id=%s error=%s", task_id, e)
        _task_store[task_id] = {"status": "failed", "result": None}
        _flush_store()
        raise HTTPException(
            status_code=500, detail=f"Crew execution failed: {e}"
        )


@router.post("/crew/run/async", response_model=CrewRunResponse)
async def crew_run_async(request: CrewRunRequest):
    task_id = str(uuid.uuid4())
    _task_store[task_id] = {"status": "pending", "result": None}
    _flush_store()

    health_msg = check_llm_health()
    if health_msg:
        _task_store[task_id] = {"status": "failed", "result": None}
        _flush_store()
        raise HTTPException(status_code=503, detail=health_msg)

    asyncio.create_task(_background_run(task_id, request.query_context))
    return CrewRunResponse(task_id=task_id, status="pending")


@router.get("/crew/status/{task_id}", response_model=CrewRunResponse)
async def crew_status(task_id: str):
    entry = _task_store.get(task_id)
    if entry is None:
        raise HTTPException(
            status_code=404, detail=f"Task {task_id} not found"
        )
    return CrewRunResponse(
        task_id=task_id,
        status=entry["status"],
        result=entry.get("result"),
    )


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
            "trend": {
                "trend_name": "Mock Trend",
                "momentum_score": 0.5,
            },
            "risk": {"is_safe": True, "risk_score": 0.1},
            "content": {
                "headline": "Mock Headline",
                "body_content": "Mock body.",
                "metadata_tags": ["mock"],
            },
            "gate_decision": "approve",
            "gate_feedback": [],
        }
    result = EnterpriseIntelligenceCrew().run_pipeline(query_context)
    return result


async def _background_run(task_id: str, query_context: str):
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, _execute_crew, query_context
        )
        _task_store[task_id] = {"status": "completed", "result": result}
        _flush_store()
    except Exception as e:
        _task_store[task_id] = {
            "status": "failed",
            "result": {"error": str(e)},
        }
        _flush_store()
