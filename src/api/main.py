from __future__ import annotations

import os
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from loguru import logger
from prometheus_fastapi_instrumentator import Instrumentator

from src.api.routes import router
from src.agents.intelligence_crew import check_llm_health
from src.config import settings

# Disable CrewAI telemetry to avoid timeout delays
os.environ["CREWAI_TELEMETRY_OPT_OUT"] = "true"

os.makedirs("logs", exist_ok=True)
logger.remove()
logger.add(sys.stderr, level=settings.log_level)
logger.add("logs/crew_{time:YYYY-MM-DD}.log", rotation="1 day", retention="7 days", level="INFO")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Enterprise Intelligence Crew API starting")
    logger.info("LLM provider: %s", settings.llm_provider)
    health_msg = check_llm_health()
    if health_msg:
        logger.warning("LLM health check: %s", health_msg)
    else:
        logger.info("LLM health check: healthy")
    yield
    logger.info("Enterprise Intelligence Crew API shutting down")


app = FastAPI(
    title="Enterprise Intelligence Crew API",
    version="1.0.0",
    description="Multi-agent orchestration pipeline for enterprise trend intelligence",
    lifespan=lifespan,
)

app.include_router(router)
Instrumentator().instrument(app).expose(app)


if __name__ == "__main__":
    import socket

    port = 8000
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(("127.0.0.1", port))
    sock.close()
    if result == 0:
        logger.warning("Port %d already in use, trying %d", port, port + 1)
        port += 1

    import uvicorn

    uvicorn.run("src.api.main:app", host="0.0.0.0", port=port)
