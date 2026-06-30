<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.12+"/>
  <img src="https://img.shields.io/badge/CrewAI-1.14+-orange?style=for-the-badge&logo=crewai" alt="CrewAI"/>
  <img src="https://img.shields.io/badge/FastAPI-0.115+-00a393?style=for-the-badge&logo=fastapi" alt="FastAPI"/>
  <img src="https://img.shields.io/badge/LangGraph-1.2+-purple?style=for-the-badge&logo=langgraph" alt="LangGraph"/>
  <img src="https://img.shields.io/badge/Pydantic_V2-2.7+-E92063?style=for-the-badge&logo=pydantic" alt="Pydantic V2"/>
  <br/>
  <img src="https://img.shields.io/badge/Ollama-native-000000?style=for-the-badge&logo=ollama" alt="Ollama"/>
  <img src="https://img.shields.io/badge/OpenAI-compatible-412991?style=for-the-badge&logo=openai" alt="OpenAI"/>
  <img src="https://img.shields.io/badge/vLLM-supported-8A2BE2?style=for-the-badge&logo=nvidia" alt="vLLM"/>
  <br/>
  <img src="https://img.shields.io/badge/ChromaDB-1.1+-yellow?style=for-the-badge&logo=chromadb" alt="ChromaDB"/>
  <img src="https://img.shields.io/badge/Sentence_Transformers-3.0+-FF6F00?style=for-the-badge&logo=huggingface" alt="Sentence Transformers"/>
  <br/>
  <img src="https://img.shields.io/badge/Web_Search-DuckDuckGo-FF6600?style=for-the-badge&logo=duckduckgo" alt="DuckDuckGo"/>
  <img src="https://img.shields.io/badge/Scraping-Trafilatura-2C8EBB?style=for-the-badge&logo=python" alt="Trafilatura"/>
  <img src="https://img.shields.io/badge/NLP-TextBlob_VADER_sumy-4CAF50?style=for-the-badge" alt="NLP Stack"/>
  <br/>
  <img src="https://img.shields.io/badge/Metrics-Prometheus-E6522C?style=for-the-badge&logo=prometheus" alt="Prometheus"/>
  <img src="https://img.shields.io/badge/Logging-Loguru-30A5DC?style=for-the-badge&logo=python" alt="Loguru"/>
  <img src="https://img.shields.io/badge/Tracing-OpenTelemetry-000000?style=for-the-badge&logo=opentelemetry" alt="OpenTelemetry"/>
  <br/>
  <img src="https://img.shields.io/badge/tests-72%20passing-brightgreen?style=for-the-badge&logo=pytest" alt="72 tests passing"/>
  <img src="https://img.shields.io/badge/Docker-ready-2496ED?style=for-the-badge&logo=docker" alt="Docker ready"/>
  <img src="https://img.shields.io/badge/license-MIT-blue?style=for-the-badge" alt="MIT license"/>
</p>

<br/>

<!-- REPO CARD -->
<table align="center">
  <tr>
    <td width="320" valign="top" align="center">
      <a href="https://github.com/aragit/enterprise-intelligence-crew">
        <img src="https://img.shields.io/badge/Enterprise_Intelligence_Crew-121013?style=for-the-badge&logo=github&logoColor=white" alt="Enterprise Intelligence Crew"/>
      </a>
    </td>
    <td width="460" valign="top">
      <table>
        <tr><td><strong>Autonomous content lifecycle platform</strong></td></tr>
        <tr><td><code>CrewAI</code> • <code>LangChain/LangGraph</code> • <code>Pydantic</code> • <code>ChromaDB</code> &nbsp; <img src="https://img.shields.io/badge/🟢_Active-2EA043?style=flat"/> <img src="https://img.shields.io/badge/Multi--agent_Orchestration-8A2BE2?style=flat"/></td></tr>
        <tr><td><strong>Architecture insight</strong><br/>
          • Hierarchical multi-agent workflow with structured delegation<br/>
          • Specialized agents for trend research, risk analysis, and content generation<br/>
          • Memory synchronization across task pipelines<br/>
          • Enforced schema validity via Pydantic output contracts
        </td></tr>
      </table>
    </td>
  </tr>
</table>
<br/>

<h1 align="center">Enterprise Intelligence Crew</h1>
<p align="center">
  <em>Multi-agent orchestration pipeline for autonomous enterprise trend intelligence — research, risk-assess, and generate production-grade content, fully local.</em>
</p>

<p align="center">
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-architecture">Architecture</a> •
  <a href="#-api">API</a> •
  <a href="#-configuration">Configuration</a> •
  <a href="#-project-structure">Structure</a> •
  <a href="#-docker">Docker</a>
</p>

---

## Overview

Enterprise Intelligence Crew is a **three-agent sequential pipeline** powered by [CrewAI](https://crewai.com) that takes a user query through the full intelligence lifecycle:

1. **Trend Investigation** — Web search, scrape, sentiment analysis
2. **Risk Assessment** — Bias detection, compliance scoring
3. **Content Generation** — SEO optimization, summarization

All LLM inference runs **locally** via [Ollama](https://ollama.ai) — no API keys, no cloud dependency, no data egress. A LangGraph-powered **risk gate** enforces quality guardrails between stages, and [ChromaDB](https://www.trychroma.com/) with sentence transformers persists research for semantic recall.

---

## Quick Start

### Prerequisites

- Python 3.12+
- [Ollama](https://ollama.ai) — with at least one model pulled:
  ```bash
  ollama pull qwen2.5:1.5b
  ollama serve
  ```
- _No cloud API keys required._

### Install & Run

```bash
git clone https://github.com/aragit/enterprise-intelligence-crew.git
cd enterprise-intelligence-crew
pip install -r requirements.txt

# CLI mode
python main.py

# API server
python3 -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000

# With mock LLM (no Ollama needed — for testing)
LLM_PROVIDER=mock python main.py

# Run tests
python3 -m pytest tests/ -v
```

---

## Architecture

```
                                    ┌──────────────────┐
                                    │   🧑‍💻 QUERY      │
                                    │  (research topic) │
                                    └────────┬─────────┘
                                             │
                    ┌──────────────────────────────────────────┐
                    │         CREWAI PIPELINE                  │
                    │        (sequential, 3 agents)            │
                    │                                          │
                    │  ┌──────────────────────┐                │
                    │  │ TrendInvestigator    │                │
                    │  │  • Web Search (Duck) │                │
                    │  │  • Web Scraper       │  TrendPayload  │
                    │  │  • Sentiment Analyzer│───────▶        │
                    │  └──────────┬───────────┘                │
                    │             │                            │
                    │             ▼                            │
                    │  ┌──────────────────────┐                │
                    │  │ RiskAnalyst          │                │
                    │  │  • Bias Detector     │  RiskPayload   │
                    │  │  • Source Validator  │───────▶        │
                    │  └──────────┬───────────┘                │
                    │             │                            │
                    │         ┌───┴───┐                        │
                    │         │ RISK  │◀── max_iter guard     │
                    │         │ GATE  │── approve/reject       │
                    │         └───┬───┘                        │
                    │             │ (approved)                 │
                    │             ▼                            │
                    │  ┌──────────────────────┐                │
                    │  │ Copywriter           │                │
                    │  │  • SEO Optimizer     │ ContentPayload │
                    │  │  • Summarizer        │───────▶        │
                    │  └──────────────────────┘                │
                    └──────────────────────────────────────────┘
                                             │
                                             ▼
                              ┌──────────────────────────────┐
                              │      CHROMADB MEMORY          │
                              │  (all-MiniLM-L6-v2 embeddings)│
                              │  Store & semantic retrieval   │
                              └──────────────────────────────┘
```

### Data Contracts

Every agent stage produces a validated Pydantic V2 payload:

| Contract | Fields | Produced By |
|---|---|---|
| `TrendPayload` | `trend_name`, `momentum_score`, `extracted_metrics`, `verified_sources` | TrendInvestigator |
| `RiskPayload` | `is_safe`, `risk_score`, `flagged_keywords`, `required_revisions` | RiskAnalyst |
| `ContentPayload` | `headline`, `body_content`, `metadata_tags` | Copywriter |

### LLM Layer

Two paths, selected by `LLM_PROVIDER`:

| Provider | Backend | When To Use |
|---|---|---|
| `ollama` (default) | `OllamaNativeLLM` — calls `/api/chat` via httpx | Local, air-gapped, no API keys |
| `openai` | `crewai.LLM` — standard OpenAI SDK | Cloud, GPT-4o, etc. |
| `mock` | `crewai.LLM` with fake key | Development / CI / testing |

> **Why not LiteLLM?** CrewAI 1.14 routes `ollama/` models through `OpenAICompatibleCompletion` which requires the `/v1/chat/completions` endpoint — unavailable on Ollama ≤0.24. `OllamaNativeLLM` bypasses this entirely by calling the native `/api/chat` endpoint directly, supporting **any** Ollama version.

### Risk Gate

The `RiskGate` (`src/orchestration/risk_gate.py`) is a state machine:

```
analyze → evaluate → approve | reject → generate
```

- Risk > 0.7 → **reject** (flagged to user)
- Risk ≤ 0.7 → **approve** (proceeds to Copywriter)
- `max_iterations` exceeded → force-approve (circuit breaker)

---

## API

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Provider status, model, health message |
| `POST` | `/crew/run` | Execute pipeline (synchronous) |
| `POST` | `/crew/run/async` | Execute pipeline (async, returns `task_id`) |
| `GET` | `/crew/status/{task_id}` | Poll async result |
| `GET` | `/crew/memory/{topic}` | ChromaDB semantic search |
| `GET` | `/metrics` | Prometheus scrape endpoint |

### Example

```bash
curl -s -X POST http://localhost:8000/crew/run \
  -H "Content-Type: application/json" \
  -d '{"query_context": "Edge AI adoption in healthcare 2026"}' | jq .
```

---

## Configuration

All settings via environment variables or `.env`:

| Variable | Default | Options |
|---|---|---|
| `LLM_PROVIDER` | `ollama` | `ollama`, `openai`, `mock` |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Any Ollama host |
| `OLLAMA_MODEL` | `qwen2.5:1.5b` | Any pulled model |
| `OPENAI_API_KEY` | — | Required for provider=openai |
| `OPENAI_MODEL` | `gpt-4o-mini` | Any OpenAI model |
| `CHROMADB_PATH` | `./data/chromadb` | Persistence path |
| `LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING` |

---

## Project Structure

```
├── src/
│   ├── agents/
│   │   └── intelligence_crew.py   # Agent/task definitions, health check, _make_llm()
│   ├── api/
│   │   ├── main.py                # FastAPI app, lifespan, instrumentation
│   │   └── routes.py              # Endpoints: health, crew run, memory, metrics
│   ├── memory/
│   │   └── crew_memory.py         # ChromaDB vector store client
│   ├── orchestration/
│   │   └── risk_gate.py           # State machine risk evaluator
│   ├── schemas/
│   │   └── payloads.py            # Pydantic V2 contracts: Trend, Risk, Content
│   ├── tools/
│   │   ├── web_search.py          # DuckDuckGo search
│   │   ├── web_scraper.py         # Trafilatura HTML extraction
│   │   ├── summarizer.py          # LSA extractive summary (sumy)
│   │   ├── sentiment_analyzer.py  # TextBlob + VADER sentiment
│   │   ├── bias_detector.py       # Heuristic political/commercial bias
│   │   ├── validator.py           # Source credibility scoring
│   │   └── seo_optimizer.py       # Readability, keyword density, suggestions
│   ├── llm.py                     # OllamaNativeLLM — native /api/chat adapter
│   ├── llm_factory.py             # Mock provider for testing/development
│   └── config.py                  # Pydantic Settings with env overrides
├── configs/
│   └── crew_config.yaml           # Agent roles, goals, backstories
├── tests/                         # 72 tests (pytest)
├── AGENTS.md                      # Anchored summary — architecture decisions
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
└── requirements.txt
```

---

## Docker

```bash
docker compose up --build
```

Pulls the model inside the container:

```bash
docker compose exec ollama ollama pull qwen2.5:1.5b
```

---

## Testing

```bash
# Full suite (72 tests, 1 skipped — web search requires network)
pytest tests/ -v

# With coverage
pytest tests/ --cov=src --cov-report=term-missing
```

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| **Native Ollama adapter** | CrewAI's `OpenAICompatibleCompletion` needs `/v1/chat/completions` (Ollama ≥0.28). `OllamaNativeLLM` uses `/api/chat` — compatible with all versions. |
| **ChromaDB over LanceDB** | Avoids CrewAI's internal memory which defaults to OpenAI for memory extraction. ChromaDB + sentence-transformers is fully local, no API keys. |
| **Agent `memory=False`** | CrewAI internal memory requires `gpt-4o-mini` extraction by default. Our own `CrewMemory` handles persistence. |
| **`HttpUrl` → `str` + validator** | Pydantic's `HttpUrl` breaks `json.dumps()` in CrewAI's task output storage. String URLs with a `field_validator` solve this. |
| **No LiteLLM dependency** | LiteLLM is optional in CrewAI 1.14. We use `httpx` directly — fewer dependencies, simpler debugging. |

---

## License

MIT
