<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.12+"/>
  <img src="https://img.shields.io/badge/CrewAI-1.14+-orange?style=for-the-badge&logo=crewai" alt="CrewAI"/>
  <img src="https://img.shields.io/badge/FastAPI-0.115+-00a393?style=for-the-badge&logo=fastapi" alt="FastAPI"/>
  <img src="https://img.shields.io/badge/LangGraph-1.2+-purple?style=for-the-badge&logo=langgraph" alt="LangGraph"/>
  <img src="https://img.shields.io/badge/Pydantic_V2-2.7+-E92063?style=for-the-badge&logo=pydantic" alt="Pydantic V2"/>
  <img src="https://img.shields.io/badge/Ollama-native-000000?style=for-the-badge&logo=ollama" alt="Ollama"/>
  <img src="https://img.shields.io/badge/OpenAI-compatible-412991?style=for-the-badge&logo=openai" alt="OpenAI"/>
  <img src="https://img.shields.io/badge/vLLM-supported-8A2BE2?style=for-the-badge&logo=nvidia" alt="vLLM"/>
  <img src="https://img.shields.io/badge/ChromaDB-1.1+-yellow?style=for-the-badge&logo=chromadb" alt="ChromaDB"/>
  <img src="https://img.shields.io/badge/Web_Search-DuckDuckGo-FF6600?style=for-the-badge&logo=duckduckgo" alt="DuckDuckGo"/>
  <img src="https://img.shields.io/badge/Scraping-Trafilatura-2C8EBB?style=for-the-badge&logo=python" alt="Trafilatura"/>
  <img src="https://img.shields.io/badge/Metrics-Prometheus-E6522C?style=for-the-badge&logo=prometheus" alt="Prometheus"/>
  <img src="https://img.shields.io/badge/Logging-Loguru-30A5DC?style=for-the-badge&logo=python" alt="Loguru"/>
  <img src="https://img.shields.io/badge/Tracing-OpenTelemetry-000000?style=for-the-badge&logo=opentelemetry" alt="OpenTelemetry"/>
 <img src="https://img.shields.io/badge/Docker-ready-2496ED?style=for-the-badge&logo=docker" alt="Docker ready"/>
 </p>


<h1 align="center">Enterprise Intelligence Crew</h1>
<p align="center">
  <b>Multi-agent orchestration pipeline for autonomous enterprise trend intelligence — research, risk-assess, and generate production-grade content.</b>
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

A **three-agent sequential CrewAI pipeline** with a **LangGraph StateGraph risk gate** that takes a user query through the full intelligence lifecycle — from live web research to risk-validated, SEO-optimized content — entirely on local infrastructure.

### Tool Arsenal 

| Tool | Library | What It Does |
|---|---|---|
| `web_search` | `duckduckgo_search` | DDGS text search, returns title/href/body |
| `web_scraper` | `trafilatura` + `requests` | HTML extraction, title, word count |
| `sentiment_analyzer` | `textblob` | Polarity (-1 to 1), subjectivity, label |
| `bias_detector` | Heuristic | 4-category keyword scan (hyperbolic, emotional, certainty, vagueness) + source domain diversity score |
| `validator` | `urllib.parse` + regex | TLD trust (`.edu`, `.gov`, `.org`), known-credible domain whitelist, content length check |
| `seo_optimizer` | `textstat` | Flesch readability, grade level, keyword density, optimization suggestions |
| `summarizer` | `sumy` (LSA) | Extractive summarization, configurable sentence count |

All tools wrapped with `@retry(max_attempts=3, delay=1.0)` and return `ToolResult(success, data, error, duration_ms)`.


### Data Contracts (Pydantic V2)

Every stage produces a strictly validated payload:

| Contract | Fields | Validation |
|---|---|---|
| `TrendPayload` | `trend_name`, `momentum_score` (0.0–1.0), `extracted_metrics` (dict), `verified_sources` (list[str]) | Regex URL validator (`http(s)://` + valid host); rejects `localhost`, bare IPs |
| `RiskPayload` | `is_safe` (bool), `risk_score` (0.0–1.0), `flagged_keywords` (list), `required_revisions` (list) | — |
| `ContentPayload` | `headline`, `body_content`, `metadata_tags` (list[str]) | — |



### LLM Layer 

#### 1. `OllamaNativeLLM` (`src/llm.py`)
Custom provider extending `crewai.llms.base_llm.BaseLLM` that calls Ollama's **native `/api/chat`** endpoint directly via `httpx`.

**Why native?** CrewAI 1.14 routes `ollama/` models through `OpenAICompatibleCompletion` which requires `/v1/chat/completions` — unavailable on Ollama ≤0.24. This adapter bypasses that entirely.

| Feature | Support |
|---|---|
| Sync (`call`) / Async (`acall`) | ✅ |
| Tool calling | ✅ |
| Structured output (`response_model`) | ✅ |
| Stop words | ✅ |
| Streaming | ✅ |
| Usage extraction (`prompt_eval_count` / `eval_count`) | ✅ |
| Health probes (`list_available_models`, `check_model_exists`) | ✅ |
| Context window | 131,072 tokens |

#### 2. `MockNativeLLM` (`src/llm.py`)
CrewAI-compatible mock provider that returns valid JSON matching the `response_model` schema (when provided) or a generic valid JSON dict. Used by `_make_llm()` when `LLM_PROVIDER=mock`. Enables full pipeline testing without any LLM connection.

In addition, three non-CrewAI providers (`OllamaProvider`, `OpenAIProvider`, `MockProvider`) following the `BaseLLMProvider` interface in `src/llm.py` cover direct use outside CrewAI.

### Risk Gate

The `RiskGate` (`src/orchestration/risk_gate.py`) is a LangGraph `StateGraph`:

```
_analyze → _evaluate → approve | reject (loop back) → _generate → END
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

## Configuration (`src/config.py`)

Pydantic Settings with `.env` override:

| Variable | Default | Options |
|---|---|---|
| `LLM_PROVIDER` | `ollama` | `ollama`, `openai`, `mock` |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Any Ollama host |
| `OLLAMA_MODEL` | `qwen2.5:1.5b` | Any pulled model |
| `OPENAI_API_KEY` | — | Required for `openai` |
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
│   │   └── risk_gate.py           # LangGraph StateGraph risk evaluator
│   ├── schemas/
│   │   └── payloads.py            # Pydantic V2 contracts: Trend, Risk, Content
│   ├── tools/
│   │   ├── web_search.py          # DuckDuckGo search
│   │   ├── web_scraper.py         # Trafilatura HTML extraction
│   │   ├── summarizer.py          # LSA extractive summary (sumy)
│   │   ├── sentiment_analyzer.py  # TextBlob sentiment
│   │   ├── bias_detector.py       # Heuristic political/commercial bias
│   │   ├── validator.py           # Source credibility scoring
│   │   ├── seo_optimizer.py       # Readability, keyword density, suggestions
│   │   └── crew_tools.py          # 7 CrewAI BaseTool wrappers + TOOL_REGISTRY
│   ├── llm.py                     # OllamaNativeLLM, BaseLLMProvider hierarchy, MockNativeLLM
│   └── config.py                  # Pydantic Settings with env overrides
├── configs/
│   └── crew_config.yaml           # Agent roles, goals, backstories
├── tests/                         # 75 tests (pytest, 1 skipped — web search needs network)
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
# Full suite (75 tests, 1 skipped — web search requires network)
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
