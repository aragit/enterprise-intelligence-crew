<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.12+"/>
  <img src="https://img.shields.io/badge/CrewAI-1.14+-orange?style=for-the-badge&logo=crewai" alt="CrewAI"/>
  <img src="https://img.shields.io/badge/FastAPI-0.115+-00a393?style=for-the-badge&logo=fastapi" alt="FastAPI"/>
  <img src="https://img.shields.io/badge/Pydantic_V2-2.7+-E92063?style=for-the-badge&logo=pydantic" alt="Pydantic V2"/>
  <img src="https://img.shields.io/badge/Ollama-native-000000?style=for-the-badge&logo=ollama" alt="Ollama"/>
  <img src="https://img.shields.io/badge/OpenAI-compatible-412991?style=for-the-badge&logo=openai" alt="OpenAI"/>
  <img src="https://img.shields.io/badge/ChromaDB-0.5+-yellow?style=for-the-badge&logo=chromadb" alt="ChromaDB"/>
  <img src="https://img.shields.io/badge/Web_Search-DuckDuckGo-FF6600?style=for-the-badge&logo=duckduckgo" alt="DuckDuckGo"/>
  <img src="https://img.shields.io/badge/Scraping-Trafilatura-2C8EBB?style=for-the-badge&logo=python" alt="Trafilatura"/>
  <img src="https://img.shields.io/badge/Metrics-Prometheus-E6522C?style=for-the-badge&logo=prometheus" alt="Prometheus"/>
  <img src="https://img.shields.io/badge/Logging-Loguru-30A5DC?style=for-the-badge&logo=python" alt="Loguru"/>
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

All LLM inference runs **locally** via [Ollama](https://ollama.ai) — no API keys, no cloud dependency, no data egress. A **dual-gate risk validation system** enforces quality guardrails between stages: a lightweight pure-Python gate validates `TrendPayload` before it reaches the RiskAnalyst, and a [LangGraph](https://langchain-ai.github.io/langgraph/) `StateGraph` gate evaluates `RiskPayload` before content generation. Both gates support feedback injection with circuit-breaker retry logic. [ChromaDB](https://www.trychroma.com/) with sentence transformers persists research for semantic recall.

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
Preemptive mini-crew orchestration. Instead of a single crew.kickoff() monolith, the pipeline runs three isolated 1-agent, 1-task mini-crews sequentially. Each execution is validated by a gate before data flows downstream, preventing malformed outputs from poisoning downstream agents.

| Stage | Agent             | Tools                                    | Output           | Gate                                                        |
| ----- | ----------------- | ---------------------------------------- | ---------------- | ----------------------------------------------------------- |
| 1     | TrendInvestigator | WebSearch, WebScraper, SentimentAnalyzer | `TrendPayload`   | **Gate 1** — Pure Python validator (<1ms)                   |
| 2     | RiskAnalyst       | BiasDetector, Validator                  | `RiskPayload`    | **Gate 2** — LangGraph `StateGraph` (5-node compiled graph) |
| 3     | Copywriter        | SEOOptimizer, Summarizer                 | `ContentPayload` | —                                                           |

<br>

- **Dual-gate validation.** Gate 1 checks TrendPayload bounds (momentum_score 0.0–1.0), non-empty sources, and metrics. Gate 2 evaluates risk_score against threshold 0.7 via a compiled LangGraph loop (analyze → evaluate_risk → [approve|reject]). Both gates support feedback injection: rejected outputs trigger agent rerun with contextual feedback, up to max_iterations=3 before circuit-breaker force-approval.

- **Schema-aware extraction.** _extract_payload() implements a 4-tier fallback (direct Pydantic → CrewOutput traversal → JSON string → raw dict) to handle output format drift across LLM providers.


- **Memory.**  CrewMemory replaces CrewAI's internal memory (which defaults to OpenAI extraction) with ChromaDB + all-MiniLM-L6-v2 embeddings — fully local, zero API keys.

- **API surface.** FastAPI with 6 endpoints (/health, /crew/run, /crew/run/async, /crew/status/{id}, /crew/memory/{topic}, /metrics). Task state persisted via fcntl-locked JSON. Prometheus instrumentation and Loguru rotation included.

**Advantages**

| #  | Advantage                        | Why It Matters                                                                                                                                |
| -- | -------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| 1  | **Fault Isolation**              | Each agent runs in its own mini-crew. A hallucinated TrendPayload cannot silently poison the RiskAnalyst or Copywriter.                       |
| 2  | **Data Integrity by Contract**   | Pydantic gates enforce `TrendPayload`/`RiskPayload` structure before downstream execution. Bad data is caught, not propagated.                |
| 3  | **Provider Agnostic**            | `OllamaNativeLLM` + `OpenAIProvider` + `MockNativeLLM` — swap providers via env var without touching agent code.                              |
| 4  | **Resilient to Output Drift**    | 4-tier `_extract_payload()` handles format variations across Ollama, OpenAI, and CrewAI versions.                                             |
| 5  | **Auditable State Machine**      | `PipelineState` is explicit and mutable. Every feedback injection and retry is logged and traceable — unlike CrewAI's hidden internal memory. |
| 6  | **Autonomous Retry with Safety** | Feedback loops + circuit breaker (`max_iterations=3`) let the system self-correct without infinite loops or human intervention.               |
| 7  | **No Vendor Lock-in on Memory**  | Custom `CrewMemory` replaces CrewAI's default OpenAI-based memory extraction. ChromaDB + sentence-transformers is fully open-source.          |
| 8 | **Cost Control**                 | Local inference eliminates per-token cloud LLM costs.                                                                                         |


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
| `crew_tools` | `crewai` | 7 `BaseTool` wrappers with `TOOL_REGISTRY` mapping |

All tools wrapped with `@retry(max_attempts=3, delay=1.0)` and return `ToolResult(success, data, error, duration_ms)`.

### Data Contracts (Pydantic V2)

Every stage produces a strictly validated payload:

| Contract | Fields | Validation |
|---|---|---|
| `TrendPayload` | `trend_name`, `momentum_score` (0.0–1.0), `extracted_metrics` (dict), `verified_sources` (list[str]) | Regex URL validator (`http(s)://` + valid host); rejects `localhost`, bare IPs; extracts URLs from raw text |
| `RiskPayload` | `is_safe` (bool), `risk_score` (0.0–1.0), `flagged_keywords` (list), `required_revisions` (list) | — |
| `ContentPayload` | `headline`, `body_content`, `metadata_tags` (list[str]) | — |

All payloads use `str` URLs (not `HttpUrl`) to avoid `json.dumps()` serialization issues in CrewAI task storage. URLs are validated via regex with extraction from raw text and placeholder fallback. Each payload has `__version__ = "1.0"`.

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
CrewAI-compatible mock provider that returns valid JSON matching the `response_model` schema (when provided) or a generic valid JSON dict. Used by `_make_llm()` when `LLM_PROVIDER=mock`. Enables full pipeline testing (76 tests) without any LLM connection.

#### 3. Standalone Provider Hierarchy (`src/llm.py`)
Three non-CrewAI providers (`OllamaProvider`, `OpenAIProvider`, `MockProvider`) following the `BaseLLMProvider` interface for direct use outside CrewAI tasks.

### Risk Gate (Dual-Gate System)

The risk validation system consists of two gates with different implementations:

**Gate 1: `run_trend_gate()`** (`src/orchestration/risk_gate.py`)
- Pure Python validator (no LangGraph overhead)
- Checks: `momentum_score` range (0.0–1.0), non-empty `verified_sources`, non-empty `extracted_metrics`, non-empty `trend_name`
- Lightweight: runs in <1ms

**Gate 2: `run_risk_gate()`** (`src/orchestration/risk_gate.py`)
- LangGraph `StateGraph` with 5 nodes: `_analyze → _evaluate → approve | reject → _generate → END`
- Reject loop: risk > threshold (0.7) → feedback + re-evaluate
- `max_iterations` exceeded → force-approve (circuit breaker)

**Feedback Loop:** Both gates return `(decision: str, feedback: list[str])`. On rejection, feedback is injected into the agent's task description via `{feedback}` placeholder in `crew_config.yaml`, triggering a rerun with corrected context.

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
│   │   └── intelligence_crew.py   # 3 mini-crew orchestration, PipelineState, _extract_payload
│   ├── api/
│   │   ├── main.py                # FastAPI app, lifespan, instrumentation
│   │   └── routes.py              # Endpoints: health, crew run, memory, metrics
│   ├── memory/
│   │   └── crew_memory.py         # ChromaDB vector store client
│   ├── orchestration/
│   │   └── risk_gate.py           # Dual-gate: pure Python (Gate 1) + LangGraph StateGraph (Gate 2)
│   ├── schemas/
│   │   └── payloads.py            # Pydantic V2 contracts with URL sanitization + version markers
│   ├── tools/
│   │   ├── web_search.py          # DuckDuckGo search
│   │   ├── web_scraper.py         # Trafilatura HTML extraction
│   │   ├── summarizer.py          # LSA extractive summary (sumy)
│   │   ├── sentiment_analyzer.py  # TextBlob sentiment
│   │   ├── bias_detector.py       # Heuristic bias scan
│   │   ├── validator.py           # Source credibility scoring
│   │   ├── seo_optimizer.py       # Readability, keyword density
│   │   └── crew_tools.py          # 7 CrewAI BaseTool wrappers + TOOL_REGISTRY
│   ├── llm.py                     # OllamaNativeLLM, MockNativeLLM, BaseLLMProvider hierarchy
│   └── config.py                  # Pydantic Settings with env overrides
├── configs/
│   └── crew_config.yaml           # Agent roles, goals, backstories (+ {feedback} placeholder)
├── tests/                         # 76 tests (pytest, 1 skipped — web search needs network)
│   ├── test_gates.py              # Gate 1 + Gate 2 logic + consistency
│   ├── test_payload_extraction.py # 4-tier _extract_payload validation
│   ├── test_pipeline_state.py     # PipelineState serialization
│   ├── test_feedback_injection.py # Feedback loop + sanitization
│   ├── test_dependencies.py       # Dead dependency verification
│   ├── test_mini_crew.py          # Per-agent mini-crew execution
│   ├── test_pipeline_integration.py # Full pipeline + circuit breaker
│   ├── test_gate_consistency.py   # Cross-gate drift detection
│   ├── test_schema_e2e.py         # URL sanitization + bounds
│   ├── test_performance.py        # Latency benchmarks
│   ├── test_concurrency.py        # Thread safety + file locks
│   ├── test_security.py           # Prompt injection + malformed payloads
│   └── test_functional.py         # Subprocess-based functional test
├── AGENTS.md                      # Anchored summary — architecture decisions + test strategy
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
└── requirements.txt               # langchain + langchain-chroma REMOVED
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
# Full suite (76 tests, 1 skipped — web search requires network)
pytest tests/ -v

# With coverage
pytest tests/ --cov=src --cov-report=term-missing

# Specific test categories
pytest tests/test_gates.py -v              # Gate logic
pytest tests/test_payload_extraction.py -v # Schema extraction
pytest tests/test_feedback_injection.py -v # Feedback loop
pytest tests/test_performance.py -v        # Benchmarks
pytest tests/test_security.py -v           # Security hardening
```

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| **Native Ollama adapter** | CrewAI's `OpenAICompatibleCompletion` needs `/v1/chat/completions` (Ollama ≥0.28). `OllamaNativeLLM` uses `/api/chat` — compatible with all versions. |
| **ChromaDB over LanceDB** | Avoids CrewAI's internal memory which defaults to OpenAI for memory extraction. ChromaDB + sentence-transformers is fully local, no API keys. |
| **Agent `memory=False`** | CrewAI internal memory requires `gpt-4o-mini` extraction by default. Our own `CrewMemory` handles persistence. |
| **`str` URLs over `HttpUrl`** | Pydantic's `HttpUrl` breaks `json.dumps()` in CrewAI's task output storage. String URLs with a `field_validator` solve this. |
| **No LiteLLM dependency** | LiteLLM is optional in CrewAI 1.14. We use `httpx` directly — fewer dependencies, simpler debugging. |
| **Preemptive dual-gate validation** | Breaking `crew.kickoff()` into per-agent mini-crews with intermediate gates prevents bad data from flowing downstream. Gate 1 (pure Python) is fast; Gate 2 (LangGraph) is expressive. |
| **Schema-aware `_extract_payload`** | Replaced fragile string-matching (`_parse_crew_output`) with 4-tier Pydantic validation: direct instance → `TaskOutput` → JSON string → raw dict. Handles CrewAI output format variations across models. |
| **Feedback injection via YAML placeholder** | `{feedback}` in `crew_config.yaml` task descriptions allows runtime injection of gate feedback without hardcoding strings in Python. Enables autonomous retry loops. |
| **Dead dependency removal** | Removed `langchain` and `langchain-chroma` (zero imports in codebase). Reduced install time and attack surface. |

---

## License

MIT
