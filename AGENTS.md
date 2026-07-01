# Enterprise Intelligence Crew — Anchored Summary

## Architecture

**Preemptive multi-agent pipeline** — 3 CrewAI `Process.sequential` agents running individually with LangGraph risk gates between each step:

```
User Query
    │
    ▼
┌─────────────────────┐
│  TrendInvestigator  │  (3 tools: WebSearch, WebScraper, SentimentAnalyzer)
│  Mini-Crew #1       │  → TrendPayload
└─────────┬───────────┘
          │
     ┌────┴────┐
     │ Gate 1  │  run_trend_gate() — validates momentum_score, sources, metrics
     │ (trend) │
     └────┬────┘
   reject │ approve
   ┌──────┘     │
   ▼            ▼
feedback    ┌──────────────────────┐
(injected   │  RiskAnalyst         │  (2 tools: BiasDetector, Validator)
 via        │  Mini-Crew #2        │  → RiskPayload
 {feedback})└──────────┬───────────┘
   ▲                    │
   │               ┌────┴────┐
   │               │ Gate 2  │  run_risk_gate() — LangGraph StateGraph
   │               │ (risk)  │  5 nodes: analyze → evaluate → approve|reject → generate
   │               └────┬────┘
   │             reject │ approve
   │             ┌──────┘     │
   │             ▼            ▼
   │        feedback      ┌──────────────────────┐
   └──────── (injected)   │  Copywriter          │  (2 tools: SEOOptimizer, Summarizer)
                          │  Mini-Crew #3        │  → ContentPayload
                          └──────────────────────┘
                                     │
                                     ▼
                          ┌──────────────────────┐
                          │  ChromaDB Memory      │
                          └──────────────────────┘
```

- **TrendInvestigator**: 3 CrewAI tools → TrendPayload
- **Gate 1**: `run_trend_gate()` — validates momentum_score range, non-empty sources/metrics, non-empty trend_name
- **RiskAnalyst**: 2 CrewAI tools → RiskPayload
- **Gate 2**: `run_risk_gate()` — LangGraph `StateGraph` (5 nodes) with internal loop + max-iteration guard
- **Copywriter**: 2 CrewAI tools → ContentPayload

All three output contracts defined in `src/schemas/payloads.py` (each with `__version__ = "1.0"`):
- `TrendPayload` — `trend_name`, `momentum_score`, `extracted_metrics`, `verified_sources` (URL sanitizer)
- `RiskPayload` — `is_safe`, `risk_score`, `flagged_keywords`, `required_revisions`
- `ContentPayload` — `headline`, `body_content`, `metadata_tags`

## LLM Integration

### `src/llm.py` — `OllamaNativeLLM`

Custom provider extending `crewai.llms.base_llm.BaseLLM` that calls Ollama's **native** `/api/chat` endpoint. Bypasses CrewAI's `OpenAICompatibleCompletion` (which requires OpenAI-compatible endpoint `/v1/chat/completions` — unavailable on Ollama 0.24.0).

Key features:
- Uses `httpx` directly (no `litellm` dependency)
- Supports messages, tools, structured output, stop words, streaming
- `call()` sync and `acall()` async
- `list_available_models()` / `check_model_exists()` — queries `/api/tags`

### `src/agents/intelligence_crew.py` — Health Check

`check_llm_health()` verifies three conditions:
1. Ollama is reachable (HTTP connection to `/api/tags`)
2. At least one model is installed
3. The configured model (`settings.ollama_model`) exists in the available list

Fails fast with actionable error messages listing available models.

### `src/agents/intelligence_crew.py` — `_make_llm()`

Routes by `settings.llm_provider`:
- **`ollama`** → `OllamaNativeLLM(model=..., base_url=..., temperature=0.7)`
- **`openai`** → `crewai.LLM(model=..., api_key=...)`
- **`mock`** → `MockNativeLLM(model="mock")` — returns valid JSON for structured output, no LLM needed

## Configuration (`src/config.py`)

| Variable | Default | Override |
|---|---|---|
| `LLM_PROVIDER` | `ollama` | `openai`, `mock` |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | env |
| `OLLAMA_MODEL` | `qwen2.5:1.5b` | env |
| `OPENAI_API_KEY` | none | env |
| `OPENAI_MODEL` | `gpt-4o-mini` | env |
| `CHROMADB_PATH` | `./data/chromadb` | env |

## API (`src/api/main.py`)

FastAPI on port 8000 with endpoints:
- `GET /health` — LLM provider status, model, health message
- `POST /crew/run` — synchronous execution
- `POST /crew/run/async` — async (returns `task_id`, poll via `/crew/status/{id}`)
- `GET /crew/status/{task_id}` — poll async result
- `GET /crew/memory/{topic}` — ChromaDB similarity search
- `GET /metrics` — Prometheus metrics

## Memory (`src/memory/crew_memory.py`)

ChromaDB-based (not CrewAI's LanceDB internal memory). Stores research results with embeddings. CrewAI agent `memory=False` to avoid LanceDB/OpenAI dependency.

## Execution Model — Preemptive Per-Agent Mini-Crews

`run_pipeline()` breaks CrewAI's monolithic `Process.sequential` into 3 individual mini-crews (1 agent + 1 task each). Between each mini-crew, a gate evaluates the output before passing it downstream.

### Per-Agent Execution (`_run_agent`)

```python
# Each agent runs in its own mini-crew
trend_result = self._run_agent("trend_investigator", inputs={...})  # Mini-Crew #1
# Gate 1: run_trend_gate(trend_result) → approve | reject
risk_result = self._run_agent("risk_analyst", inputs={...})        # Mini-Crew #2
# Gate 2: run_risk_gate(trend_result, risk_result) → approve | reject
content_result = self._run_agent("copywriter", inputs={...})        # Mini-Crew #3
```

Each `_run_agent()` call:
1. Builds fresh agents via `build_agents()`
2. Creates a single `Task` with the correct `output_json` model
3. Creates a `Crew` with 1 agent + 1 task
4. Calls `crew.kickoff(inputs={...})` with current state (context + feedback)
5. Extracts the Pydantic payload via `_extract_payload()`

### Context Passing

CrewAI's built-in context passing (task N output → task N+1 prompt) is lost with mini-crews. Instead, context is passed explicitly through `kickoff(inputs=...)`:

| Placeholder | Source | Used by |
|---|---|---|
| `{query_context}` | User input | All agents |
| `{trend_context}` | TrendPayload (JSON) | RiskAnalyst, Copywriter |
| `{risk_context}` | RiskPayload (JSON) | Copywriter |
| `{feedback}` | Gate rejection messages | All agents (re-run only) |

### Feedback Loop

When a gate rejects output:
1. Feedback is appended to `PipelineState.feedback`
2. `{feedback}` in the YAML task description is replaced with formatted rejection messages
3. The agent re-runs with the same `query_context` + feedback
4. After `max_iterations` (default 3), force-approve (circuit breaker)

### Pipeline State

`PipelineState` dataclass tracks all mutable state across pipeline steps:

```python
@dataclass
class PipelineState:
    query_context: str
    feedback: list[str]        # Accumulated gate rejection messages
    iteration: int              # Current re-run attempt
    trend_data: dict | None     # Latest TrendPayload
    risk_data: dict | None      # Latest RiskPayload
    content_data: dict | None   # Latest ContentPayload
```

## Data Flow

```
1. TrendInvestigator (Mini-Crew #1) → TrendPayload
2. Gate 1: run_trend_gate() validates trend quality
3. On reject: feedback → re-run TrendInvestigator (up to 3 iterations)
4. RiskAnalyst (Mini-Crew #2) → RiskPayload
5. Gate 2: run_risk_gate() LangGraph evaluates risk score
6. On reject: feedback → re-run RiskAnalyst (up to 3 iterations)
7. Copywriter (Mini-Crew #3) → ContentPayload
8. All stored in ChromaDB CrewMemory
```

## Known Constraints

- Model `qwen2.5:1.5b` is small (1.5B params) — output quality for structured URL validation is limited. Larger model recommended for production.
- Ollama version must support `/api/chat` endpoint (all versions do). OpenAI-compatible endpoint `/v1/chat/completions` is NOT required.
- No `litellm` dependency — uses `httpx` directly for Ollama calls.
- **No `langchain` or `langchain-chroma` dependency** — removed (dead code, zero imports).
- `langgraph` kept for RiskGate `StateGraph` — justified by planned expansion (parallel branches, conditional agent routing).
- Per-agent mini-crews are 3× slower than a single `crew.kickoff()` because CrewAI rebuilds internals for each call. Acceptable for prototype; optimization target for production.

## Run Commands

```bash
# Direct pipeline
python3 main.py

# API server
python3 -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000

# With mock LLM (no Ollama needed)
LLM_PROVIDER=mock python3 main.py

# With different model
OLLAMA_MODEL=tinyllama python3 main.py

# Tests
python3 -m pytest tests/
```
