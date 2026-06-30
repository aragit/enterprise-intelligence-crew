# Enterprise Intelligence Crew — Anchored Summary

## Architecture

Three-agent sequential CrewAI pipeline with manual risk gate:

```
User Query → TrendInvestigator → RiskAnalyst → RiskGate → Copywriter → Output
```

- **TrendInvestigator**: `src/tools/web_search.py` + `src/tools/web_scraper.py` + `src/tools/sentiment_analyzer.py` → searches/scrapes/analyzes → produces `TrendPayload`
- **RiskAnalyst**: `src/tools/bias_detector.py` → detects bias in the trend output → produces `RiskPayload`
- **RiskGate** (`src/risk_gate/risk_gate.py`): State machine (`analyze → evaluate → approve | reject → generate`) checking risk score + max-iteration guard
- **Copywriter**: `src/tools/seo_optimizer.py` + `src/tools/summarizer.py` → generates `ContentPayload`

All three output contracts defined in `src/schemas/payloads.py`:
- `TrendPayload` — `trend_name`, `momentum_score`, `extracted_metrics`, `verified_sources` (str list with URL validator)
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
- **`mock`** → `crewai.LLM(model="gpt-4o-mini", api_key="sk-mock-placeholder")`

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

## Data Flow

```
1. Web Search → scrape top results → sentiment analysis → TrendPayload
2. Bias detection on trend → RiskPayload
3. RiskGate evaluates RiskPayload → approve/reject
4. SEO optimization + summarization → ContentPayload
5. All stored in ChromaDB CrewMemory for future queries
```

## Known Constraints

- Model `qwen2.5:1.5b` is small (1.5B params) — output quality for structured URL validation is limited. Larger model recommended for production.
- Ollama version must support `/api/chat` endpoint (all versions do). OpenAI-compatible endpoint `/v1/chat/completions` is NOT required.
- No `litellm` dependency — uses `httpx` directly for Ollama calls.

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
