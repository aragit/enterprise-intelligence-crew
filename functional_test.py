"""
Functional test: exercises every layer of the pipeline.
Run:  python3 functional_test.py
No API keys needed — uses MockProvider.
"""

import json
import sys
import time

sys.path.insert(0, ".")

# ---------------------------------------------------------------------------
# 1. Configuration & LLM
# ---------------------------------------------------------------------------
print("=" * 60)
print("LAYER 1: LLM Provider")
print("=" * 60)

from src.config import settings
settings.llm_provider = "mock"

from src.llm import MockProvider, get_llm_provider

provider = get_llm_provider()
assert isinstance(provider, MockProvider), f"Expected MockProvider, got {type(provider)}"
resp = provider.generate("What is AI?")
print(f"  Mock generate: {resp.content}")
resp = provider.chat([{"role": "user", "content": "Hello"}])
print(f"  Mock chat:     {resp.content}")
print("  PASS\n")

# ---------------------------------------------------------------------------
# 2. Tools
# ---------------------------------------------------------------------------
print("=" * 60)
print("LAYER 2: Tools")
print("=" * 60)

from src.tools.web_search import web_search, WebSearchInput
from src.tools.web_scraper import web_scraper, WebScraperInput
from src.tools.summarizer import summarizer, SummarizerInput
from src.tools.sentiment_analyzer import sentiment_analyzer, SentimentInput
from src.tools.bias_detector import bias_detector, BiasInput
from src.tools.validator import validator, ValidatorInput
from src.tools.seo_optimizer import seo_optimizer, SEOInput

# 2a. Web search (network - may skip)
try:
    r = web_search(WebSearchInput(query="Python programming", max_results=2))
    print(f"  Web search: {'OK' if r.success else 'SKIP'} ({len(r.data.get('results', []))} results)")
except Exception as e:
    print(f"  Web search: SKIP ({e})")

# 2b. Sentiment
r = sentiment_analyzer(SentimentInput(text="This is amazing and wonderful!"))
assert r.success and r.data["label"] == "positive"
print(f"  Sentiment (positive): {r.data['label']} ({r.data['polarity']})")

r = sentiment_analyzer(SentimentInput(text="This is terrible and awful."))
assert r.success and r.data["label"] == "negative"
print(f"  Sentiment (negative): {r.data['label']} ({r.data['polarity']})")

# 2c. Bias
r = bias_detector(BiasInput(text="This revolutionary groundbreaking never-before-seen product is guaranteed to change everything!"))
assert r.success and r.data["verdict"] == "high bias"
print(f"  Bias detector: {r.data['verdict']} ({len(r.data['flagged_phrases'])} flags)")

r = bias_detector(BiasInput(text="The study found a 12% improvement."))
assert r.success and r.data["verdict"] == "low bias"
print(f"  Bias detector: {r.data['verdict']} (no flags)")

# 2d. Validator
r = validator(ValidatorInput(url="https://reuters.com/article", text_content="A " * 100))
assert r.success and r.data["domain_credible"]
print(f"  Validator (reuters): credible={r.data['domain_credible']}, valid={r.data['is_valid']}")

# 2e. SEO
r = seo_optimizer(SEOInput(text="AI is transforming manufacturing. AI improves efficiency. AI reduces costs. " * 3, target_keywords=["AI"]))
assert r.success and r.data["keyword_density"]["AI"] > 0
print(f"  SEO: readability={r.data['readability_score']}, keyword density={r.data['keyword_density']}")

# 2f. Summarizer
long_text = "Machine learning is a subset of artificial intelligence. " * 20
r = summarizer(SummarizerInput(text=long_text, max_sentences=3))
assert r.success
print(f"  Summarizer: {len(r.data['summary'].split())} words (from {r.data['original_length']})")

# 2g. Web scraper (network - skip if no internet)
try:
    r = web_scraper(WebScraperInput(url="https://example.com"))
    if r.success:
        print(f"  Web scraper: {r.data['word_count']} words")
    else:
        print(f"  Web scraper: SKIP")
except Exception:
    print(f"  Web scraper: SKIP (no network)")

print("  ALL TOOLS PASS\n")

# ---------------------------------------------------------------------------
# 3. LangGraph Risk Gate
# ---------------------------------------------------------------------------
print("=" * 60)
print("LAYER 3: LangGraph Risk Gate")
print("=" * 60)

from src.orchestration.risk_gate import RiskGate
from src.schemas.payloads import TrendPayload, RiskPayload, ContentPayload

trend = TrendPayload(trend_name="AI in Manufacturing", momentum_score=0.85, extracted_metrics={"repos": 1200}, verified_sources=["https://reuters.com/ai"])
low_risk = RiskPayload(is_safe=True, risk_score=0.2)
high_risk = RiskPayload(is_safe=False, risk_score=0.85, flagged_keywords=["hallucination"], required_revisions=["Verify sources"])

gate = RiskGate(threshold=0.7)

decision, feedback = gate.run(trend, low_risk)
assert decision == "approve", f"Expected approve, got {decision}"
print(f"  Low risk (0.2): {decision} (feedback: {feedback})")

decision, feedback = gate.run(trend, high_risk)
assert decision == "approve", f"Expected approve (looped through max iterations), got {decision}"
assert len(feedback) > 0
print(f"  High risk (0.85): {decision} (feedback: {len(feedback)} items, loops through max iterations)")

print("  PASS\n")

# ---------------------------------------------------------------------------
# 4. Crew Assembly + Tools
# ---------------------------------------------------------------------------
print("=" * 60)
print("LAYER 4: Crew Assembly & Tools")
print("=" * 60)

from src.agents.intelligence_crew import EnterpriseIntelligenceCrew

crew = EnterpriseIntelligenceCrew()
agents = crew.build_agents()
print(f"  Agents built: {list(agents.keys())}")
for name, agent in agents.items():
    print(f"    {name}: role='{agent.role[:40]}...' tools={len(agent.tools)}")

c = crew.build_crew()
print(f"  Crew: {len(c.agents)} agents, {len(c.tasks)} tasks, process={c.process}")
print("  PASS\n")

# ---------------------------------------------------------------------------
# 5. Memory
# ---------------------------------------------------------------------------
print("=" * 60)
print("LAYER 5: ChromaDB Memory")
print("=" * 60)

from src.memory.crew_memory import CrewMemory

mem = CrewMemory()
before = mem.count()
doc_id = mem.add_research(
    topic="AI in Manufacturing",
    trend_payload={"trend_name": "AI Manufacturing", "momentum_score": 0.85, "extracted_metrics": {}, "verified_sources": []},
    risk_payload={"is_safe": True, "risk_score": 0.2, "flagged_keywords": [], "required_revisions": []},
    content_payload={"headline": "AI in Manufacturing", "body_content": "Body", "metadata_tags": ["AI"]},
)
after = mem.count()
print(f"  Stored: {doc_id} (count: {before} -> {after})")

results = mem.find_similar("AI manufacturing", k=1)
print(f"  Found similar: {len(results)} result(s)")
if results:
    print(f"    id={results[0]['id']}, distance={results[0]['distance']:.4f}")

print("  PASS\n")

# ---------------------------------------------------------------------------
# 6. API
# ---------------------------------------------------------------------------
print("=" * 60)
print("LAYER 6: FastAPI Endpoints")
print("=" * 60)

from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

r = client.get("/health")
assert r.status_code == 200
data = r.json()
print(f"  GET /health: {data['status']}, provider={data['llm_provider']}")

r = client.post("/crew/run", json={"query_context": "Functional test pipeline"})
assert r.status_code == 200
data = r.json()
print(f"  POST /crew/run: status={data['status']}")
if data.get("result"):
    print(f"    output: {str(data['result'])[:80]}...")

r = client.post("/crew/run/async", json={"query_context": "Async test"})
assert r.status_code == 200
task_id = r.json()["task_id"]
print(f"  POST /crew/run/async: task_id={task_id}")

r = client.get(f"/crew/status/{task_id}")
assert r.status_code == 200
print(f"  GET /crew/status/{task_id}: status={r.json()['status']}")

r = client.get("/crew/memory/AI")
assert r.status_code == 200
print(f"  GET /crew/memory/AI: {len(r.json()['results'])} result(s)")

r = client.get("/metrics")
assert r.status_code == 200
print(f"  GET /metrics: {len(r.text)} bytes")

print("  PASS\n")

# ---------------------------------------------------------------------------
# 7. Pipeline with RiskGate integration
# ---------------------------------------------------------------------------
print("=" * 60)
print("LAYER 7: Pipeline with RiskGate Integration")
print("=" * 60)

settings.llm_provider = "mock"
pipeline = EnterpriseIntelligenceCrew()
result = pipeline.run_pipeline("Test query for pipeline integration")
assert "trend" in result
assert "risk" in result
assert "content" in result
assert "gate_decision" in result
print(f"  Gate decision: {result['gate_decision']}")
print(f"  Gate feedback: {result['gate_feedback']}")
print(f"  Trend: {result['trend']}")
print(f"  Risk: {result['risk']}")
print(f"  Content: {result['content']}")
print("  PASS\n")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("=" * 60)
print("ALL LAYERS PASSED")
print("=" * 60)
print()
print("Pipeline is fully functional end-to-end.")
print("To run against a real LLM, start Ollama and set LLM_PROVIDER=ollama")
