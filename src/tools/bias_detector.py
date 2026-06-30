from __future__ import annotations

from pydantic import BaseModel, Field
from src.tools.base import ToolResult, retry

BIAS_KEYWORDS = {
    "hyperbolic": ["revolutionary", "game-changer", "unprecedented", "never-before", "mind-blowing", "groundbreaking"],
    "emotional": ["shocking", "outrageous", "terrifying", "incredible", "horrifying"],
    "certainty": ["always", "never", "everyone", "nobody", "undoubtedly", "certainly", "guaranteed"],
    "vagueness": ["some people say", "experts claim", "it is believed", "many are saying", "sources say"],
}


class BiasInput(BaseModel):
    text: str = Field(..., min_length=1, description="Text to check for bias")
    source_urls: list[str] = Field(default_factory=list, description="Source URLs for diversity check")


class BiasOutput(BaseModel):
    bias_score: float = 0.0
    flagged_phrases: list[str] = Field(default_factory=list)
    source_diversity_score: float = 1.0
    verdict: str = "low bias"


@retry(max_attempts=3, delay=1.0)
def bias_detector(input_data: BiasInput) -> ToolResult:
    text_lower = input_data.text.lower()
    flagged: list[str] = []
    total_checks = 0
    hits = 0

    for category, keywords in BIAS_KEYWORDS.items():
        for kw in keywords:
            total_checks += 1
            if kw in text_lower:
                hits += 1
                flagged.append(f"[{category}] \"{kw}\"")

    bias_score = min(hits / max(total_checks, 1) * 5, 1.0)

    source_diversity = 1.0
    if input_data.source_urls:
        domains = set()
        for url in input_data.source_urls:
            parts = url.split("/")
            if len(parts) >= 3:
                domains.add(parts[2])
        source_diversity = min(len(domains) / 3.0, 1.0)

    if bias_score > 0.3:
        verdict = "high bias"
    elif bias_score > 0.1:
        verdict = "moderate bias"
    else:
        verdict = "low bias"

    output = BiasOutput(
        bias_score=round(bias_score, 4),
        flagged_phrases=flagged,
        source_diversity_score=round(source_diversity, 4),
        verdict=verdict,
    )
    return ToolResult(success=True, data=output.model_dump())
