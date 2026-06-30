from __future__ import annotations

import textstat

from pydantic import BaseModel, Field
from src.tools.base import ToolResult, retry


class SEOInput(BaseModel):
    text: str = Field(..., min_length=1, description="Text content to analyze")
    target_keywords: list[str] = Field(default_factory=list, description="Expected keywords")


class SEOOutput(BaseModel):
    readability_score: float = 0.0
    grade_level: str = ""
    word_count: int = 0
    keyword_density: dict[str, float] = Field(default_factory=dict)
    suggestions: list[str] = Field(default_factory=list)


@retry(max_attempts=3, delay=1.0)
def seo_optimizer(input_data: SEOInput) -> ToolResult:
    text = input_data.text
    words = text.split()
    word_count = len(words)
    readability = textstat.flesch_reading_ease(text)

    if readability >= 80:
        grade = "very easy"
    elif readability >= 60:
        grade = "easy"
    elif readability >= 40:
        grade = "moderate"
    elif readability >= 20:
        grade = "difficult"
    else:
        grade = "very difficult"

    keyword_density: dict[str, float] = {}
    suggestions: list[str] = []
    text_lower = text.lower()

    for kw in input_data.target_keywords:
        if not kw:
            continue
        count = text_lower.count(kw.lower())
        density = round(count / max(word_count, 1) * 100, 2)
        keyword_density[kw] = density
        if density < 0.5:
            suggestions.append(f"Keyword \"{kw}\" appears only {count} time(s) (density: {density}%) — consider increasing")
        elif density > 5.0:
            suggestions.append(f"Keyword \"{kw}\" appears {count} time(s) (density: {density}%) — may be over-optimized")

    if readability < 40:
        suggestions.append("Readability is low — consider shorter sentences and simpler words")
    if word_count < 150:
        suggestions.append("Content is too short for SEO (< 150 words)")

    output = SEOOutput(
        readability_score=round(readability, 2),
        grade_level=grade,
        word_count=word_count,
        keyword_density=keyword_density,
        suggestions=suggestions,
    )
    return ToolResult(success=True, data=output.model_dump())
