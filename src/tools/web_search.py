from __future__ import annotations

from pydantic import BaseModel, Field
from src.tools.base import ToolResult, retry


class WebSearchInput(BaseModel):
    query: str = Field(..., description="Search query string")
    max_results: int = Field(default=5, ge=1, le=20)


class WebSearchOutput(BaseModel):
    results: list[dict] = Field(default_factory=list)
    total_found: int = 0


@retry(max_attempts=3, delay=1.0)
def web_search(input_data: WebSearchInput) -> ToolResult:
    from duckduckgo_search import DDGS

    with DDGS() as ddgs:
        results = list(ddgs.text(input_data.query, max_results=input_data.max_results))
    output = WebSearchOutput(
        results=[{"title": r.get("title", ""), "href": r.get("href", ""), "body": r.get("body", "")} for r in results],
        total_found=len(results),
    )
    return ToolResult(success=True, data=output.model_dump())
