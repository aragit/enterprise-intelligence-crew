from __future__ import annotations

from pydantic import BaseModel, Field, HttpUrl
from src.tools.base import ToolResult, retry


class WebScraperInput(BaseModel):
    url: HttpUrl = Field(..., description="Target URL to scrape")


class WebScraperOutput(BaseModel):
    title: str = ""
    content: str = ""
    word_count: int = 0


@retry(max_attempts=3, delay=1.0)
def web_scraper(input_data: WebScraperInput) -> ToolResult:
    import trafilatura
    import requests

    resp = requests.get(str(input_data.url), timeout=10)
    resp.raise_for_status()

    content = trafilatura.extract(resp.text)
    title = trafilatura.bare_extraction(resp.text).get("title", "") if content else ""

    output = WebScraperOutput(
        title=title or "",
        content=content or "",
        word_count=len((content or "").split()),
    )
    return ToolResult(success=True, data=output.model_dump())
