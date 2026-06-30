from __future__ import annotations

import re
from urllib.parse import urlparse

from pydantic import BaseModel, Field, HttpUrl
from src.tools.base import ToolResult, retry

TRUSTED_TLDS = {".edu", ".gov", ".org"}
KNOWN_CREDIBLE = {
    "reuters.com", "ap.org", "bbc.com", "nature.com", "science.org",
    "ieee.org", "acm.org", "arxiv.org", "who.int", "worldbank.org",
}


class ValidatorInput(BaseModel):
    url: HttpUrl = Field(..., description="URL to validate")
    text_content: str | None = Field(default=None, description="Optional text from the URL")


class ValidatorOutput(BaseModel):
    is_valid: bool = False
    domain: str = ""
    tld_trusted: bool = False
    domain_credible: bool = False
    has_content: bool = False
    issues: list[str] = Field(default_factory=list)


@retry(max_attempts=3, delay=1.0)
def validator(input_data: ValidatorInput) -> ToolResult:
    parsed = urlparse(str(input_data.url))
    domain = parsed.netloc.lower()
    tld_match = re.search(r"\.[a-z]{2,}$", domain)

    tld_trusted = False
    if tld_match:
        tld = tld_match.group()
        tld_trusted = tld in TRUSTED_TLDS

    domain_credible = domain in KNOWN_CREDIBLE or any(
        domain.endswith(f".{credible}") or domain == credible for credible in KNOWN_CREDIBLE
    )

    has_content = bool(input_data.text_content and len(input_data.text_content.strip()) > 50)

    issues: list[str] = []
    if not parsed.scheme or not parsed.netloc:
        issues.append("Malformed URL")
    if not domain_credible and not tld_trusted:
        issues.append("Unknown or low-credibility domain")
    if not has_content:
        issues.append("URL returned insufficient content")

    output = ValidatorOutput(
        is_valid=(domain_credible or tld_trusted) and has_content,
        domain=domain,
        tld_trusted=tld_trusted,
        domain_credible=domain_credible,
        has_content=has_content,
        issues=issues,
    )
    return ToolResult(success=True, data=output.model_dump())
