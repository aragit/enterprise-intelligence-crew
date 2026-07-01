from __future__ import annotations

from crewai.tools import BaseTool

from src.tools.web_search import web_search, WebSearchInput
from src.tools.web_scraper import web_scraper, WebScraperInput
from src.tools.sentiment_analyzer import sentiment_analyzer, SentimentInput
from src.tools.bias_detector import bias_detector, BiasInput
from src.tools.validator import validator, ValidatorInput
from src.tools.seo_optimizer import seo_optimizer, SEOInput
from src.tools.summarizer import summarizer, SummarizerInput


def _fmt(result) -> str:
    if result.success:
        import json
        return json.dumps(result.data, indent=2, default=str)
    return f"Error: {result.error}"


class WebSearchTool(BaseTool):
    name: str = "Web Search"
    description: str = "Search the web for current information on a topic. Returns a list of results with titles, URLs, and snippets."

    def _run(self, query: str, max_results: int = 5) -> str:
        return _fmt(web_search(WebSearchInput(query=query, max_results=max_results)))


class WebScraperTool(BaseTool):
    name: str = "Web Scraper"
    description: str = "Scrape and extract the main content from a given URL."

    def _run(self, url: str) -> str:
        return _fmt(web_scraper(WebScraperInput(url=url)))


class SentimentAnalyzerTool(BaseTool):
    name: str = "Sentiment Analyzer"
    description: str = "Analyze the sentiment polarity and subjectivity of a given text."

    def _run(self, text: str) -> str:
        return _fmt(sentiment_analyzer(SentimentInput(text=text)))


class BiasDetectorTool(BaseTool):
    name: str = "Bias Detector"
    description: str = "Analyze text for biased language, hyperbolic claims, emotional manipulation, and source diversity."

    def _run(self, text: str, source_urls: str = "") -> str:
        urls = [u.strip() for u in source_urls.split(",") if u.strip()]
        return _fmt(bias_detector(BiasInput(text=text, source_urls=urls)))


class ValidatorTool(BaseTool):
    name: str = "Source Validator"
    description: str = "Validate the credibility of a source URL based on domain, TLD, and content analysis."

    def _run(self, url: str, text_content: str = "") -> str:
        return _fmt(
            validator(
                ValidatorInput(
                    url=url, text_content=text_content or None
                )
            )
        )


class SEOOptimizerTool(BaseTool):
    name: str = "SEO Optimizer"
    description: str = "Analyze text for readability score, grade level, keyword density, and SEO suggestions."

    def _run(self, text: str, target_keywords: str = "") -> str:
        kws = [k.strip() for k in target_keywords.split(",") if k.strip()]
        return _fmt(seo_optimizer(SEOInput(text=text, target_keywords=kws)))


class SummarizerTool(BaseTool):
    name: str = "Summarizer"
    description: str = "Summarize a long text into a concise summary using extractive LSA summarization."

    def _run(self, text: str, max_sentences: int = 5) -> str:
        return _fmt(summarizer(SummarizerInput(text=text, max_sentences=max_sentences)))


TOOL_REGISTRY: dict[str, list[BaseTool]] = {
    "trend_investigator": [
        WebSearchTool(),
        WebScraperTool(),
        SentimentAnalyzerTool(),
    ],
    "risk_analyst": [
        BiasDetectorTool(),
        ValidatorTool(),
    ],
    "copywriter": [
        SEOOptimizerTool(),
        SummarizerTool(),
    ],
}
