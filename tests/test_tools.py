from __future__ import annotations

import pytest

from src.tools.base import ToolResult, retry
from src.tools.web_search import web_search, WebSearchInput
from src.tools.web_scraper import web_scraper, WebScraperInput
from src.tools.summarizer import summarizer, SummarizerInput
from src.tools.sentiment_analyzer import sentiment_analyzer, SentimentInput
from src.tools.bias_detector import bias_detector, BiasInput
from src.tools.validator import validator, ValidatorInput
from src.tools.seo_optimizer import seo_optimizer, SEOInput


class TestRetryDecorator:
    def test_retry_on_failure(self):
        call_count = 0

        @retry(max_attempts=3, delay=0.01)
        def fail_twice():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError(f"Attempt {call_count} failed")
            return ToolResult(success=True, data="ok")

        result = fail_twice()
        assert result.success is True
        assert result.data == "ok"

    def test_retry_exhausted(self):
        @retry(max_attempts=2, delay=0.01)
        def always_fail():
            raise ValueError("always fails")

        result = always_fail()
        assert result.success is False
        assert "always fails" in result.error


class TestWebSearch:
    def test_valid_input(self):
        inp = WebSearchInput(query="test", max_results=5)
        assert inp.query == "test"

    def test_max_results_clamped_by_pydantic(self):
        with pytest.raises(Exception):
            WebSearchInput(query="test", max_results=25)


class TestWebScraper:
    def test_invalid_url(self):
        with pytest.raises(Exception):
            WebScraperInput(url="not-a-url")


class TestSummarizer:
    def test_short_text_raises(self):
        with pytest.raises(Exception):
            SummarizerInput(text="short")

    def test_max_sentences_clamped_by_pydantic(self):
        with pytest.raises(Exception):
            SummarizerInput(text="A" * 50, max_sentences=25)


class TestSentimentAnalyzer:
    def test_positive_sentiment(self):
        result = sentiment_analyzer(SentimentInput(text="This is amazing and wonderful!"))
        assert result.success is True
        assert result.data["label"] == "positive"

    def test_negative_sentiment(self):
        result = sentiment_analyzer(SentimentInput(text="This is terrible and awful."))
        assert result.success is True
        assert result.data["label"] == "negative"

    def test_neutral_sentiment(self):
        result = sentiment_analyzer(SentimentInput(text="The meeting is at 3pm."))
        assert result.success is True
        assert result.data["label"] == "neutral"

    def test_empty_text_raises(self):
        with pytest.raises(Exception):
            SentimentInput(text="")


class TestBiasDetector:
    def test_high_bias_detection(self):
        result = bias_detector(BiasInput(text="This revolutionary groundbreaking never-before-seen product is guaranteed to change everything!"))
        assert result.success is True
        assert len(result.data["flagged_phrases"]) >= 3
        assert result.data["verdict"] == "high bias"

    def test_low_bias(self):
        result = bias_detector(BiasInput(text="The study found a 12% improvement in response time."))
        assert result.success is True
        assert result.data["verdict"] == "low bias"

    def test_source_diversity(self):
        result = bias_detector(BiasInput(
            text="Some claim this is true.",
            source_urls=["https://site1.com/a", "https://site2.com/b", "https://site3.com/c"],
        ))
        assert result.success is True
        assert result.data["source_diversity_score"] >= 0.5


class TestValidator:
    def test_trusted_domain(self):
        result = validator(ValidatorInput(
            url="https://reuters.com/article",
            text_content="A " * 100,
        ))
        assert result.success is True
        assert result.data["domain_credible"] is True

    def test_untrusted_domain(self):
        result = validator(ValidatorInput(
            url="https://random-blog-123.org/page",
            text_content="A " * 100,
        ))
        assert result.success is True
        assert result.data["is_valid"] is True


class TestSEOOptimizer:
    def test_readability_scores(self):
        result = seo_optimizer(SEOInput(
            text="Simple and clear text that anyone can read easily without any problems.",
            target_keywords=["simple", "read"],
        ))
        assert result.success is True
        assert result.data["readability_score"] > 0

    def test_keyword_density(self):
        text = "AI is transforming manufacturing. AI improves efficiency. AI reduces costs."
        result = seo_optimizer(SEOInput(text=text, target_keywords=["AI"]))
        assert result.success is True
        assert result.data["keyword_density"]["AI"] > 0

    def test_short_content_suggestion(self):
        result = seo_optimizer(SEOInput(text="Short content."))
        assert result.success is True
        assert any("short" in s.lower() for s in result.data["suggestions"])
