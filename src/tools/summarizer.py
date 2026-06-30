from __future__ import annotations

from pydantic import BaseModel, Field
from src.tools.base import ToolResult, retry


class SummarizerInput(BaseModel):
    text: str = Field(..., min_length=20, description="Text to summarize")
    max_sentences: int = Field(default=5, ge=1, le=20)


class SummarizerOutput(BaseModel):
    summary: str = ""
    original_length: int = 0
    summary_length: int = 0


@retry(max_attempts=3, delay=1.0)
def summarizer(input_data: SummarizerInput) -> ToolResult:
    from sumy.parsers.plaintext import PlaintextParser
    from sumy.nlp.tokenizers import Tokenizer
    from sumy.summarizers.lsa import LsaSummarizer

    parser = PlaintextParser.from_string(input_data.text, Tokenizer("english"))
    summarizer_engine = LsaSummarizer()
    sentences = summarizer_engine(parser.document, input_data.max_sentences)
    summary = " ".join(str(s) for s in sentences)

    output = SummarizerOutput(
        summary=summary,
        original_length=len(input_data.text.split()),
        summary_length=len(summary.split()),
    )
    return ToolResult(success=True, data=output.model_dump())
