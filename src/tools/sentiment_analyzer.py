from __future__ import annotations

from pydantic import BaseModel, Field
from src.tools.base import ToolResult, retry


class SentimentInput(BaseModel):
    text: str = Field(..., min_length=1, description="Text to analyze")


class SentimentOutput(BaseModel):
    polarity: float = 0.0
    subjectivity: float = 0.0
    label: str = "neutral"


@retry(max_attempts=3, delay=1.0)
def sentiment_analyzer(input_data: SentimentInput) -> ToolResult:
    from textblob import TextBlob

    blob = TextBlob(input_data.text)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity

    if polarity > 0.1:
        label = "positive"
    elif polarity < -0.1:
        label = "negative"
    else:
        label = "neutral"

    output = SentimentOutput(polarity=polarity, subjectivity=subjectivity, label=label)
    return ToolResult(success=True, data=output.model_dump())
