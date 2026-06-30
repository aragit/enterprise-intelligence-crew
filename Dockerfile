FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    --only-binary=:all: \
    || pip install --no-cache-dir \
        crewai langgraph chromadb fastapi uvicorn pydantic pydantic-settings \
        httpx duckduckgo_search beautifulsoup4 trafilatura lxml \
        textblob vaderSentiment textstat loguru \
        prometheus-client prometheus-fastapi-instrumentator \
        python-dotenv numpy

COPY . .

RUN mkdir -p data/chromadb

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
