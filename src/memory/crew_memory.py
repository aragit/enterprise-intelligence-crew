from __future__ import annotations

import logging
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from src.config import settings

logger = logging.getLogger(__name__)

COLLECTION_NAME = "enterprise_research"


class CrewMemory:
    def __init__(self) -> None:
        self._client = chromadb.PersistentClient(
            path=settings.chromadb_path,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        self._embedder = None

    def _get_embedder(self):
        if self._embedder is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
            except ImportError:
                import hashlib
                import numpy as np

                class _FallbackEmbedder:
                    @staticmethod
                    def encode(texts: str | list[str]) -> np.ndarray:
                        if isinstance(texts, str):
                            texts = [texts]
                        dim = 384
                        out = np.zeros((len(texts), dim), dtype=np.float32)
                        for i, t in enumerate(texts):
                            h = hashlib.sha256(t.encode()).digest()
                            for j in range(dim):
                                out[i, j] = (h[j % 32] / 255.0) - 0.5
                        return out.squeeze(0) if len(texts) == 1 else out

                self._embedder = _FallbackEmbedder()
        return self._embedder

    def add_research(
        self,
        topic: str,
        trend_payload: dict[str, Any],
        risk_payload: dict[str, Any],
        content_payload: dict[str, Any],
    ) -> str:
        embedder = self._get_embedder()
        text = f"{topic}: {trend_payload.get('trend_name', '')}"
        embedding = embedder.encode(text).tolist()

        doc_id = f"research_{topic.lower().replace(' ', '_')[:50]}"
        metadata = {
            "topic": topic,
            "trend_name": trend_payload.get("trend_name", ""),
            "risk_score": str(risk_payload.get("risk_score", 0)),
        }

        self._collection.add(
            ids=[doc_id],
            embeddings=[embedding],
            metadatas=[metadata],
            documents=[
                f"Topic: {topic}\nTrend: {trend_payload}\nRisk: {risk_payload}\nContent: {content_payload}"
            ],
        )
        logger.info("CrewMemory: stored research '%s' as %s", topic, doc_id)
        return doc_id

    def find_similar(self, topic: str, k: int = 3) -> list[dict[str, Any]]:
        embedder = self._get_embedder()
        embedding = embedder.encode(topic).tolist()

        results = self._collection.query(
            query_embeddings=[embedding],
            n_results=min(k, self._collection.count()),
        )

        output: list[dict[str, Any]] = []
        if results["ids"]:
            for i in range(len(results["ids"][0])):
                output.append(
                    {
                        "id": results["ids"][0][i],
                        "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                        "distance": results["distances"][0][i] if results["distances"] else 0.0,
                    }
                )
        return output

    def count(self) -> int:
        return self._collection.count()
