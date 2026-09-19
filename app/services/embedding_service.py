"""Sentence-transformer embedding service."""

from __future__ import annotations

import logging
from typing import Protocol

from app.core.config import settings

logger = logging.getLogger(__name__)


class Embedder(Protocol):
    """Minimal embedding interface used by retrieval and ingestion."""

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        ...

    def embed_query(self, text: str) -> list[float]:
        ...


class EmbeddingService:
    """Lazy-loaded wrapper around sentence-transformers."""

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or settings.embedding_model
        self._model = None

    def _load_model(self):
        if self._model is None:
            logger.info("Loading embedding model: %s", self.model_name)
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed_texts(self, texts: list[str], batch_size: int = 64) -> list[list[float]]:
        """Encode a list of documents into embeddings."""
        if not texts:
            return []
        model = self._load_model()
        vectors = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        """Encode a single query string."""
        embeddings = self.embed_texts([text], batch_size=1)
        return embeddings[0]
