"""Retrieve top-k medical facts for a user claim."""

from __future__ import annotations

import logging

from app.core.config import settings
from app.core.exceptions import KnowledgeBaseEmptyError, KnowledgeBaseUnavailableError
from app.models.schemas import RetrievedFact
from app.services.embedding_service import Embedder, EmbeddingService
from app.services.vector_store import VectorStore

logger = logging.getLogger(__name__)


class RetrievalService:
    """Query ChromaDB for facts similar to a user claim."""

    def __init__(
        self,
        vector_store: VectorStore | None = None,
        embedder: Embedder | None = None,
        top_k: int | None = None,
    ) -> None:
        self.vector_store = vector_store or VectorStore()
        self.embedder = embedder or EmbeddingService()
        self.top_k = top_k if top_k is not None else settings.top_k

    def retrieve(self, claim: str, top_k: int | None = None) -> list[RetrievedFact]:
        """Return the most relevant stored facts, or raise if the knowledge base is empty."""
        try:
            document_count = self.vector_store.count()
        except KnowledgeBaseUnavailableError:
            raise
        except Exception as exc:
            logger.exception("Vector store count failed")
            raise KnowledgeBaseUnavailableError(
                "Knowledge base is unavailable. Please try again later."
            ) from exc

        if document_count == 0:
            raise KnowledgeBaseEmptyError(
                "Knowledge base is empty. Please ingest medical facts first."
            )

        k = top_k if top_k is not None else self.top_k
        query_embedding = self.embedder.embed_query(claim)

        try:
            raw = self.vector_store.query(query_embedding=query_embedding, top_k=k)
        except Exception as exc:
            logger.exception("Vector store query failed")
            raise KnowledgeBaseUnavailableError(
                "Knowledge base is unavailable. Retrieval failed."
            ) from exc

        return self._parse_results(raw)

    def _parse_results(self, raw: dict) -> list[RetrievedFact]:
        metadatas = (raw.get("metadatas") or [[]])[0]
        distances = (raw.get("distances") or [[]])[0]
        facts: list[RetrievedFact] = []

        for metadata, distance in zip(metadatas, distances):
            metadata = metadata or {}
            distance_value = float(distance) if distance is not None else None
            score = 0.0 if distance_value is None else max(0.0, min(1.0, 1.0 - distance_value))
            facts.append(
                RetrievedFact(
                    claim=str(metadata.get("claim") or ""),
                    evidence=str(metadata.get("evidence") or ""),
                    verdict=str(metadata.get("verdict") or ""),
                    dataset=str(metadata.get("dataset") or ""),
                    score=score,
                    distance=distance_value,
                )
            )
        return facts
