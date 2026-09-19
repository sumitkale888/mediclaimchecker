"""Persistent ChromaDB vector store for medical facts."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import chromadb
from chromadb.api.models.Collection import Collection

from app.core.config import settings
from app.core.exceptions import KnowledgeBaseUnavailableError

logger = logging.getLogger(__name__)


class VectorStore:
    """Thin wrapper around a persistent Chroma collection."""

    def __init__(
        self,
        persist_directory: Path | str | None = None,
        collection_name: str | None = None,
    ) -> None:
        self.persist_directory = Path(persist_directory or settings.chroma_db_path)
        self.collection_name = collection_name or settings.chroma_collection
        self._client: chromadb.PersistentClient | None = None
        self._collection: Collection | None = None

    def _get_client(self) -> chromadb.PersistentClient:
        if self._client is None:
            try:
                self.persist_directory.mkdir(parents=True, exist_ok=True)
                self._client = chromadb.PersistentClient(path=str(self.persist_directory))
            except Exception as exc:  # pragma: no cover - defensive
                logger.exception("Failed to initialize ChromaDB client")
                raise KnowledgeBaseUnavailableError(
                    "Knowledge base is unavailable. Check CHROMA_DB_PATH and try again."
                ) from exc
        return self._client

    def get_collection(self) -> Collection:
        """Create or load the medical facts collection."""
        if self._collection is None:
            try:
                self._collection = self._get_client().get_or_create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"},
                )
            except Exception as exc:  # pragma: no cover - defensive
                logger.exception("Failed to open ChromaDB collection")
                raise KnowledgeBaseUnavailableError(
                    "Knowledge base is unavailable. The vector collection could not be opened."
                ) from exc
        return self._collection

    def count(self) -> int:
        """Return the number of stored documents."""
        return int(self.get_collection().count())

    def upsert(
        self,
        ids: list[str],
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]],
    ) -> None:
        """Insert or replace documents. Deterministic IDs make ingestion idempotent."""
        if not ids:
            return
        self.get_collection().upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def query(
        self,
        query_embedding: list[float],
        top_k: int,
    ) -> dict[str, Any]:
        """Search the collection using a precomputed query embedding."""
        collection = self.get_collection()
        available = self.count()
        n_results = max(1, min(top_k, available)) if available else 1
        return collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )

    def delete_collection(self) -> None:
        """Drop the current collection so it can be rebuilt."""
        client = self._get_client()
        try:
            client.delete_collection(self.collection_name)
        except Exception:
            logger.info("Collection %s did not exist or could not be deleted", self.collection_name)
        self._collection = None

    def rebuild(self) -> Collection:
        """Delete and recreate an empty collection."""
        self.delete_collection()
        return self.get_collection()
