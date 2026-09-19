"""FastAPI dependency providers."""

from __future__ import annotations

from functools import lru_cache

from app.services.embedding_service import EmbeddingService
from app.services.llm_service import LLMService
from app.services.rag_service import RAGService
from app.services.retrieval_service import RetrievalService
from app.services.vector_store import VectorStore


@lru_cache
def get_vector_store() -> VectorStore:
    return VectorStore()


@lru_cache
def get_embedding_service() -> EmbeddingService:
    return EmbeddingService()


@lru_cache
def get_retrieval_service() -> RetrievalService:
    return RetrievalService(
        vector_store=get_vector_store(),
        embedder=get_embedding_service(),
    )


@lru_cache
def get_llm_service() -> LLMService:
    return LLMService()


@lru_cache
def get_rag_service() -> RAGService:
    return RAGService(
        retrieval_service=get_retrieval_service(),
        llm_service=get_llm_service(),
    )
