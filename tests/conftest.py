"""Shared pytest fixtures. No real Groq or sentence-transformer calls."""

from __future__ import annotations

import math
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.api import deps
from app.main import app
from app.models.schemas import LLMVerificationResult
from app.services.rag_service import RAGService
from app.services.retrieval_service import RetrievalService
from app.services.vector_store import VectorStore


class FakeEmbedder:
    """Deterministic bag-of-characters embeddings for tests."""

    dim = 32

    def embed_texts(self, texts: list[str], batch_size: int = 64) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        for index, char in enumerate(text.lower()):
            vector[index % self.dim] += (ord(char) % 17) / 17.0
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]


class FakeLLMService:
    def generate_verification(self, user_prompt: str) -> LLMVerificationResult:
        return LLMVerificationResult(
            verdict="False",
            confidence=0.82,
            explanation="Retrieved evidence contradicts the absolute wording of the claim.",
            disclaimer="This is a research tool, not a medical diagnosis system.",
        )


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def vector_store(tmp_path) -> VectorStore:
    return VectorStore(persist_directory=tmp_path / "chroma_db", collection_name="medical_facts")


@pytest.fixture
def seeded_vector_store(vector_store: VectorStore, fake_embedder: FakeEmbedder) -> VectorStore:
    documents = [
        "Claim: Antibiotics can disrupt the gut microbiome.\nEvidence: Multiple studies show antibiotics reduce gut microbial diversity, but do not completely destroy it.",
        "Claim: Drinking water prevents all infections.\nEvidence: Hydration supports health but does not prevent all infections.",
    ]
    metadatas = [
        {
            "claim": "Antibiotics can disrupt the gut microbiome.",
            "evidence": "Multiple studies show antibiotics reduce gut microbial diversity, but do not completely destroy it.",
            "verdict": "True",
            "original_label": "SUPPORT",
            "dataset": "healthver",
        },
        {
            "claim": "Drinking water prevents all infections.",
            "evidence": "Hydration supports health but does not prevent all infections.",
            "verdict": "False",
            "original_label": "CONTRADICT",
            "dataset": "scifact",
        },
    ]
    vector_store.upsert(
        ids=["medfact_0", "medfact_1"],
        documents=documents,
        embeddings=fake_embedder.embed_texts(documents),
        metadatas=metadatas,
    )
    return vector_store


@pytest.fixture
def retrieval_service(seeded_vector_store: VectorStore, fake_embedder: FakeEmbedder) -> RetrievalService:
    return RetrievalService(vector_store=seeded_vector_store, embedder=fake_embedder, top_k=5)


@pytest.fixture
def client(seeded_vector_store: VectorStore, fake_embedder: FakeEmbedder) -> Iterator[TestClient]:
    rag_service = RAGService(
        retrieval_service=RetrievalService(
            vector_store=seeded_vector_store,
            embedder=fake_embedder,
            top_k=5,
        ),
        llm_service=FakeLLMService(),
    )

    app.dependency_overrides[deps.get_vector_store] = lambda: seeded_vector_store
    app.dependency_overrides[deps.get_rag_service] = lambda: rag_service
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.fixture
def empty_kb_client(vector_store: VectorStore, fake_embedder: FakeEmbedder) -> Iterator[TestClient]:
    rag_service = RAGService(
        retrieval_service=RetrievalService(
            vector_store=vector_store,
            embedder=fake_embedder,
            top_k=5,
        ),
        llm_service=FakeLLMService(),
    )
    app.dependency_overrides[deps.get_vector_store] = lambda: vector_store
    app.dependency_overrides[deps.get_rag_service] = lambda: rag_service
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()
