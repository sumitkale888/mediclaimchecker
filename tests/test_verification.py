"""Verification API, retrieval, and vector-store tests."""

from __future__ import annotations

import pytest

from app.core.exceptions import KnowledgeBaseEmptyError
from app.services.retrieval_service import RetrievalService
from app.services.vector_store import VectorStore
from tests.conftest import FakeEmbedder


def test_empty_claim_rejected(client) -> None:
    response = client.post("/api/v1/verify", json={"claim": "   "})
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_missing_claim_rejected(client) -> None:
    response = client.post("/api/v1/verify", json={})
    assert response.status_code == 400


def test_vector_store_initialization(tmp_path) -> None:
    store = VectorStore(persist_directory=tmp_path / "chroma_db", collection_name="medical_facts")
    collection = store.get_collection()
    assert collection.name == "medical_facts"
    assert store.count() == 0


def test_knowledge_base_stats_empty(empty_kb_client) -> None:
    response = empty_kb_client.get("/api/v1/knowledge-base/stats")
    assert response.status_code == 200
    body = response.json()
    assert body["collection"] == "medical_facts"
    assert body["document_count"] == 0


def test_knowledge_base_stats_seeded(client) -> None:
    response = client.get("/api/v1/knowledge-base/stats")
    assert response.status_code == 200
    body = response.json()
    assert body["collection"] == "medical_facts"
    assert body["document_count"] == 2


def test_retrieval_returns_facts(retrieval_service: RetrievalService) -> None:
    facts = retrieval_service.retrieve("Antibiotics completely destroy the gut microbiome.")
    assert len(facts) >= 1
    assert facts[0].claim
    assert facts[0].evidence
    assert facts[0].dataset
    assert 0.0 <= facts[0].score <= 1.0


def test_retrieval_empty_knowledge_base(vector_store: VectorStore) -> None:
    service = RetrievalService(vector_store=vector_store, embedder=FakeEmbedder(), top_k=5)
    with pytest.raises(KnowledgeBaseEmptyError, match="Knowledge base is empty"):
        service.retrieve("Any medical claim")


def test_verify_with_mocked_llm(client) -> None:
    response = client.post(
        "/api/v1/verify",
        json={"claim": "Antibiotics completely destroy the gut microbiome."},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["claim"] == "Antibiotics completely destroy the gut microbiome."
    assert body["verdict"] == "False"
    assert body["confidence"] == 0.82
    assert body["explanation"]
    assert body["retrieved_facts"]
    assert "claim" in body["retrieved_facts"][0]
    assert "evidence" in body["retrieved_facts"][0]
    assert "score" in body["retrieved_facts"][0]


def test_verify_empty_knowledge_base(empty_kb_client) -> None:
    response = empty_kb_client.post(
        "/api/v1/verify",
        json={"claim": "Antibiotics completely destroy the gut microbiome."},
    )
    assert response.status_code == 503
    assert "empty" in response.json()["detail"].lower()


def test_upsert_is_idempotent(vector_store: VectorStore, fake_embedder: FakeEmbedder) -> None:
    documents = ["Claim: Example claim\nEvidence: Example evidence"]
    metadata = [
        {
            "claim": "Example claim",
            "evidence": "Example evidence",
            "verdict": "True",
            "original_label": "SUPPORT",
            "dataset": "scifact",
        }
    ]
    embeddings = fake_embedder.embed_texts(documents)
    vector_store.upsert(["medfact_0"], documents, embeddings, metadata)
    vector_store.upsert(["medfact_0"], documents, embeddings, metadata)
    assert vector_store.count() == 1
