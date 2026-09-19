"""Claim verification and knowledge-base stats routes."""

from fastapi import APIRouter, Depends

from app.api.deps import get_rag_service, get_vector_store
from app.models.schemas import KnowledgeBaseStatsResponse, VerifyRequest, VerifyResponse
from app.services.rag_service import RAGService
from app.services.vector_store import VectorStore

router = APIRouter(prefix="/api/v1", tags=["verification"])


@router.get("/knowledge-base/stats", response_model=KnowledgeBaseStatsResponse)
def knowledge_base_stats(vector_store: VectorStore = Depends(get_vector_store)) -> KnowledgeBaseStatsResponse:
    """Return the live ChromaDB document count."""
    return KnowledgeBaseStatsResponse(
        collection=vector_store.collection_name,
        document_count=vector_store.count(),
    )


@router.post("/verify", response_model=VerifyResponse)
def verify_claim(
    payload: VerifyRequest,
    rag_service: RAGService = Depends(get_rag_service),
) -> VerifyResponse:
    """Verify a medical claim using retrieval-augmented generation."""
    return rag_service.verify_claim(payload.claim)
