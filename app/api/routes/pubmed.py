"""PubMed verification API routes."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException
from app.core.exceptions import LLMServiceError
from app.models.pubmed import PubMedVerifyRequest, PubMedVerifyResponse
from app.services.pubmed_rag_service import PubMedRAGService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/pubmed", tags=["PubMed Verification"])

pubmed_rag_service = PubMedRAGService()


@router.post("/verify", response_model=PubMedVerifyResponse)
def verify_claim_with_pubmed(request: PubMedVerifyRequest) -> PubMedVerifyResponse:
    """
    Verify a medical claim using PubMed research evidence.

    This endpoint:
    1. Searches PubMed for relevant research papers
    2. Retrieves article metadata and abstracts
    3. Analyzes the claim using the retrieved evidence
    4. Returns a structured verification with research sources

    Args:
        request: Claim verification request

    Returns:
        Structured verification response with PubMed evidence

    Raises:
        HTTPException: If verification fails
    """
    try:
        result = pubmed_rag_service.verify_claim_with_pubmed(request.claim)
        return result
    except LLMServiceError as exc:
        logger.error("PubMed verification failed: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected error during PubMed verification")
        raise HTTPException(status_code=500, detail="An unexpected error occurred during PubMed verification.") from exc
