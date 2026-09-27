"""API and RAG data contracts."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

Verdict = Literal["True", "False", "Misleading", "Unverified"]


class VerifyRequest(BaseModel):
    """Incoming medical claim to verify."""

    claim: str = Field(..., description="Medical claim submitted by the user.")

    @field_validator("claim")
    @classmethod
    def normalize_claim(cls, value: str) -> str:
        if value is None:
            raise ValueError("claim is required")
        claim = value.strip()
        if not claim:
            raise ValueError("claim must not be empty")
        from app.core.config import settings

        if len(claim) > settings.max_claim_length:
            raise ValueError(f"claim must be at most {settings.max_claim_length} characters")
        return claim


class RetrievedFact(BaseModel):
    """One fact retrieved from the vector store."""

    claim: str
    evidence: str
    verdict: str = ""
    dataset: str = ""
    score: float = Field(..., description="Relevance score derived from vector distance.")
    distance: float | None = None


class VerifyResponse(BaseModel):
    """Grounded verification result returned to the client."""

    claim: str
    verdict: Verdict
    confidence: float = Field(..., ge=0.0, le=1.0)
    explanation: str
    retrieved_facts: list[RetrievedFact]
    disclaimer: str
    cluster_id: int | None = Field(None, description="K-Means cluster ID for semantic analysis")


class HealthResponse(BaseModel):
    status: str


class KnowledgeBaseStatsResponse(BaseModel):
    collection: str
    document_count: int


class ErrorResponse(BaseModel):
    detail: str


class LLMVerificationResult(BaseModel):
    """Structured payload expected from the LLM."""

    verdict: Verdict
    confidence: float = Field(..., ge=0.0, le=1.0)
    explanation: str
    disclaimer: str = ""


class ClusterAnalysisRequest(BaseModel):
    """Request for cluster analysis."""

    claim: str = Field(..., description="Medical claim to analyze")


class ClusterAnalysisResponse(BaseModel):
    """Response from cluster analysis."""

    claim: str
    predicted_cluster_id: int
    example_claims: list[str] = Field(default_factory=list)
    note: str | None = None


class ClusterStatsResponse(BaseModel):
    """Response with cluster statistics."""

    total_clusters: int
    model_type: str
    cluster_distribution: dict[int, str]
    model_path: str


class ClusterExamplesResponse(BaseModel):
    """Response with cluster examples."""

    cluster_id: int
    example_claims: list[str]
    note: str | None = None
