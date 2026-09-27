"""PubMed data models."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

PubMedVerdict = Literal["Supported", "Contradicted", "Partially Supported", "Insufficient Evidence", "Unverified"]


class PubMedArticle(BaseModel):
    """Single PubMed article with metadata."""

    pmid: str = Field(..., description="PubMed ID")
    title: str = Field(..., description="Article title")
    abstract: str | None = Field(None, description="Article abstract")
    authors: list[str] = Field(default_factory=list, description="List of authors")
    journal: str | None = Field(None, description="Journal name")
    publication_year: int | None = Field(None, description="Publication year")
    pubmed_url: str = Field(..., description="URL to PubMed entry")


class PubMedEvidence(BaseModel):
    """Research evidence from a single paper."""

    pmid: str
    title: str
    authors: list[str]
    journal: str | None
    publication_year: int | None
    relevance: str = Field(..., description="How this paper relates to the claim")
    pubmed_url: str


class PubMedVerifyRequest(BaseModel):
    """Request to verify a claim using PubMed evidence."""

    claim: str = Field(..., description="Medical claim to verify")


class PubMedVerifyResponse(BaseModel):
    """Response from PubMed-based verification."""

    claim: str
    verdict: PubMedVerdict
    confidence: float = Field(..., ge=0.0, le=1.0, description="System confidence estimate")
    explanation: str
    research_papers: list[PubMedEvidence]
    disclaimer: str
