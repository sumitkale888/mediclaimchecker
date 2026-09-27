"""PubMed RAG service for orchestrating PubMed-based claim verification."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from app.core.config import settings
from app.core.exceptions import LLMServiceError
from app.models.pubmed import PubMedArticle, PubMedEvidence, PubMedVerifyResponse, PubMedVerdict
from app.services.llm_service import LLMService
from app.services.pubmed_service import PubMedService

logger = logging.getLogger(__name__)

PUBMED_SYSTEM_PROMPT = """You are a medical claim verification assistant using PubMed research evidence.

Analyze the user's medical claim using ONLY the provided PubMed research evidence.

Do not invent studies, authors, PMIDs, findings, or citations.

Determine whether the retrieved evidence:
- SUPPORTS the claim
- CONTRADICTS the claim
- PARTIALLY SUPPORTS the claim
- is INSUFFICIENT to determine the claim

Explain the conclusion using the provided research evidence.

If the retrieved papers do not provide sufficient evidence, return "Insufficient Evidence" or "Unverified" rather than guessing.

Mention the relevant research papers used to reach the conclusion."""


class PubMedRAGService:
    """Orchestrates PubMed search, evidence retrieval, and LLM analysis."""

    def __init__(
        self,
        pubmed_service: PubMedService | None = None,
        llm_service: LLMService | None = None,
    ) -> None:
        self.pubmed_service = pubmed_service or PubMedService()
        self.llm_service = llm_service or LLMService()

    def _build_pubmed_query(self, claim: str) -> str:
        """Build a simple PubMed search query from the claim."""
        # Simple deterministic query generation
        # Remove common stop words and keep medical terms
        stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "could",
            "should", "may", "might", "must", "shall", "can", "to", "of", "in",
            "for", "on", "with", "at", "by", "from", "as", "into", "through",
            "during", "before", "after", "above", "below", "between", "under",
            "again", "further", "then", "once", "here", "there", "when", "where",
            "why", "how", "all", "each", "few", "more", "most", "other", "some",
            "such", "no", "nor", "not", "only", "own", "same", "so", "than",
            "too", "very", "just", "and", "but", "if", "or", "because", "until",
            "while", "although", "though", "cures", "cure", "treats", "treat",
        }

        words = claim.lower().split()
        medical_terms = [word for word in words if word not in stop_words and len(word) > 2]

        if not medical_terms:
            return claim

        # Join with AND for Boolean search
        query = " AND ".join(medical_terms)
        return query

    def _build_evidence_context(self, claim: str, articles: list[PubMedArticle]) -> str:
        """Build RAG context from retrieved PubMed articles."""
        if not articles:
            return f"CLAIM:\n{claim}\n\nNo PubMed research evidence was found for this claim."

        context = f"CLAIM:\n{claim}\n\nPUBMED RESEARCH EVIDENCE:\n\n"

        for i, article in enumerate(articles, 1):
            context += f"[PAPER {i}]\n"
            context += f"PMID: {article.pmid}\n"
            context += f"Title: {article.title}\n"
            if article.authors:
                context += f"Authors: {', '.join(article.authors[:3])}"
                if len(article.authors) > 3:
                    context += f" et al. ({len(article.authors)} authors)"
                context += "\n"
            if article.journal:
                context += f"Journal: {article.journal}\n"
            if article.publication_year:
                context += f"Year: {article.publication_year}\n"
            if article.abstract:
                context += f"Abstract:\n{article.abstract}\n"
            else:
                context += "Abstract: Not available\n"
            context += "\n"

        return context

    def _extract_relevant_articles(self, articles: list[dict[str, Any]]) -> list[PubMedArticle]:
        """Extract and filter articles with useful abstracts."""
        relevant_articles = []

        for article_dict in articles:
            # Only include articles with abstracts for meaningful analysis
            if article_dict.get("abstract"):
                article = PubMedArticle(**article_dict)
                relevant_articles.append(article)

        logger.info(f"Extracted {len(relevant_articles)} articles with abstracts from {len(articles)} total articles")
        return relevant_articles

    def verify_claim_with_pubmed(self, claim: str) -> PubMedVerifyResponse:
        """
        Verify a medical claim using PubMed research evidence.

        Args:
            claim: Medical claim to verify

        Returns:
            Structured verification response with PubMed evidence

        Raises:
            LLMServiceError: If PubMed search or LLM analysis fails
        """
        logger.info("Verifying claim with PubMed evidence: %s", claim[:100])

        # Step 1: Build PubMed query
        query = self._build_pubmed_query(claim)
        logger.info("Generated PubMed query: %s", query)

        # Step 2: Search PubMed
        try:
            pmids = self.pubmed_service.search_pubmed(query, max_results=settings.pubmed_max_results)
        except Exception as exc:
            logger.exception("PubMed search failed")
            raise LLMServiceError("Failed to search PubMed. Please try again later.") from exc

        if not pmids:
            logger.warning("No PubMed results found for query: %s", query)
            return PubMedVerifyResponse(
                claim=claim,
                verdict="Insufficient Evidence",
                confidence=0.5,
                explanation="No relevant research papers were found in PubMed to verify this claim.",
                research_papers=[],
                disclaimer="This is a research tool and not medical advice.",
            )

        # Step 3: Fetch articles
        try:
            article_dicts = self.pubmed_service.fetch_pubmed_articles(pmids)
        except Exception as exc:
            logger.exception("PubMed article fetch failed")
            raise LLMServiceError("Failed to fetch PubMed articles. Please try again later.") from exc

        if not article_dicts:
            logger.warning("No articles fetched from PMIDs: %s", pmids)
            return PubMedVerifyResponse(
                claim=claim,
                verdict="Insufficient Evidence",
                confidence=0.5,
                explanation="Research papers were found but could not be retrieved from PubMed.",
                research_papers=[],
                disclaimer="This is a research tool and not medical advice.",
            )

        # Step 4: Extract relevant articles (with abstracts)
        articles = self._extract_relevant_articles(article_dicts)

        if not articles:
            logger.warning("No articles with abstracts found")
            return PubMedVerifyResponse(
                claim=claim,
                verdict="Insufficient Evidence",
                confidence=0.5,
                explanation="Research papers were found but none had available abstracts for analysis.",
                research_papers=[],
                disclaimer="This is a research tool and not medical advice.",
            )

        # Step 5: Build evidence context
        context = self._build_evidence_context(claim, articles)

        # Step 6: Send to LLM with custom prompt for PubMed analysis
        user_prompt = f"""Please analyze this medical claim and provide a structured JSON response:

{context}

Return your analysis in this exact JSON format:
{{
    "verdict": "Supported" or "Contradicted" or "Partially Supported" or "Insufficient Evidence" or "Unverified",
    "confidence": 0.0 to 1.0,
    "explanation": "Your detailed explanation based on the PubMed evidence"
}}"""

        try:
            response_dict = self.llm_service.generate_custom_response(PUBMED_SYSTEM_PROMPT, user_prompt)
        except Exception as exc:
            logger.exception("LLM analysis failed")
            raise LLMServiceError("Failed to analyze PubMed evidence. Please try again later.") from exc

        # Step 7: Build structured response
        verdict_str = response_dict.get("verdict", "Unverified")
        # Map to allowed enum values
        valid_verdicts = ["Supported", "Contradicted", "Partially Supported", "Insufficient Evidence", "Unverified"]
        if verdict_str not in valid_verdicts:
            verdict_str = "Unverified"

        confidence = float(response_dict.get("confidence", 0.5))
        explanation = response_dict.get("explanation", "Analysis failed to generate explanation.")

        # Build research papers with relevance
        research_papers = []
        for article in articles:
            research_papers.append(
                PubMedEvidence(
                    pmid=article.pmid,
                    title=article.title,
                    authors=article.authors,
                    journal=article.journal,
                    publication_year=article.publication_year,
                    relevance="Referenced in analysis",
                    pubmed_url=article.pubmed_url,
                )
            )

        return PubMedVerifyResponse(
            claim=claim,
            verdict=verdict_str,
            confidence=confidence,
            explanation=explanation,
            research_papers=research_papers,
            disclaimer="This is a research tool and not medical advice.",
        )

    def close(self) -> None:
        """Close services."""
        self.pubmed_service.close()
