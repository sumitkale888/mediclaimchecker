"""RAG orchestration: retrieve evidence, prompt the LLM, return a grounded answer."""

from __future__ import annotations

import logging

from app.models.schemas import LLMVerificationResult, RetrievedFact, VerifyResponse
from app.prompts.verification_prompt import build_verification_prompt
from app.services.llm_service import LLMService
from app.services.retrieval_service import RetrievalService

logger = logging.getLogger(__name__)

DEFAULT_DISCLAIMER = (
    "This is a research/fact-checking tool, not a medical diagnosis or treatment system. "
    "Do not use it as a substitute for professional medical advice."
)


class RAGService:
    """Single RAG pipeline for medical claim verification."""

    def __init__(
        self,
        retrieval_service: RetrievalService | None = None,
        llm_service: LLMService | None = None,
    ) -> None:
        self.retrieval_service = retrieval_service or RetrievalService()
        self.llm_service = llm_service or LLMService()

    def verify_claim(self, claim: str) -> VerifyResponse:
        """Retrieve facts, generate a grounded verdict, and attach retrieved evidence."""
        logger.info("Verifying claim (%s chars)", len(claim))
        retrieved_facts = self.retrieval_service.retrieve(claim)
        prompt = build_verification_prompt(claim, retrieved_facts)
        llm_result = self.llm_service.generate_verification(prompt)
        validated = self._validate_result(llm_result, retrieved_facts)
        return VerifyResponse(
            claim=claim,
            verdict=validated.verdict,
            confidence=validated.confidence,
            explanation=validated.explanation,
            retrieved_facts=retrieved_facts,
            disclaimer=validated.disclaimer or DEFAULT_DISCLAIMER,
        )

    def _validate_result(
        self,
        result: LLMVerificationResult,
        retrieved_facts: list[RetrievedFact],
    ) -> LLMVerificationResult:
        explanation = result.explanation.strip()
        if not explanation:
            explanation = (
                "The model did not provide an explanation. "
                "Treat the verdict as unverified based on the retrieved evidence."
            )
        disclaimer = result.disclaimer.strip() or DEFAULT_DISCLAIMER
        if not retrieved_facts and result.verdict != "Unverified":
            return result.model_copy(
                update={
                    "verdict": "Unverified",
                    "confidence": 0.0,
                    "explanation": explanation,
                    "disclaimer": disclaimer,
                }
            )
        return result.model_copy(update={"explanation": explanation, "disclaimer": disclaimer})
