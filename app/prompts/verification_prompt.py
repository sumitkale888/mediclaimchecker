"""Prompt builder for medical claim verification."""

from __future__ import annotations

from app.models.schemas import RetrievedFact

SYSTEM_PROMPT = """You are a careful medical claim fact-checking assistant for a research tool.

Rules:
- Base your answer primarily on the retrieved evidence provided below.
- Do not invent evidence, studies, statistics, citations, or sources.
- Do not treat semantic similarity as proof. Similar wording is not the same as verification.
- Distinguish between SUPPORT, CONTRADICT, and NEI/insufficient evidence.
- If the retrieved evidence is missing, weak, off-topic, or conflicting without resolution, choose Unverified.
- Verdict must be exactly one of: True, False, Misleading, Unverified.
- True: retrieved evidence clearly supports the claim.
- False: retrieved evidence clearly contradicts the claim.
- Misleading: the claim is only partly accurate, overstated, or missing important caveats found in the evidence.
- Unverified: the knowledge base is insufficient to verify the claim reliably.
- Explain why the evidence supports or contradicts the claim.
- Mention the source dataset names when they are provided.
- Do not give dangerous medical instructions, dosages, or treatment advice.
- This is a fact-checking/research tool, not a medical diagnosis system.
- Return valid JSON only, with this schema:
{
  "verdict": "True | False | Misleading | Unverified",
  "confidence": 0.0,
  "explanation": "...",
  "disclaimer": "..."
}
confidence must be a number between 0 and 1 reflecting how well the retrieved evidence covers the claim.
"""


def build_verification_prompt(claim: str, retrieved_facts: list[RetrievedFact]) -> str:
    """Build the user prompt containing the claim and retrieved evidence only."""
    if not retrieved_facts:
        evidence_block = "No retrieved evidence was provided."
    else:
        parts: list[str] = []
        for index, fact in enumerate(retrieved_facts, start=1):
            parts.append(
                "\n".join(
                    [
                        f"[Evidence {index}]",
                        f"Dataset: {fact.dataset or 'unknown'}",
                        f"KB claim: {fact.claim}",
                        f"KB verdict: {fact.verdict or 'unknown'}",
                        f"Evidence: {fact.evidence or 'Not provided in knowledge base record.'}",
                        f"Relevance score: {fact.score:.4f}",
                    ]
                )
            )
        evidence_block = "\n\n".join(parts)

    return (
        f"USER CLAIM:\n{claim}\n\n"
        f"RETRIEVED EVIDENCE:\n{evidence_block}\n\n"
        "Use only the retrieved evidence above. "
        "If it is insufficient, say the claim cannot be reliably verified from the available knowledge base."
    )
