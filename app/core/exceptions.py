"""Domain exceptions mapped to HTTP responses by the API layer."""


class AppError(Exception):
    """Base application error."""


class KnowledgeBaseEmptyError(AppError):
    """Raised when ChromaDB has no ingested medical facts."""


class KnowledgeBaseUnavailableError(AppError):
    """Raised when the vector store cannot be reached."""


class LLMServiceError(AppError):
    """Raised when the Groq LLM call fails or returns unusable output."""


class PubMedServiceError(AppError):
    """Raised when PubMed API calls fail or return unusable data."""
