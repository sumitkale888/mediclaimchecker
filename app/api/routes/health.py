"""Health check route."""

from fastapi import APIRouter

from app.models.schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness probe. Does not require the knowledge base or Groq."""
    return HealthResponse(status="ok")
