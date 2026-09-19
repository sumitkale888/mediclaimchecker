"""FastAPI application entrypoint."""

from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api.routes.health import router as health_router
from app.api.routes.verification import router as verification_router
from app.core.config import settings
from app.core.exceptions import KnowledgeBaseEmptyError, KnowledgeBaseUnavailableError, LLMServiceError

logging.basicConfig(
    level=getattr(logging, settings.log_level, logging.INFO),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Medical Claim Checker",
    description="Single RAG pipeline for verifying medical claims against a MedFact-Bench knowledge base.",
    version="0.1.0",
)

app.include_router(health_router)
app.include_router(verification_router)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_request: Request, exc: RequestValidationError) -> JSONResponse:
    messages: list[str] = []
    for error in exc.errors():
        loc = ".".join(str(part) for part in error.get("loc", []) if part != "body")
        msg = error.get("msg", "Invalid request")
        messages.append(f"{loc}: {msg}" if loc else msg)
    detail = messages[0] if len(messages) == 1 else "; ".join(messages)
    return JSONResponse(status_code=400, content={"detail": detail})


@app.exception_handler(KnowledgeBaseEmptyError)
async def empty_kb_handler(_request: Request, exc: KnowledgeBaseEmptyError) -> JSONResponse:
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.exception_handler(KnowledgeBaseUnavailableError)
async def unavailable_kb_handler(_request: Request, exc: KnowledgeBaseUnavailableError) -> JSONResponse:
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.exception_handler(LLMServiceError)
async def llm_error_handler(_request: Request, exc: LLMServiceError) -> JSONResponse:
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.exception_handler(Exception)
async def unhandled_error_handler(_request: Request, exc: Exception) -> JSONResponse:
    if isinstance(exc, (HTTPException, StarletteHTTPException, RequestValidationError)):
        raise exc
    logger.exception("Unhandled server error: %s", type(exc).__name__)
    return JSONResponse(status_code=500, content={"detail": "An unexpected internal error occurred."})
