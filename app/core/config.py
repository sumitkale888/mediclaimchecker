"""Application configuration loaded from environment variables."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env", override=True)


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, str(default)).strip()
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be an integer, got {raw!r}.") from exc


def _env_path(name: str, default: str) -> Path:
    raw = os.getenv(name, default).strip() or default
    path = Path(raw)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


class Settings:
    """Runtime settings for the RAG backend."""

    groq_api_key: str
    groq_model: str
    chroma_db_path: Path
    chroma_collection: str
    embedding_model: str
    top_k: int
    max_claim_length: int
    ingest_batch_size: int
    app_host: str
    app_port: int
    log_level: str
    database_url: str | None

    def __init__(self) -> None:
        self.groq_api_key = os.getenv("GROQ_API_KEY", "").strip()
        self.groq_model = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b").strip()
        self.chroma_db_path = _env_path("CHROMA_DB_PATH", "./chroma_db")
        self.chroma_collection = os.getenv("CHROMA_COLLECTION", "medical_facts").strip()
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2").strip()
        self.top_k = _env_int("TOP_K", 5)
        self.max_claim_length = _env_int("MAX_CLAIM_LENGTH", 2000)
        self.ingest_batch_size = _env_int("INGEST_BATCH_SIZE", 128)
        self.app_host = os.getenv("APP_HOST", "0.0.0.0").strip()
        self.app_port = _env_int("APP_PORT", 8000)
        self.log_level = os.getenv("LOG_LEVEL", "INFO").strip().upper()
        database_url = os.getenv("DATABASE_URL", "").strip()
        self.database_url = database_url or None


settings = Settings()
