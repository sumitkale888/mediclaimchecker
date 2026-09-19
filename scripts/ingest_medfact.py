"""Ingest MedFact-Bench JSON records into ChromaDB.

Usage (PowerShell, from the project root):

    python scripts\\ingest_medfact.py
    python scripts\\ingest_medfact.py --rebuild
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import settings  # noqa: E402
from app.services.embedding_service import EmbeddingService  # noqa: E402
from app.services.vector_store import VectorStore  # noqa: E402

logger = logging.getLogger("ingest_medfact")

DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "medfact_bench.json"
PROGRESS_EVERY = 500


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest MedFact-Bench records into ChromaDB.")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Path to medfact_bench.json",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Delete and recreate the medical_facts collection before ingesting.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=settings.ingest_batch_size,
        help="Number of records to embed and upsert per batch.",
    )
    return parser.parse_args()


def load_records(data_path: Path) -> list[dict[str, Any]]:
    if not data_path.exists():
        print(
            "MedFact-Bench dataset not found. "
            "Place it at data/medfact_bench.json and run scripts/ingest_medfact.py."
        )
        sys.exit(1)

    print("Loading data...")
    with data_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if isinstance(payload, dict):
        for key in ("records", "data", "items", "examples"):
            if isinstance(payload.get(key), list):
                payload = payload[key]
                break
        else:
            payload = [payload]

    if not isinstance(payload, list):
        print("Invalid dataset format: expected a JSON list of records.")
        sys.exit(1)

    return payload


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value).strip()


def normalize_record(raw: Any, index: int) -> dict[str, str] | None:
    if not isinstance(raw, dict):
        logger.warning("Skipping non-object record at index %s", index)
        return None

    claim = _as_text(raw.get("claim") or raw.get("text") or raw.get("statement"))
    if not claim:
        logger.warning("Skipping record %s because it has no claim", index)
        return None

    evidence = _as_text(
        raw.get("evidence")
        or raw.get("rationale")
        or raw.get("explanation")
        or raw.get("abstract")
    )
    source = _as_text(raw.get("source"))
    if not evidence and source and len(source) > 80:
        evidence = source

    verdict = _as_text(raw.get("verdict") or raw.get("label"))
    original_label = _as_text(raw.get("original_label") or raw.get("label"))
    dataset = _as_text(raw.get("dataset") or raw.get("source_dataset"))
    if not dataset and source and len(source) <= 80:
        dataset = source

    document = f"Claim: {claim}"
    if evidence:
        document = f"{document}\nEvidence: {evidence}"

    return {
        "id": f"medfact_{index}",
        "document": document,
        "claim": claim,
        "evidence": evidence,
        "verdict": verdict,
        "original_label": original_label,
        "dataset": dataset,
    }


def chroma_metadata(record: dict[str, str]) -> dict[str, str]:
    return {
        "claim": record["claim"],
        "verdict": record["verdict"],
        "original_label": record["original_label"],
        "dataset": record["dataset"],
        "evidence": record["evidence"],
    }


def ingest(records: list[dict[str, Any]], batch_size: int, rebuild: bool) -> None:
    normalized: list[dict[str, str]] = []
    skipped = 0
    for index, raw in enumerate(records):
        item = normalize_record(raw, index)
        if item is None:
            skipped += 1
            continue
        normalized.append(item)

    total = len(normalized)
    print(f"Loaded {len(records)} records")
    if skipped:
        print(f"Skipped {skipped} invalid records")
    print(f"Preparing {total} valid records")
    if total == 0:
        print("No valid records to ingest.")
        print("Completed.")
        return

    vector_store = VectorStore()
    if rebuild:
        print("Rebuilding collection...")
        vector_store.rebuild()
        print("Collection rebuilt.")

    embedder = EmbeddingService()
    print("Embedding...")

    ingested = 0
    last_reported = 0
    for start in range(0, total, batch_size):
        batch = normalized[start : start + batch_size]
        embeddings = embedder.embed_texts([item["document"] for item in batch], batch_size=batch_size)
        vector_store.upsert(
            ids=[item["id"] for item in batch],
            documents=[item["document"] for item in batch],
            embeddings=embeddings,
            metadatas=[chroma_metadata(item) for item in batch],
        )
        ingested += len(batch)
        while ingested >= last_reported + PROGRESS_EVERY:
            last_reported += PROGRESS_EVERY
            print(f"Ingested {last_reported}/{total}")
    if last_reported != ingested:
        print(f"Ingested {ingested}/{total}")

    print("Completed.")
    print(f"Collection '{vector_store.collection_name}' now contains {vector_store.count()} documents.")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    data_path = args.data_path if args.data_path.is_absolute() else PROJECT_ROOT / args.data_path
    records = load_records(data_path)
    ingest(records, batch_size=max(1, args.batch_size), rebuild=args.rebuild)


if __name__ == "__main__":
    main()
