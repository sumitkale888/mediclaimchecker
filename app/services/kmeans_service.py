"""K-Means clustering service for semantic analysis of medical claims."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from app.core.config import settings
from app.core.exceptions import KMeansServiceError
from app.services.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)





class KMeansService:
    """Service for K-Means clustering analysis of medical claims."""

    def __init__(
        self,
        model_path: Path | str | None = None,
        embedding_service: EmbeddingService | None = None,
    ) -> None:
        self.model_path = Path(model_path or "app/models/kmeans_model.pkl")
        if not self.model_path.is_absolute():
            self.model_path = Path(__file__).resolve().parents[2] / self.model_path

        self.embedding_service = embedding_service or EmbeddingService()
        self._model = None
        self._cluster_examples: dict[int, list[str]] | None = None

    def _load_model(self):
        """Load the K-Means model from disk."""
        if self._model is None:
            if not self.model_path.exists():
                raise KMeansServiceError(
                    f"K-Means model not found at {self.model_path}. "
                    "Please ensure the model file exists."
                )

            try:
                logger.info("Loading K-Means model from %s", self.model_path)
                self._model = joblib.load(self.model_path)
                logger.info("K-Means model loaded successfully with %d clusters", self._model.n_clusters)
            except Exception as exc:
                logger.exception("Failed to load K-Means model")
                raise KMeansServiceError(f"Failed to load K-Means model: {exc}") from exc

        return self._model

    def predict_cluster(self, claim: str) -> int:
        """
        Predict the cluster for a given claim.

        Args:
            claim: Medical claim text

        Returns:
            Cluster ID (integer)

        Raises:
            KMeansServiceError: If prediction fails
        """
        try:
            # Generate embedding using existing service
            embedding = self.embedding_service.embed_query(claim)

            # Try different dtype approaches for compatibility
            model = self._load_model()

            # Try multiple approaches for dtype compatibility
            approaches = [
                lambda emb: np.array(emb, dtype=np.float64).reshape(1, -1),
                lambda emb: np.array(emb, dtype='float64').reshape(1, -1),
                lambda emb: np.array(emb, dtype=np.double).reshape(1, -1),
                lambda emb: np.array(emb).reshape(1, -1),
                lambda emb: np.array(emb, dtype='float32').reshape(1, -1),
            ]

            last_error = None
            for i, approach in enumerate(approaches):
                try:
                    embedding_array = approach(embedding)
                    cluster_id = int(model.predict(embedding_array)[0])
                    logger.info("Claim assigned to cluster %d using approach %d", cluster_id, i)
                    return cluster_id
                except Exception as dtype_error:
                    last_error = dtype_error
                    if i < len(approaches) - 1:
                        logger.debug("Approach %d failed, trying next", i)
                        continue

            # If all approaches failed, raise the last error
            raise last_error if last_error else Exception("All dtype approaches failed")

        except KMeansServiceError:
            raise
        except Exception as exc:
            logger.exception("Failed to predict cluster for claim")
            raise KMeansServiceError(f"Failed to predict cluster: {exc}") from exc

    def get_cluster_stats(self) -> dict[str, Any]:
        """
        Get statistics about the K-Means clusters.

        Returns:
            Dictionary with cluster statistics
        """
        try:
            model = self._load_model()

            stats = {
                "total_clusters": model.n_clusters,
                "model_type": type(model).__name__,
                "cluster_distribution": {},  # Would need training data for this
                "model_path": str(self.model_path),
            }

            # Note: Actual cluster distribution would require training data
            # This is a placeholder for the structure
            for i in range(model.n_clusters):
                stats["cluster_distribution"][i] = "unknown"

            return stats

        except KMeansServiceError:
            raise
        except Exception as exc:
            logger.exception("Failed to get cluster stats")
            raise KMeansServiceError(f"Failed to get cluster statistics: {exc}") from exc

    def get_cluster_examples(self, cluster_id: int, limit: int = 5) -> list[str]:
        """
        Get example claims for a specific cluster.

        Note: This requires access to the original training data or
        cluster metadata stored in ChromaDB. For now, returns empty list
        with explanation.

        Args:
            cluster_id: Cluster ID to get examples for
            limit: Maximum number of examples to return

        Returns:
            List of example claims (empty if no cluster metadata available)
        """
        # This would require cluster metadata to be stored with documents
        # For now, return empty list with logging
        logger.warning(
            "Cluster examples requested but cluster metadata not available. "
            "To enable this feature, cluster IDs must be stored in ChromaDB metadata."
        )
        return []

    def analyze_claim_cluster(self, claim: str) -> dict[str, Any]:
        """
        Analyze a claim and return cluster information.

        Args:
            claim: Medical claim text

        Returns:
            Dictionary with claim and cluster information
        """
        try:
            cluster_id = self.predict_cluster(claim)
            examples = self.get_cluster_examples(cluster_id, limit=3)

            return {
                "claim": claim,
                "predicted_cluster_id": cluster_id,
                "example_claims": examples,
                "note": "Cluster examples require cluster metadata in ChromaDB" if not examples else None,
            }

        except KMeansServiceError:
            raise
        except Exception as exc:
            logger.exception("Failed to analyze claim cluster")
            raise KMeansServiceError(f"Failed to analyze claim cluster: {exc}") from exc
