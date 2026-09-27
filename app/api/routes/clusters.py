"""Cluster analysis API routes."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from app.services.kmeans_service import KMeansServiceError
from app.models.schemas import (
    ClusterAnalysisRequest,
    ClusterAnalysisResponse,
    ClusterExamplesResponse,
    ClusterStatsResponse,
)
from app.services.kmeans_service import KMeansService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/clusters", tags=["Cluster Analysis"])

kmeans_service = KMeansService()


# @router.get("/stats", response_model=ClusterStatsResponse)
# def get_cluster_stats() -> ClusterStatsResponse:
#     """
#     Get statistics about the K-Means clusters.

#     Returns:
#         Cluster statistics including total clusters and distribution
#     """
#     try:
#         stats = kmeans_service.get_cluster_stats()
#         return ClusterStatsResponse(**stats)
#     except KMeansServiceError as exc:
#         logger.error("Failed to get cluster stats: %s", exc)
#         raise HTTPException(status_code=503, detail=str(exc)) from exc
#     except Exception as exc:
#         logger.exception("Unexpected error getting cluster stats")
#         raise HTTPException(status_code=500, detail="Failed to get cluster statistics.") from exc


# @router.get("/{cluster_id}", response_model=ClusterExamplesResponse)
# def get_cluster_examples(cluster_id: int) -> ClusterExamplesResponse:
#     """
#     Get example claims for a specific cluster.

#     Args:
#         cluster_id: Cluster ID to get examples for

#     Returns:
#         Example claims from the cluster
#     """
#     try:
#         examples = kmeans_service.get_cluster_examples(cluster_id, limit=5)
#         note = "Cluster examples require cluster metadata in ChromaDB" if not examples else None
#         return ClusterExamplesResponse(
#             cluster_id=cluster_id,
#             example_claims=examples,
#             note=note,
#         )
#     except KMeansServiceError as exc:
#         logger.error("Failed to get cluster examples: %s", exc)
#         raise HTTPException(status_code=503, detail=str(exc)) from exc
#     except Exception as exc:
#         logger.exception("Unexpected error getting cluster examples")
#         raise HTTPException(status_code=500, detail="Failed to get cluster examples.") from exc


@router.post("/analyze", response_model=ClusterAnalysisResponse)
def analyze_cluster(request: ClusterAnalysisRequest) -> ClusterAnalysisResponse:
    """
    Analyze a claim and return cluster information.

    Args:
        request: Cluster analysis request with claim

    Returns:
        Cluster analysis with predicted cluster and examples
    """
    try:
        analysis = kmeans_service.analyze_claim_cluster(request.claim)
        return ClusterAnalysisResponse(**analysis)
    except KMeansServiceError as exc:
        logger.error("Failed to analyze claim cluster: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected error during cluster analysis")
        raise HTTPException(status_code=500, detail="Failed to analyze claim cluster.") from exc
