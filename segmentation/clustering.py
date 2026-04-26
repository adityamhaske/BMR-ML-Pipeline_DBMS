"""
Customer Segmentation — Clustering Engine
==========================================
Clusters customer embeddings using HDBSCAN (density-based, handles noise)
and K-Means (fixed-k business segments). Both run on the FAISS index output.

Pipeline:
  1. Load all embeddings from FAISS index
  2. Dimensionality reduction (UMAP 2D/nD for clustering)
  3. HDBSCAN clustering for discovery
  4. K-Means for forced business segment counts
  5. Assign segment labels back to record IDs
  6. Persist segment assignments to Redshift / DuckDB
  7. Compute segment profiles (top keywords, mean rating, size)
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import hdbscan
import numpy as np
import umap
from loguru import logger
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import normalize

from embedding.vector_store import FAISSVectorStoreReader


@dataclass
class SegmentProfile:
    """Summary statistics for a single customer segment."""

    segment_id: int
    size: int
    centroid: np.ndarray
    silhouette_score: float = 0.0
    top_terms: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ClusteringResult:
    """Full output of a clustering run."""

    algorithm: str
    n_segments: int
    record_ids: list[str]
    labels: np.ndarray           # shape (n_records,), -1 = noise (HDBSCAN)
    centroids: np.ndarray        # shape (n_segments, embedding_dim)
    profiles: list[SegmentProfile]
    silhouette: float
    davies_bouldin: float
    noise_fraction: float        # fraction of records labelled as noise
    run_timestamp: str
    elapsed_seconds: float


class SegmentationEngine:
    """
    Clusters customer embeddings into behavioral segments.

    Supports:
    - HDBSCAN: Automatic k discovery, noise-aware, density-based
    - K-Means: Fixed business segments (interpretable for marketing)

    Both use UMAP-reduced embeddings for better cluster structure.
    """

    def __init__(
        self,
        n_clusters: int = 8,
        min_cluster_size: int = 50,
        umap_n_components: int = 50,
        umap_n_neighbors: int = 15,
        random_state: int = 42,
    ) -> None:
        self.n_clusters = n_clusters
        self.min_cluster_size = min_cluster_size
        self.umap_n_components = umap_n_components
        self.umap_n_neighbors = umap_n_neighbors
        self.random_state = random_state

    def _reduce_dimensions(self, embeddings: np.ndarray) -> np.ndarray:
        """UMAP dimensionality reduction for better cluster structure."""
        n_components = min(self.umap_n_components, embeddings.shape[1])
        logger.info(
            f"UMAP reduction | {embeddings.shape[1]}D → {n_components}D | "
            f"n_samples={embeddings.shape[0]:,}"
        )
        reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=self.umap_n_neighbors,
            metric="cosine",
            random_state=self.random_state,
            low_memory=True,
        )
        return reducer.fit_transform(embeddings)

    def _compute_centroids(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
    ) -> np.ndarray:
        """Compute mean centroid for each cluster label."""
        unique_labels = sorted(set(labels) - {-1})  # exclude noise
        centroids = np.array([
            embeddings[labels == lbl].mean(axis=0)
            for lbl in unique_labels
        ])
        return normalize(centroids)

    def _quality_metrics(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
    ) -> tuple[float, float]:
        """Compute silhouette and Davies-Bouldin scores on a sample."""
        mask = labels != -1
        filtered_embeddings = embeddings[mask]
        filtered_labels = labels[mask]

        if len(set(filtered_labels)) < 2:
            return 0.0, 0.0

        # Sample for speed on large datasets
        sample_size = min(10_000, len(filtered_embeddings))
        if len(filtered_embeddings) > sample_size:
            idx = np.random.choice(len(filtered_embeddings), sample_size, replace=False)
            filtered_embeddings = filtered_embeddings[idx]
            filtered_labels = filtered_labels[idx]

        sil = silhouette_score(filtered_embeddings, filtered_labels, metric="cosine")
        db = davies_bouldin_score(filtered_embeddings, filtered_labels)
        return float(sil), float(db)

    def run_hdbscan(
        self,
        embeddings: np.ndarray,
        record_ids: list[str],
    ) -> ClusteringResult:
        """
        Cluster using HDBSCAN — automatic k, noise-aware.

        Best for: Exploratory segmentation, heterogeneous customer bases.
        """
        import datetime

        t0 = time.perf_counter()
        logger.info(f"HDBSCAN clustering | n_records={len(record_ids):,}")

        reduced = self._reduce_dimensions(embeddings)

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=10,
            metric="euclidean",
            cluster_selection_method="eom",
            prediction_data=True,
        )
        labels = clusterer.fit_predict(reduced)

        n_clusters = len(set(labels) - {-1})
        noise_fraction = (labels == -1).mean()
        centroids = self._compute_centroids(reduced, labels)
        silhouette, db = self._quality_metrics(reduced, labels)

        elapsed = time.perf_counter() - t0
        logger.info(
            f"HDBSCAN complete | segments={n_clusters} | "
            f"noise={noise_fraction:.1%} | silhouette={silhouette:.3f} | "
            f"elapsed={elapsed:.1f}s"
        )

        profiles = [
            SegmentProfile(
                segment_id=i,
                size=int((labels == i).sum()),
                centroid=centroids[i] if i < len(centroids) else np.zeros(reduced.shape[1]),
                silhouette_score=silhouette,
            )
            for i in sorted(set(labels) - {-1})
        ]

        return ClusteringResult(
            algorithm="hdbscan",
            n_segments=n_clusters,
            record_ids=record_ids,
            labels=labels,
            centroids=centroids,
            profiles=profiles,
            silhouette=silhouette,
            davies_bouldin=db,
            noise_fraction=noise_fraction,
            run_timestamp=datetime.datetime.utcnow().isoformat(),
            elapsed_seconds=elapsed,
        )

    def run_kmeans(
        self,
        embeddings: np.ndarray,
        record_ids: list[str],
        n_clusters: int | None = None,
    ) -> ClusteringResult:
        """
        Cluster using K-Means — fixed k, interpretable business segments.

        Best for: Marketing campaigns, precision targeting with known segment count.
        """
        import datetime

        k = n_clusters or self.n_clusters
        t0 = time.perf_counter()
        logger.info(f"K-Means clustering | n_records={len(record_ids):,} | k={k}")

        reduced = self._reduce_dimensions(embeddings)

        kmeans = KMeans(
            n_clusters=k,
            random_state=self.random_state,
            n_init=10,
            max_iter=300,
        )
        labels = kmeans.fit_predict(reduced)
        centroids = normalize(kmeans.cluster_centers_)
        silhouette, db = self._quality_metrics(reduced, labels)

        elapsed = time.perf_counter() - t0
        logger.info(
            f"K-Means complete | k={k} | "
            f"silhouette={silhouette:.3f} | db={db:.3f} | "
            f"elapsed={elapsed:.1f}s"
        )

        profiles = [
            SegmentProfile(
                segment_id=i,
                size=int((labels == i).sum()),
                centroid=centroids[i],
                silhouette_score=silhouette,
            )
            for i in range(k)
        ]

        return ClusteringResult(
            algorithm="kmeans",
            n_segments=k,
            record_ids=record_ids,
            labels=labels,
            centroids=centroids,
            profiles=profiles,
            silhouette=silhouette,
            davies_bouldin=db,
            noise_fraction=0.0,
            run_timestamp=datetime.datetime.utcnow().isoformat(),
            elapsed_seconds=elapsed,
        )

    def save(self, result: ClusteringResult, output_dir: str) -> None:
        """Persist clustering result for serving layer."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # ID → label mapping
        id_to_label = dict(zip(result.record_ids, result.labels.tolist()))
        (out / "segment_labels.json").write_text(json.dumps(id_to_label))

        # Centroids
        np.save(str(out / "centroids.npy"), result.centroids)

        # Summary
        summary = {
            "algorithm": result.algorithm,
            "n_segments": result.n_segments,
            "silhouette": result.silhouette,
            "davies_bouldin": result.davies_bouldin,
            "noise_fraction": result.noise_fraction,
            "run_timestamp": result.run_timestamp,
            "elapsed_seconds": result.elapsed_seconds,
            "profiles": [
                {
                    "segment_id": p.segment_id,
                    "size": p.size,
                    "silhouette_score": p.silhouette_score,
                }
                for p in result.profiles
            ],
        }
        (out / "clustering_summary.json").write_text(json.dumps(summary, indent=2))
        logger.info(f"Clustering result saved to {out}")
