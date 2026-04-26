"""
Embedding Drift Detector
=========================
Compares cluster centroids between consecutive clustering runs to detect
behavioral drift in customer segments.

Alert triggered when:
  - Centroid cosine distance between runs exceeds `threshold` (default 0.15)
  - Segment distribution shift (JS divergence) exceeds `dist_threshold`
  - A segment disappears or a new one forms

Usage:
    detector = DriftDetector(threshold=0.15)
    report = detector.compare(current_result, previous_result)
    if report.has_drift:
        notify_slack(report)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from loguru import logger
from sklearn.preprocessing import normalize


@dataclass
class DriftReport:
    """Summary of drift analysis between two clustering runs."""

    has_drift: bool
    max_centroid_drift: float
    mean_centroid_drift: float
    js_divergence: float
    drifted_segments: list[int]
    new_segments: list[int]
    disappeared_segments: list[int]
    threshold: float
    current_timestamp: str
    previous_timestamp: str
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "has_drift": self.has_drift,
            "max_centroid_drift": round(self.max_centroid_drift, 4),
            "mean_centroid_drift": round(self.mean_centroid_drift, 4),
            "js_divergence": round(self.js_divergence, 4),
            "drifted_segments": self.drifted_segments,
            "new_segments": self.new_segments,
            "disappeared_segments": self.disappeared_segments,
            "threshold": self.threshold,
            "current_timestamp": self.current_timestamp,
            "previous_timestamp": self.previous_timestamp,
        }


class DriftDetector:
    """
    Detects embedding and segmentation drift between clustering runs.

    Metrics:
    1. Centroid cosine distance per segment
    2. Jensen-Shannon divergence on segment size distributions
    3. Segment appearance / disappearance
    """

    def __init__(
        self,
        threshold: float = 0.15,
        dist_threshold: float = 0.10,
    ) -> None:
        self.threshold = threshold
        self.dist_threshold = dist_threshold

    @staticmethod
    def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
        """1 - cosine_similarity between two vectors."""
        a_norm = a / (np.linalg.norm(a) + 1e-9)
        b_norm = b / (np.linalg.norm(b) + 1e-9)
        return float(1.0 - np.dot(a_norm, b_norm))

    @staticmethod
    def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
        """Jensen-Shannon divergence between two probability distributions."""
        p = p / (p.sum() + 1e-9)
        q = q / (q.sum() + 1e-9)
        m = (p + q) / 2.0

        def kl(a: np.ndarray, b: np.ndarray) -> float:
            mask = (a > 0) & (b > 0)
            return float(np.sum(a[mask] * np.log(a[mask] / b[mask])))

        return (kl(p, m) + kl(q, m)) / 2.0

    def compare(
        self,
        current_dir: str,
        previous_dir: str,
    ) -> DriftReport:
        """
        Compare centroids and segment distributions between two clustering runs.

        Args:
            current_dir: Path to current clustering output
            previous_dir: Path to previous clustering output

        Returns:
            DriftReport with drift analysis
        """
        current_centroids = np.load(f"{current_dir}/centroids.npy")
        previous_centroids = np.load(f"{previous_dir}/centroids.npy")

        current_summary = json.loads(Path(f"{current_dir}/clustering_summary.json").read_text())
        previous_summary = json.loads(Path(f"{previous_dir}/clustering_summary.json").read_text())

        current_k = current_centroids.shape[0]
        previous_k = previous_centroids.shape[0]
        min_k = min(current_k, previous_k)

        # ── Centroid drift ────────────────────────────────────────────────────
        current_norm = normalize(current_centroids[:min_k])
        previous_norm = normalize(previous_centroids[:min_k])

        centroid_distances = np.array([
            self._cosine_distance(current_norm[i], previous_norm[i])
            for i in range(min_k)
        ])

        drifted_segments = [
            int(i) for i, d in enumerate(centroid_distances) if d > self.threshold
        ]
        max_drift = float(centroid_distances.max()) if len(centroid_distances) > 0 else 0.0
        mean_drift = float(centroid_distances.mean()) if len(centroid_distances) > 0 else 0.0

        # ── Distribution drift ────────────────────────────────────────────────
        current_sizes = np.array([p["size"] for p in current_summary.get("profiles", [])])
        previous_sizes = np.array([p["size"] for p in previous_summary.get("profiles", [])])

        # Pad to same length
        max_k = max(len(current_sizes), len(previous_sizes))
        current_sizes = np.pad(current_sizes, (0, max_k - len(current_sizes)))
        previous_sizes = np.pad(previous_sizes, (0, max_k - len(previous_sizes)))
        js_div = self._js_divergence(current_sizes.astype(float), previous_sizes.astype(float))

        # ── Segment structure changes ─────────────────────────────────────────
        new_segments = list(range(previous_k, current_k)) if current_k > previous_k else []
        disappeared = list(range(current_k, previous_k)) if previous_k > current_k else []

        has_drift = (
            max_drift > self.threshold
            or js_div > self.dist_threshold
            or len(new_segments) > 0
            or len(disappeared) > 0
        )

        report = DriftReport(
            has_drift=has_drift,
            max_centroid_drift=max_drift,
            mean_centroid_drift=mean_drift,
            js_divergence=js_div,
            drifted_segments=drifted_segments,
            new_segments=new_segments,
            disappeared_segments=disappeared,
            threshold=self.threshold,
            current_timestamp=current_summary.get("run_timestamp", ""),
            previous_timestamp=previous_summary.get("run_timestamp", ""),
        )

        if has_drift:
            logger.warning(
                f"DRIFT DETECTED | max_centroid_drift={max_drift:.3f} | "
                f"js_divergence={js_div:.3f} | drifted_segments={drifted_segments}"
            )
        else:
            logger.info(
                f"No drift detected | max_drift={max_drift:.3f} | js_div={js_div:.3f}"
            )

        return report
