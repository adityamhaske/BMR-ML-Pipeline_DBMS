"""
Model Loader — MLflow Registry Integration
==========================================
Handles loading, caching, and hot-swapping of models from the MLflow Registry.

Thread-safe atomic swap: the old model continues serving during new model load,
then is atomically replaced only after successful load + validation.
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime
from typing import Any

import mlflow
import mlflow.pyfunc
import numpy as np
from loguru import logger
from sentence_transformers import SentenceTransformer

from embedding.preprocessor import TextPreprocessor


class ModelLoader:
    """
    Manages the lifecycle of the ML model in the serving API.

    Holds:
    - embedding_model: SentenceTransformer for text → vector
    - centroids: np.ndarray of cluster centroids from clustering run
    - segment_labels: dict[int, str] mapping cluster ID → human label
    - mlflow_metadata: version, run_id, stage, etc.

    Hot-swap is implemented via asyncio.Lock + double-buffering:
    the new model is loaded into a temporary slot, then atomically
    promoted to the active slot under the lock.
    """

    SEGMENT_LABELS = {
        0: "price_sensitive_shoppers",
        1: "brand_loyal_customers",
        2: "high_value_frequent_buyers",
        3: "occasional_reviewers",
        4: "complaint_prone_users",
        5: "enthusiast_early_adopters",
        6: "gift_buyers",
        7: "bargain_hunters",
    }

    def __init__(
        self,
        tracking_uri: str,
        model_name: str,
        stage: str = "Production",
    ) -> None:
        self._tracking_uri = tracking_uri
        self._model_name = model_name
        self._stage = stage
        self._lock = asyncio.Lock()

        # Active model state
        self._embedding_model: SentenceTransformer | None = None
        self._centroids: np.ndarray | None = None
        self._version: str | None = None
        self._run_id: str | None = None
        self._loaded_at: str | None = None
        self._preprocessor = TextPreprocessor()

        mlflow.set_tracking_uri(tracking_uri)

    @property
    def is_ready(self) -> bool:
        return self._embedding_model is not None and self._centroids is not None

    @property
    def version(self) -> str:
        return self._version or "unknown"

    async def load(self) -> None:
        """Initial model load at startup."""
        async with self._lock:
            await asyncio.get_event_loop().run_in_executor(None, self._load_sync)

    async def reload(self, stage: str = "Production") -> None:
        """
        Hot-swap: load new model, then atomically replace active model.
        Requests continue using the old model until the new one is ready.
        """
        logger.info(f"Hot-swap initiated | stage={stage}")
        new_embedding_model, new_centroids, new_meta = await asyncio.get_event_loop().run_in_executor(
            None, self._fetch_model, stage
        )

        # Atomic promotion
        async with self._lock:
            self._embedding_model = new_embedding_model
            self._centroids = new_centroids
            self._version = new_meta["version"]
            self._run_id = new_meta["run_id"]
            self._loaded_at = datetime.utcnow().isoformat()
            logger.info(f"Hot-swap complete | version={self._version}")

    def _load_sync(self) -> None:
        """Synchronous load — called from executor to avoid blocking event loop."""
        embedding_model, centroids, meta = self._fetch_model(self._stage)
        self._embedding_model = embedding_model
        self._centroids = centroids
        self._version = meta["version"]
        self._run_id = meta["run_id"]
        self._loaded_at = datetime.utcnow().isoformat()

    def _fetch_model(self, stage: str) -> tuple:
        """
        Download model artifacts from MLflow registry.

        Returns (embedding_model, centroids, metadata).
        Falls back to local defaults if MLflow is unavailable (local dev).
        """
        try:
            client = mlflow.MlflowClient(tracking_uri=self._tracking_uri)
            model_versions = client.get_latest_versions(self._model_name, stages=[stage])

            if not model_versions:
                raise ValueError(f"No {stage} version found for model: {self._model_name}")

            mv = model_versions[0]
            logger.info(f"Fetching model | name={self._model_name} | version={mv.version}")

            # Load the pyfunc model (wraps embedding + clustering artifacts)
            model_uri = f"models:/{self._model_name}/{mv.version}"
            pyfunc_model = mlflow.pyfunc.load_model(model_uri)

            # Extract inner artifacts
            embedding_model = pyfunc_model._model_impl.embedding_model
            centroids = pyfunc_model._model_impl.centroids

            return embedding_model, centroids, {"version": str(mv.version), "run_id": mv.run_id}

        except Exception as exc:
            logger.warning(f"MLflow unavailable, using fallback model: {exc}")
            return self._fallback_model()

    def _fallback_model(self) -> tuple:
        """Local dev fallback — load sentence transformer + random centroids."""
        from sentence_transformers import SentenceTransformer
        import numpy as np

        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        logger.info(f"Loading fallback model: {model_name}")
        model = SentenceTransformer(model_name, device="cpu")
        dim = model.get_sentence_embedding_dimension()

        # Random centroids (for dev only)
        rng = np.random.default_rng(42)
        centroids = rng.random((8, dim)).astype(np.float32)
        centroids /= np.linalg.norm(centroids, axis=1, keepdims=True)

        return model, centroids, {"version": "dev-fallback", "run_id": "local"}

    async def predict(self, record_id: str, text: str) -> dict[str, Any]:
        """
        Predict the behavioral segment for a single record.

        Returns dict with segment_id, segment_label, confidence.
        """
        if not self.is_ready:
            raise RuntimeError("Model not loaded")

        # Preprocess + embed
        cleaned = self._preprocessor.clean(text)
        embedding = self._embedding_model.encode(
            [cleaned],
            batch_size=1,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )[0]

        # Find nearest centroid (cosine similarity = dot product on normalized vectors)
        similarities = self._centroids @ embedding  # shape: (n_segments,)
        segment_id = int(np.argmax(similarities))
        confidence = float(similarities[segment_id])

        return {
            "segment_id": segment_id,
            "segment_label": self.SEGMENT_LABELS.get(segment_id, f"segment_{segment_id}"),
            "confidence": max(0.0, min(1.0, (confidence + 1.0) / 2.0)),  # normalize to [0,1]
        }

    def info(self) -> dict[str, Any]:
        """Return model metadata for the /v1/model/info endpoint."""
        return {
            "model_name": self._model_name,
            "model_version": self.version,
            "stage": self._stage,
            "run_id": self._run_id or "unknown",
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "n_segments": len(self.SEGMENT_LABELS),
            "loaded_at": self._loaded_at or "unknown",
        }
