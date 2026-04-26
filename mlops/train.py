"""
MLflow Training Pipeline — Pillar 3: Model Deployment & Versioning
===================================================================
Trains the customer segmentation model and logs everything to MLflow.

What gets logged:
  - Parameters: model name, batch size, n_clusters, algorithm
  - Metrics: silhouette score, Davies-Bouldin, noise fraction, throughput
  - Artifacts: FAISS index, centroids.npy, clustering_summary.json
  - Model: registered as MLflow pyfunc model

Run with:
    python mlops/train.py
    python mlops/train.py --n-clusters 10 --algorithm hdbscan
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import mlflow
import mlflow.pyfunc
import numpy as np
from loguru import logger

from embedding.batch_embedder import BatchEmbedder
from embedding.config import EmbeddingConfig
from embedding.vector_store import FAISSVectorStoreWriter, FAISSVectorStoreReader
from etl.extractors.amazon_reviews_extractor import AmazonReviewsExtractor
from segmentation.clustering import SegmentationEngine


EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "bmr-customer-segmentation")
MODEL_NAME = "bmr-customer-segmentation"


class SegmentationPyfuncModel(mlflow.pyfunc.PythonModel):
    """
    MLflow pyfunc wrapper for the segmentation model.

    Bundles: embedding model + centroids + segment label map.
    predict(context, model_input) accepts a DataFrame with a 'text' column
    and returns segment predictions.
    """

    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        from sentence_transformers import SentenceTransformer

        self.embedding_model = SentenceTransformer(
            context.artifacts["embedding_model_path"],
            device="cpu",
        )
        self.centroids = np.load(context.artifacts["centroids_path"])
        with open(context.artifacts["labels_path"]) as f:
            self.labels: dict = json.load(f)

    def predict(self, context, model_input) -> list[int]:  # noqa: ANN001
        texts = model_input["text"].tolist()
        embeddings = self.embedding_model.encode(
            texts, normalize_embeddings=True, batch_size=64, show_progress_bar=False
        )
        similarities = embeddings @ self.centroids.T  # (n, k)
        return similarities.argmax(axis=1).tolist()


def train(
    input_file: str,
    n_clusters: int = 8,
    algorithm: str = "kmeans",
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 512,
    device: str = "cpu",
) -> str:
    """
    Run the full training pipeline and log to MLflow.

    Returns the MLflow run_id.
    """
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment(EXPERIMENT_NAME)

    logger.info(f"Starting training run | algorithm={algorithm} | n_clusters={n_clusters}")

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        logger.info(f"MLflow run_id: {run_id}")

        # ── Log parameters ────────────────────────────────────────────────────
        mlflow.log_params({
            "embedding_model": embedding_model,
            "batch_size": batch_size,
            "device": device,
            "n_clusters": n_clusters,
            "algorithm": algorithm,
            "input_file": input_file,
        })

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)

            # ── Step 1: Embed ─────────────────────────────────────────────────
            logger.info("Step 1: Generating embeddings")
            extractor = AmazonReviewsExtractor(batch_size=batch_size)
            records = extractor.load_all(input_file)
            mlflow.log_metric("input_record_count", len(records))

            config = EmbeddingConfig(
                model_name=embedding_model, batch_size=batch_size, device=device,
                output_dir=str(tmp_path / "embeddings"),
                checkpoint_dir=str(tmp_path / "checkpoints"),
            )
            embedder = BatchEmbedder(config)
            writer = FAISSVectorStoreWriter(
                output_dir=str(tmp_path / "embeddings"),
                embedding_dim=embedder.embedding_dim,
            )
            stats = embedder.run(records=records, writer=writer, text_field="text", id_field="id")
            writer.flush()

            mlflow.log_metrics({
                "embedding_throughput_rps": stats.throughput_rps,
                "embedding_success_rate": stats.success_rate,
                "processed_records": stats.processed_records,
            })

            # ── Step 2: Cluster ───────────────────────────────────────────────
            logger.info("Step 2: Clustering embeddings")
            import faiss
            reader = FAISSVectorStoreReader(str(tmp_path / "embeddings"))
            index = faiss.read_index(str(tmp_path / "embeddings" / "index.faiss"))
            embeddings = index.reconstruct_n(0, reader.total_vectors)
            id_map = json.loads((tmp_path / "embeddings" / "id_map.json").read_text())

            engine = SegmentationEngine(n_clusters=n_clusters)
            if algorithm == "hdbscan":
                result = engine.run_hdbscan(embeddings, id_map)
            else:
                result = engine.run_kmeans(embeddings, id_map)

            mlflow.log_metrics({
                "silhouette_score": result.silhouette,
                "davies_bouldin_score": result.davies_bouldin,
                "n_segments_discovered": result.n_segments,
                "noise_fraction": result.noise_fraction,
                "clustering_elapsed_seconds": result.elapsed_seconds,
            })

            # ── Step 3: Save artifacts ────────────────────────────────────────
            artifact_dir = tmp_path / "artifacts"
            artifact_dir.mkdir()

            centroids_path = artifact_dir / "centroids.npy"
            labels_path = artifact_dir / "segment_labels.json"
            summary_path = artifact_dir / "clustering_summary.json"

            np.save(str(centroids_path), result.centroids)
            labels_path.write_text(json.dumps(dict(zip(result.record_ids, result.labels.tolist()))))
            engine.save(result, str(artifact_dir))

            mlflow.log_artifacts(str(artifact_dir), artifact_path="clustering")

            # ── Step 4: Register pyfunc model ─────────────────────────────────
            logger.info("Step 4: Registering pyfunc model")
            artifacts = {
                "embedding_model_path": str(tmp_path / "embeddings"),
                "centroids_path": str(centroids_path),
                "labels_path": str(labels_path),
            }
            mlflow.pyfunc.log_model(
                artifact_path="model",
                python_model=SegmentationPyfuncModel(),
                artifacts=artifacts,
                registered_model_name=MODEL_NAME,
                pip_requirements=[
                    "sentence-transformers>=2.6.1",
                    "numpy>=1.26.4",
                    "scikit-learn>=1.4.2",
                    "hdbscan>=0.8.38",
                ],
            )

        logger.info(f"Training complete | run_id={run_id}")
        return run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BMR Segmentation Model Training")
    parser.add_argument("--input", default="data/sample/amazon_reviews_sample.json")
    parser.add_argument("--n-clusters", type=int, default=8)
    parser.add_argument("--algorithm", choices=["kmeans", "hdbscan"], default="kmeans")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    run_id = train(
        input_file=args.input,
        n_clusters=args.n_clusters,
        algorithm=args.algorithm,
        embedding_model=args.model,
        batch_size=args.batch_size,
        device=args.device,
    )
    print(f"\n✓ Training complete | run_id={run_id}")
    print(f"  View at: {os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')}")
