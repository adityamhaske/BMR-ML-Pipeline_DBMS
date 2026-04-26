"""
Batch Embedder — Pillar 1: Customer Segmentation Pipeline
=========================================================
Processes 1M+ customer feedback records through a Sentence Transformer model.

Design principles:
- Chunked inference: process in configurable batches to prevent OOM
- Idempotency: each batch is fingerprinted; already-processed batches are skipped
- Embedding versioning: model name/version stored alongside each vector
- Progress persistence: checkpoints allow resume after failure
- Dual output: local FAISS index + optional S3 upload

Usage:
    python -m embedding.batch_embedder \\
        --input data/sample/amazon_reviews_sample.json \\
        --output data/embeddings/sample/ \\
        --batch-size 512 \\
        --device cpu
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from loguru import logger
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from embedding.config import EmbeddingConfig
from embedding.preprocessor import TextPreprocessor
from embedding.vector_store import VectorStoreWriter


@dataclass
class BatchResult:
    """Result of processing a single batch."""

    batch_id: str
    record_ids: list[str]
    embeddings: np.ndarray
    model_name: str
    model_version: str
    processing_time_ms: float
    skipped: bool = False


@dataclass
class PipelineStats:
    """Accumulated pipeline execution statistics."""

    total_records: int = 0
    processed_records: int = 0
    skipped_records: int = 0
    failed_records: int = 0
    total_batches: int = 0
    elapsed_seconds: float = 0.0
    errors: list[str] = field(default_factory=list)

    @property
    def throughput_rps(self) -> float:
        """Records processed per second."""
        return self.processed_records / max(self.elapsed_seconds, 1e-9)

    @property
    def success_rate(self) -> float:
        """Fraction of records successfully embedded."""
        return self.processed_records / max(self.total_records, 1)


class BatchEmbedder:
    """
    Production-grade batch embedding engine.

    Processes large corpora through a Sentence Transformer model with:
    - Configurable batch size and device placement
    - Checkpoint-based resume on failure
    - Embedding deduplication by record ID
    - Prometheus metrics (if enabled)
    """

    MODEL_VERSION = "1.0"

    def __init__(self, config: EmbeddingConfig) -> None:
        self.config = config
        self.preprocessor = TextPreprocessor()
        self._model: SentenceTransformer | None = None
        self._checkpoint_dir = Path(config.checkpoint_dir)
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"BatchEmbedder initialized | model={config.model_name} "
            f"| batch_size={config.batch_size} | device={config.device}"
        )

    @property
    def model(self) -> SentenceTransformer:
        """Lazy-load model on first access."""
        if self._model is None:
            logger.info(f"Loading model: {self.config.model_name}")
            self._model = SentenceTransformer(
                self.config.model_name,
                device=self.config.device,
                cache_folder=self.config.model_cache_dir,
            )
            # Warm up the model
            _ = self._model.encode(["warm-up"], batch_size=1, show_progress_bar=False)
            logger.info(f"Model loaded | embedding_dim={self._model.get_sentence_embedding_dimension()}")
        return self._model

    @property
    def embedding_dim(self) -> int:
        return self.model.get_sentence_embedding_dimension()

    # ─── Checkpoint management ────────────────────────────────────────────────

    def _batch_fingerprint(self, record_ids: list[str]) -> str:
        """Deterministic hash for a batch — used for idempotency checks."""
        content = "|".join(sorted(record_ids))
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _checkpoint_path(self, batch_id: str) -> Path:
        return self._checkpoint_dir / f"{batch_id}.done"

    def _is_processed(self, batch_id: str) -> bool:
        return self._checkpoint_path(batch_id).exists()

    def _mark_processed(self, batch_id: str, stats: dict) -> None:
        self._checkpoint_path(batch_id).write_text(json.dumps(stats))

    # ─── Core embedding logic ─────────────────────────────────────────────────

    def _embed_batch(self, texts: list[str]) -> np.ndarray:
        """Run inference on a single batch of texts."""
        with torch.no_grad():
            embeddings = self.model.encode(
                texts,
                batch_size=min(self.config.batch_size, len(texts)),
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=self.config.normalize_embeddings,
            )
        return embeddings  # shape: (n, embedding_dim)

    def process_batch(
        self,
        records: list[dict],
        text_field: str = "text",
        id_field: str = "id",
    ) -> BatchResult:
        """
        Process a single batch of records.

        Args:
            records: List of record dicts with at least `id_field` and `text_field`
            text_field: Key to extract text from each record
            id_field: Key to use as record identifier

        Returns:
            BatchResult with embeddings and metadata
        """
        record_ids = [str(r[id_field]) for r in records]
        batch_id = self._batch_fingerprint(record_ids)

        # Idempotency check
        if self._is_processed(batch_id):
            logger.debug(f"Batch {batch_id} already processed, skipping")
            return BatchResult(
                batch_id=batch_id,
                record_ids=record_ids,
                embeddings=np.array([]),
                model_name=self.config.model_name,
                model_version=self.MODEL_VERSION,
                processing_time_ms=0.0,
                skipped=True,
            )

        # Preprocess texts
        texts = [
            self.preprocessor.clean(str(r.get(text_field, "")))
            for r in records
        ]

        # Embed
        t0 = time.perf_counter()
        embeddings = self._embed_batch(texts)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        # Mark done
        self._mark_processed(batch_id, {
            "record_count": len(records),
            "processing_time_ms": elapsed_ms,
            "model": self.config.model_name,
        })

        logger.debug(
            f"Batch {batch_id} | {len(records)} records "
            f"| {elapsed_ms:.0f}ms | {len(records) / (elapsed_ms / 1000):.0f} rec/s"
        )

        return BatchResult(
            batch_id=batch_id,
            record_ids=record_ids,
            embeddings=embeddings,
            model_name=self.config.model_name,
            model_version=self.MODEL_VERSION,
            processing_time_ms=elapsed_ms,
        )

    # ─── Streaming pipeline ───────────────────────────────────────────────────

    def _iter_batches(
        self,
        records: list[dict],
        batch_size: int,
    ) -> Iterator[list[dict]]:
        """Yield chunks of `batch_size` records."""
        for i in range(0, len(records), batch_size):
            yield records[i : i + batch_size]

    def run(
        self,
        records: list[dict],
        writer: VectorStoreWriter,
        text_field: str = "text",
        id_field: str = "id",
    ) -> PipelineStats:
        """
        Run the full embedding pipeline over a dataset.

        Args:
            records: All records to embed
            writer: VectorStoreWriter that persists embeddings
            text_field: Field name containing the text to embed
            id_field: Field name to use as unique record identifier

        Returns:
            PipelineStats with throughput and quality metrics
        """
        stats = PipelineStats(total_records=len(records))
        start_time = time.perf_counter()

        logger.info(
            f"Starting embedding pipeline | "
            f"total_records={len(records):,} | "
            f"batch_size={self.config.batch_size}"
        )

        batches = list(self._iter_batches(records, self.config.batch_size))
        stats.total_batches = len(batches)

        for batch_idx, batch in enumerate(tqdm(batches, desc="Embedding batches", unit="batch")):
            try:
                result = self.process_batch(batch, text_field=text_field, id_field=id_field)

                if result.skipped:
                    stats.skipped_records += len(batch)
                    continue

                # Write embeddings to vector store
                writer.write(
                    record_ids=result.record_ids,
                    embeddings=result.embeddings,
                    metadata={
                        "batch_id": result.batch_id,
                        "model_name": result.model_name,
                        "model_version": result.model_version,
                    },
                )

                stats.processed_records += len(batch)

            except Exception as exc:  # noqa: BLE001
                logger.error(f"Batch {batch_idx} failed: {exc}")
                stats.failed_records += len(batch)
                stats.errors.append(str(exc))

                if self.config.fail_fast:
                    raise

        stats.elapsed_seconds = time.perf_counter() - start_time

        logger.info(
            f"Pipeline complete | "
            f"processed={stats.processed_records:,} | "
            f"skipped={stats.skipped_records:,} | "
            f"failed={stats.failed_records:,} | "
            f"throughput={stats.throughput_rps:.0f} rec/s | "
            f"success_rate={stats.success_rate:.1%} | "
            f"elapsed={stats.elapsed_seconds:.1f}s"
        )

        return stats


# ─── CLI entrypoint ───────────────────────────────────────────────────────────

def _load_records(input_path: str) -> list[dict]:
    """Load records from JSON or JSONL file."""
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    with path.open() as f:
        if path.suffix == ".jsonl":
            return [json.loads(line) for line in f if line.strip()]
        return json.load(f)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="BMR Batch Embedding Pipeline")
    parser.add_argument("--input", required=True, help="Input JSON/JSONL file or S3 URI")
    parser.add_argument("--output", required=True, help="Output directory or S3 URI")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--id-field", default="id")
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()

    from embedding.config import EmbeddingConfig
    from embedding.vector_store import FAISSVectorStoreWriter

    config = EmbeddingConfig(
        model_name=args.model,
        batch_size=args.batch_size,
        device=args.device,
        output_dir=args.output,
        fail_fast=args.fail_fast,
    )

    embedder = BatchEmbedder(config)
    records = _load_records(args.input)
    writer = FAISSVectorStoreWriter(output_dir=args.output, embedding_dim=embedder.embedding_dim)

    pipeline_stats = embedder.run(
        records=records,
        writer=writer,
        text_field=args.text_field,
        id_field=args.id_field,
    )

    print(f"\n✓ Done — {pipeline_stats.processed_records:,} records at "
          f"{pipeline_stats.throughput_rps:.0f} rec/s")
