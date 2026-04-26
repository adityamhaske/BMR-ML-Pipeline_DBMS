"""
Vector Store — FAISS and pgvector backends
==========================================
Provides a unified interface for writing and reading embedding vectors.

Two backends:
1. FAISSVectorStoreWriter  — Local file-based FAISS index (dev + batch jobs)
2. PGVectorStoreWriter     — PostgreSQL pgvector extension (production serving)
"""

from __future__ import annotations

import json
import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import psycopg2
import psycopg2.extras
from loguru import logger


class VectorStoreWriter(ABC):
    """Abstract base class for vector store backends."""

    @abstractmethod
    def write(
        self,
        record_ids: list[str],
        embeddings: np.ndarray,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Persist a batch of embeddings with their record IDs."""
        ...

    @abstractmethod
    def flush(self) -> None:
        """Flush any buffered writes to durable storage."""
        ...


# ─── FAISS Backend ────────────────────────────────────────────────────────────

class FAISSVectorStoreWriter(VectorStoreWriter):
    """
    Writes embeddings to a local FAISS IndexFlatIP (inner product / cosine similarity).

    Files created:
        {output_dir}/index.faiss       — FAISS binary index
        {output_dir}/id_map.json       — Maps FAISS integer ID → record string ID
        {output_dir}/metadata.json     — Pipeline metadata (model name, counts, etc.)

    The index is append-safe: multiple write() calls accumulate vectors.
    Call flush() to persist to disk.
    """

    def __init__(self, output_dir: str, embedding_dim: int) -> None:
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._embedding_dim = embedding_dim

        # Load existing index if present
        index_path = self._output_dir / "index.faiss"
        id_map_path = self._output_dir / "id_map.json"

        if index_path.exists():
            self._index = faiss.read_index(str(index_path))
            self._id_map: list[str] = json.loads(id_map_path.read_text())
            logger.info(f"Loaded existing FAISS index with {self._index.ntotal:,} vectors")
        else:
            # IndexFlatIP requires normalized vectors (L2 norm = 1) for cosine similarity
            self._index = faiss.IndexFlatIP(embedding_dim)
            self._id_map = []
            logger.info(f"Created new FAISS IndexFlatIP | dim={embedding_dim}")

        self._pending_ids: list[str] = []
        self._pending_embeddings: list[np.ndarray] = []
        self._pipeline_metadata: dict[str, Any] = {}

    def write(
        self,
        record_ids: list[str],
        embeddings: np.ndarray,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Buffer embeddings and IDs. Call flush() to commit."""
        if len(record_ids) != embeddings.shape[0]:
            raise ValueError(
                f"Mismatch: {len(record_ids)} IDs but {embeddings.shape[0]} embeddings"
            )

        self._pending_ids.extend(record_ids)
        self._pending_embeddings.append(embeddings.astype(np.float32))

        if metadata:
            self._pipeline_metadata.update(metadata)

        # Auto-flush every 50K vectors to bound memory
        if len(self._pending_ids) >= 50_000:
            self.flush()

    def flush(self) -> None:
        """Commit all buffered vectors to the FAISS index and save to disk."""
        if not self._pending_ids:
            return

        combined = np.vstack(self._pending_embeddings).astype(np.float32)
        faiss.normalize_L2(combined)  # Ensure cosine similarity correctness

        self._index.add(combined)
        self._id_map.extend(self._pending_ids)

        # Persist
        faiss.write_index(self._index, str(self._output_dir / "index.faiss"))
        (self._output_dir / "id_map.json").write_text(json.dumps(self._id_map))
        (self._output_dir / "metadata.json").write_text(
            json.dumps({
                "total_vectors": self._index.ntotal,
                "embedding_dim": self._embedding_dim,
                **self._pipeline_metadata,
            }, indent=2)
        )

        logger.info(
            f"FAISS flush | added={len(self._pending_ids):,} | "
            f"total={self._index.ntotal:,}"
        )

        self._pending_ids.clear()
        self._pending_embeddings.clear()

    def __del__(self) -> None:
        """Ensure pending writes are flushed on destruction."""
        try:
            self.flush()
        except Exception:  # noqa: BLE001
            pass


# ─── pgvector Backend ─────────────────────────────────────────────────────────

class PGVectorStoreWriter(VectorStoreWriter):
    """
    Writes embeddings to a PostgreSQL table using the pgvector extension.

    Table schema (created automatically if not exists):
        CREATE TABLE embeddings (
            id          SERIAL PRIMARY KEY,
            record_id   TEXT NOT NULL UNIQUE,
            embedding   VECTOR({dim}),
            model_name  TEXT,
            model_version TEXT,
            created_at  TIMESTAMPTZ DEFAULT NOW()
        );

    Uses ON CONFLICT DO NOTHING for idempotent upserts.
    """

    _CREATE_TABLE_SQL = """
        CREATE EXTENSION IF NOT EXISTS vector;
        CREATE TABLE IF NOT EXISTS embeddings (
            id            SERIAL PRIMARY KEY,
            record_id     TEXT NOT NULL,
            embedding     VECTOR({dim}),
            model_name    TEXT,
            model_version TEXT,
            created_at    TIMESTAMPTZ DEFAULT NOW(),
            CONSTRAINT uq_record_id UNIQUE (record_id)
        );
        CREATE INDEX IF NOT EXISTS embeddings_vector_idx
            ON embeddings USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100);
    """

    def __init__(
        self,
        dsn: str,
        embedding_dim: int,
        table_name: str = "embeddings",
        buffer_size: int = 1000,
    ) -> None:
        self._dsn = dsn
        self._embedding_dim = embedding_dim
        self._table_name = table_name
        self._buffer_size = buffer_size
        self._buffer: list[tuple] = []

        self._conn = psycopg2.connect(dsn)
        self._conn.autocommit = False
        self._ensure_table()

    def _ensure_table(self) -> None:
        with self._conn.cursor() as cur:
            cur.execute(self._CREATE_TABLE_SQL.format(dim=self._embedding_dim))
        self._conn.commit()
        logger.info(f"pgvector table '{self._table_name}' ready | dim={self._embedding_dim}")

    def write(
        self,
        record_ids: list[str],
        embeddings: np.ndarray,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        model_name = (metadata or {}).get("model_name", "unknown")
        model_version = (metadata or {}).get("model_version", "unknown")

        for rid, emb in zip(record_ids, embeddings):
            self._buffer.append((rid, emb.tolist(), model_name, model_version))

        if len(self._buffer) >= self._buffer_size:
            self.flush()

    def flush(self) -> None:
        if not self._buffer:
            return

        insert_sql = f"""
            INSERT INTO {self._table_name} (record_id, embedding, model_name, model_version)
            VALUES %s
            ON CONFLICT (record_id) DO NOTHING
        """
        template = "(%s, %s::vector, %s, %s)"

        with self._conn.cursor() as cur:
            psycopg2.extras.execute_values(cur, insert_sql, self._buffer, template=template)
        self._conn.commit()

        logger.info(f"pgvector flush | committed={len(self._buffer):,} vectors")
        self._buffer.clear()

    def close(self) -> None:
        self.flush()
        self._conn.close()


# ─── Reader (for segmentation service) ───────────────────────────────────────

class FAISSVectorStoreReader:
    """Read-only access to a FAISS index for similarity search."""

    def __init__(self, index_dir: str) -> None:
        self._index_dir = Path(index_dir)
        self._index = faiss.read_index(str(self._index_dir / "index.faiss"))
        self._id_map: list[str] = json.loads((self._index_dir / "id_map.json").read_text())
        self._metadata: dict = json.loads((self._index_dir / "metadata.json").read_text())

        logger.info(
            f"FAISSVectorStoreReader | vectors={self._index.ntotal:,} | "
            f"dim={self._metadata.get('embedding_dim')}"
        )

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
    ) -> list[dict]:
        """
        Find the top-k most similar vectors.

        Returns:
            List of dicts with `record_id` and `score` (cosine similarity)
        """
        query = query_vector.astype(np.float32).reshape(1, -1)
        faiss.normalize_L2(query)
        scores, indices = self._index.search(query, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append({
                "record_id": self._id_map[idx],
                "score": float(score),
            })
        return results

    @property
    def total_vectors(self) -> int:
        return self._index.ntotal
