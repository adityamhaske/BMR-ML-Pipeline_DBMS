"""
Unit Tests — Embedding Pipeline
================================
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from embedding.batch_embedder import BatchEmbedder, BatchResult, PipelineStats
from embedding.config import EmbeddingConfig
from embedding.preprocessor import TextPreprocessor
from embedding.vector_store import FAISSVectorStoreWriter


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def embedding_config(tmp_path: Path) -> EmbeddingConfig:
    return EmbeddingConfig(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        batch_size=4,
        device="cpu",
        output_dir=str(tmp_path / "embeddings"),
        checkpoint_dir=str(tmp_path / "checkpoints"),
        fail_fast=True,
    )


@pytest.fixture
def sample_records() -> list[dict]:
    return [
        {"id": f"rec_{i}", "text": f"This is sample review number {i} with some useful content."}
        for i in range(12)
    ]


@pytest.fixture
def preprocessor() -> TextPreprocessor:
    return TextPreprocessor(min_length=5, max_length=512)


# ─── TextPreprocessor Tests ───────────────────────────────────────────────────

class TestTextPreprocessor:

    @pytest.mark.unit
    def test_strips_html(self, preprocessor: TextPreprocessor) -> None:
        result = preprocessor.clean("<p>Hello <b>world</b></p>")
        assert "<" not in result
        assert "Hello" in result
        assert "world" in result

    @pytest.mark.unit
    def test_removes_urls(self, preprocessor: TextPreprocessor) -> None:
        result = preprocessor.clean("Check out https://example.com for more info")
        assert "https://" not in result
        assert "Check out" in result

    @pytest.mark.unit
    def test_normalizes_whitespace(self, preprocessor: TextPreprocessor) -> None:
        result = preprocessor.clean("Hello    world\n\n\tthere")
        assert "  " not in result
        assert "Hello world there" == result

    @pytest.mark.unit
    def test_valid_text(self, preprocessor: TextPreprocessor) -> None:
        assert preprocessor.is_valid("This is a valid review text")
        assert not preprocessor.is_valid("Hi")  # too short
        assert not preprocessor.is_valid("")  # empty

    @pytest.mark.unit
    def test_deduplication(self, preprocessor: TextPreprocessor) -> None:
        text = "This is a duplicate text"
        assert not preprocessor.is_duplicate(text)  # first time: not duplicate
        assert preprocessor.is_duplicate(text)       # second time: duplicate
        preprocessor.reset_dedup_cache()
        assert not preprocessor.is_duplicate(text)  # after reset: not duplicate

    @pytest.mark.unit
    def test_fingerprint_deterministic(self, preprocessor: TextPreprocessor) -> None:
        text = "Hello world"
        fp1 = preprocessor.fingerprint(text)
        fp2 = preprocessor.fingerprint(text)
        assert fp1 == fp2
        assert len(fp1) == 64  # SHA-256 hex

    @pytest.mark.unit
    def test_batch_clean_filters_invalid(self, preprocessor: TextPreprocessor) -> None:
        texts = ["Valid review text here", "X", "", "Another valid text"]
        cleaned, indices = preprocessor.batch_clean(texts, filter_invalid=True)
        assert len(cleaned) == 2
        assert indices == [0, 3]

    @pytest.mark.unit
    def test_batch_clean_deduplicates(self, preprocessor: TextPreprocessor) -> None:
        texts = ["Same valid text here", "Same valid text here", "Different valid text"]
        cleaned, indices = preprocessor.batch_clean(texts, deduplicate=True)
        assert len(cleaned) == 2  # duplicate removed


# ─── BatchEmbedder Tests ──────────────────────────────────────────────────────

class TestBatchEmbedder:

    @pytest.mark.unit
    def test_batch_fingerprint_deterministic(self, embedding_config: EmbeddingConfig) -> None:
        """Same record IDs → same fingerprint."""
        embedder = BatchEmbedder(embedding_config)
        ids = ["a", "b", "c"]
        fp1 = embedder._batch_fingerprint(ids)
        fp2 = embedder._batch_fingerprint(ids)
        assert fp1 == fp2
        assert len(fp1) == 16

    @pytest.mark.unit
    def test_batch_fingerprint_order_independent(self, embedding_config: EmbeddingConfig) -> None:
        """Order of IDs doesn't matter for fingerprint (sorted internally)."""
        embedder = BatchEmbedder(embedding_config)
        fp1 = embedder._batch_fingerprint(["a", "b", "c"])
        fp2 = embedder._batch_fingerprint(["c", "a", "b"])
        assert fp1 == fp2

    @pytest.mark.unit
    def test_idempotency_check(self, embedding_config: EmbeddingConfig, tmp_path: Path) -> None:
        """Already-processed batches are skipped on re-run."""
        embedder = BatchEmbedder(embedding_config)
        record_ids = ["r1", "r2", "r3"]
        batch_id = embedder._batch_fingerprint(record_ids)

        assert not embedder._is_processed(batch_id)
        embedder._mark_processed(batch_id, {"record_count": 3})
        assert embedder._is_processed(batch_id)

    @pytest.mark.unit
    @patch("embedding.batch_embedder.SentenceTransformer")
    def test_process_batch_returns_correct_shape(
        self,
        mock_st: MagicMock,
        embedding_config: EmbeddingConfig,
        sample_records: list[dict],
    ) -> None:
        """Embeddings shape matches (batch_size, embedding_dim)."""
        dim = 384
        mock_model = MagicMock()
        mock_model.encode.return_value = np.random.rand(4, dim).astype(np.float32)
        mock_model.get_sentence_embedding_dimension.return_value = dim
        mock_st.return_value = mock_model

        embedder = BatchEmbedder(embedding_config)
        batch = sample_records[:4]
        result = embedder.process_batch(batch)

        assert not result.skipped
        assert result.embeddings.shape == (4, dim)
        assert len(result.record_ids) == 4

    @pytest.mark.unit
    @patch("embedding.batch_embedder.SentenceTransformer")
    def test_process_batch_idempotent(
        self,
        mock_st: MagicMock,
        embedding_config: EmbeddingConfig,
        sample_records: list[dict],
    ) -> None:
        """Second call for same batch returns skipped=True."""
        dim = 384
        mock_model = MagicMock()
        mock_model.encode.return_value = np.random.rand(4, dim).astype(np.float32)
        mock_model.get_sentence_embedding_dimension.return_value = dim
        mock_st.return_value = mock_model

        embedder = BatchEmbedder(embedding_config)
        batch = sample_records[:4]

        result1 = embedder.process_batch(batch)
        result2 = embedder.process_batch(batch)  # Same batch, should skip

        assert not result1.skipped
        assert result2.skipped


# ─── FAISSVectorStoreWriter Tests ─────────────────────────────────────────────

class TestFAISSVectorStoreWriter:

    @pytest.mark.unit
    def test_write_and_flush(self, tmp_path: Path) -> None:
        """Written embeddings are persisted and index file exists."""
        dim = 384
        writer = FAISSVectorStoreWriter(output_dir=str(tmp_path), embedding_dim=dim)

        ids = ["r1", "r2", "r3"]
        embeddings = np.random.rand(3, dim).astype(np.float32)
        writer.write(ids, embeddings)
        writer.flush()

        assert (tmp_path / "index.faiss").exists()
        assert (tmp_path / "id_map.json").exists()
        assert (tmp_path / "metadata.json").exists()

    @pytest.mark.unit
    def test_id_map_matches_written_ids(self, tmp_path: Path) -> None:
        dim = 384
        writer = FAISSVectorStoreWriter(output_dir=str(tmp_path), embedding_dim=dim)

        ids = ["customer_1", "customer_2"]
        embeddings = np.random.rand(2, dim).astype(np.float32)
        writer.write(ids, embeddings)
        writer.flush()

        id_map = json.loads((tmp_path / "id_map.json").read_text())
        assert id_map == ids

    @pytest.mark.unit
    def test_mismatched_ids_embeddings_raises(self, tmp_path: Path) -> None:
        dim = 384
        writer = FAISSVectorStoreWriter(output_dir=str(tmp_path), embedding_dim=dim)

        with pytest.raises(ValueError, match="Mismatch"):
            writer.write(["only_one_id"], np.random.rand(3, dim).astype(np.float32))

    @pytest.mark.unit
    def test_accumulates_across_multiple_writes(self, tmp_path: Path) -> None:
        dim = 384
        writer = FAISSVectorStoreWriter(output_dir=str(tmp_path), embedding_dim=dim)

        writer.write(["a", "b"], np.random.rand(2, dim).astype(np.float32))
        writer.write(["c", "d"], np.random.rand(2, dim).astype(np.float32))
        writer.flush()

        id_map = json.loads((tmp_path / "id_map.json").read_text())
        assert len(id_map) == 4
