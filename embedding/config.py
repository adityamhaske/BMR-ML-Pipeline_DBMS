"""
Embedding Configuration — Hydra/Pydantic Settings
===================================================
"""
from __future__ import annotations

import os
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class EmbeddingConfig(BaseSettings):
    """
    Configuration for the batch embedding pipeline.
    Values can be set via environment variables (prefixed EMBEDDING_)
    or passed directly at construction time.
    """

    model_config = SettingsConfigDict(
        env_prefix="EMBEDDING_",
        env_file=".env",
        extra="ignore",
    )

    # ─── Model ────────────────────────────────────────────────────────────────
    model_name: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="HuggingFace model ID or local path",
    )
    model_cache_dir: str = Field(
        default=str(Path.home() / ".cache" / "sentence_transformers"),
        description="Local directory to cache downloaded models",
    )
    normalize_embeddings: bool = Field(
        default=True,
        description="L2-normalize embeddings (required for cosine similarity FAISS index)",
    )

    # ─── Inference ────────────────────────────────────────────────────────────
    batch_size: int = Field(default=512, ge=1, le=4096)
    device: str = Field(
        default="cpu",
        description="PyTorch device: cpu | cuda | mps",
    )
    workers: int = Field(default=4, ge=1, description="Parallel data loader workers")

    # ─── Output ───────────────────────────────────────────────────────────────
    output_dir: str = Field(default="data/embeddings/")
    checkpoint_dir: str = Field(default="data/embeddings/.checkpoints/")
    faiss_index_path: str = Field(default="data/faiss_index/")

    # ─── Behavior ─────────────────────────────────────────────────────────────
    fail_fast: bool = Field(
        default=False,
        description="Raise on first batch failure instead of continuing",
    )
    min_text_length: int = Field(default=10, description="Minimum character length after cleaning")
    max_text_length: int = Field(default=512, description="Maximum tokens per text (model truncates beyond this)")
