"""
Pytest shared fixtures and configuration.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest


# ─── Global test environment overrides ───────────────────────────────────────

os.environ.setdefault("ENVIRONMENT", "test")
os.environ.setdefault("USE_LOCALSTACK", "false")
os.environ.setdefault("EMBEDDING_DEVICE", "cpu")
os.environ.setdefault("EMBEDDING_BATCH_SIZE", "16")
os.environ.setdefault("DUCKDB_PATH", ":memory:")


# ─── Shared fixtures ──────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def sample_reviews() -> list[dict]:
    """100 synthetic review records for testing."""
    return [
        {
            "id": f"rev_{i:04d}",
            "text": f"This is review number {i}. The product quality was {'good' if i % 2 == 0 else 'bad'}.",
            "rating": (i % 5) + 1,
            "product_id": f"ASIN_{i % 20:04d}",
            "customer_id": f"CUST_{i % 30:04d}",
        }
        for i in range(100)
    ]


@pytest.fixture(scope="session")
def sample_reviews_file(sample_reviews: list[dict], tmp_path_factory) -> str:
    """Write sample reviews to a temp JSON file."""
    p = tmp_path_factory.mktemp("data") / "reviews.json"
    p.write_text(json.dumps(sample_reviews))
    return str(p)


@pytest.fixture
def random_embeddings() -> np.ndarray:
    """384-dim random embeddings (all-MiniLM-L6-v2 dimension)."""
    rng = np.random.default_rng(42)
    emb = rng.random((50, 384)).astype(np.float32)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    return emb / norms


@pytest.fixture
def duckdb_loader():
    """In-memory DuckDB loader for tests."""
    from etl.loaders.redshift_loader import DuckDBLoader
    loader = DuckDBLoader(db_path=":memory:")
    yield loader
    loader.close()
