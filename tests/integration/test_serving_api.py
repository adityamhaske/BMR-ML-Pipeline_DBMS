"""
Integration Tests — Serving API
================================
Tests the FastAPI serving API end-to-end using httpx TestClient.
Requires the full app to be importable (run after `pip install -e .`).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def mock_model_loader():
    """Mock ModelLoader that returns predictable results."""
    loader = MagicMock()
    loader.is_ready = True
    loader.version = "test-v1"
    loader.predict = AsyncMock(return_value={
        "segment_id": 2,
        "segment_label": "high_value_frequent_buyers",
        "confidence": 0.87,
    })
    loader.info.return_value = {
        "model_name": "bmr-customer-segmentation",
        "model_version": "test-v1",
        "stage": "Production",
        "run_id": "test-run-id",
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "n_segments": 8,
        "loaded_at": "2024-04-25T21:00:00",
    }
    return loader


@pytest.fixture
def test_client(mock_model_loader):
    """FastAPI test client with mocked model loader."""
    with patch("serving.api.main._model_loader", mock_model_loader):
        from serving.api.main import app
        with TestClient(app) as client:
            yield client


# ─── Health Endpoint ──────────────────────────────────────────────────────────

@pytest.mark.integration
class TestHealthEndpoint:

    def test_health_returns_200(self, test_client: TestClient) -> None:
        response = test_client.get("/health")
        assert response.status_code == 200

    def test_health_model_loaded(self, test_client: TestClient) -> None:
        data = test_client.get("/health").json()
        assert data["model_loaded"] is True
        assert data["status"] == "healthy"

    def test_health_has_uptime(self, test_client: TestClient) -> None:
        data = test_client.get("/health").json()
        assert "uptime_seconds" in data
        assert data["uptime_seconds"] >= 0


# ─── Segment Prediction ───────────────────────────────────────────────────────

@pytest.mark.integration
class TestSegmentPredict:

    def test_predict_returns_200(self, test_client: TestClient) -> None:
        response = test_client.post(
            "/v1/segment/predict",
            json={"record_id": "cust_001", "text": "Great product, fast shipping!"},
        )
        assert response.status_code == 200

    def test_predict_response_schema(self, test_client: TestClient) -> None:
        data = test_client.post(
            "/v1/segment/predict",
            json={"record_id": "cust_001", "text": "Great product, fast shipping!"},
        ).json()
        assert "record_id" in data
        assert "segment_id" in data
        assert "segment_label" in data
        assert "confidence" in data
        assert "model_version" in data
        assert "latency_ms" in data

    def test_predict_record_id_echoed(self, test_client: TestClient) -> None:
        data = test_client.post(
            "/v1/segment/predict",
            json={"record_id": "my-record-id-123", "text": "Test review text here"},
        ).json()
        assert data["record_id"] == "my-record-id-123"

    def test_predict_confidence_in_range(self, test_client: TestClient) -> None:
        data = test_client.post(
            "/v1/segment/predict",
            json={"record_id": "r1", "text": "Product worked as expected"},
        ).json()
        assert 0.0 <= data["confidence"] <= 1.0

    def test_predict_empty_text_fails(self, test_client: TestClient) -> None:
        response = test_client.post(
            "/v1/segment/predict",
            json={"record_id": "r1", "text": ""},
        )
        assert response.status_code == 422  # Validation error (min_length=1)

    def test_predict_missing_record_id_fails(self, test_client: TestClient) -> None:
        response = test_client.post(
            "/v1/segment/predict",
            json={"text": "Some review text"},
        )
        assert response.status_code == 422


# ─── Batch Prediction ─────────────────────────────────────────────────────────

@pytest.mark.integration
class TestBatchPredict:

    def test_batch_returns_200(self, test_client: TestClient) -> None:
        response = test_client.post(
            "/v1/segment/batch",
            json={
                "records": [
                    {"record_id": "r1", "text": "Great product!"},
                    {"record_id": "r2", "text": "Terrible experience"},
                ]
            },
        )
        assert response.status_code == 200

    def test_batch_total_matches_input(self, test_client: TestClient) -> None:
        n = 5
        data = test_client.post(
            "/v1/segment/batch",
            json={
                "records": [
                    {"record_id": f"r{i}", "text": f"Review text number {i}"}
                    for i in range(n)
                ]
            },
        ).json()
        assert data["total"] == n
        assert len(data["results"]) == n


# ─── Model Info ───────────────────────────────────────────────────────────────

@pytest.mark.integration
class TestModelInfo:

    def test_model_info_returns_200(self, test_client: TestClient) -> None:
        response = test_client.get("/v1/model/info")
        assert response.status_code == 200

    def test_model_info_schema(self, test_client: TestClient) -> None:
        data = test_client.get("/v1/model/info").json()
        required_fields = [
            "model_name", "model_version", "stage",
            "embedding_model", "n_segments", "loaded_at"
        ]
        for field in required_fields:
            assert field in data, f"Missing field: {field}"
