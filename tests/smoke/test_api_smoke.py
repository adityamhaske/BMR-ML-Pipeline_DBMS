"""
Smoke Tests — Running Service
===============================
Runs against a live serving API endpoint.
Usage: pytest tests/smoke/ -m "smoke" --base-url http://localhost:8000
"""

from __future__ import annotations

import os
import pytest
import httpx

BASE_URL = os.getenv("SERVING_URL", "http://localhost:8000")


@pytest.fixture(scope="module")
def client():
    with httpx.Client(base_url=BASE_URL, timeout=10.0) as c:
        yield c


@pytest.mark.smoke
def test_health_endpoint_reachable(client: httpx.Client) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] in ("healthy", "degraded")


@pytest.mark.smoke
def test_docs_reachable(client: httpx.Client) -> None:
    r = client.get("/docs")
    assert r.status_code == 200


@pytest.mark.smoke
def test_single_prediction_returns_segment(client: httpx.Client) -> None:
    r = client.post(
        "/v1/segment/predict",
        json={
            "record_id": "smoke-test-001",
            "text": "The product arrived quickly and was exactly as described. Very satisfied!",
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert "segment_id" in data
    assert isinstance(data["segment_id"], int)
    assert 0 <= data["confidence"] <= 1.0
    assert data["latency_ms"] < 5000  # < 5 seconds


@pytest.mark.smoke
def test_batch_prediction_works(client: httpx.Client) -> None:
    r = client.post(
        "/v1/segment/batch",
        json={
            "records": [
                {"record_id": f"smoke-{i}", "text": f"Test review for smoke test {i}"}
                for i in range(5)
            ]
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert data["total"] == 5


@pytest.mark.smoke
def test_model_info_returns_version(client: httpx.Client) -> None:
    r = client.get("/v1/model/info")
    assert r.status_code == 200
    data = r.json()
    assert data["model_version"] is not None
    assert data["n_segments"] > 0


@pytest.mark.smoke
def test_metrics_endpoint_reachable(client: httpx.Client) -> None:
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "http_requests_total" in r.text or "process_" in r.text
