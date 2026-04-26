"""
Model Serving API — Pillar 3: Model Deployment & Versioning
===========================================================
FastAPI application serving customer segment predictions and model inference.

Endpoints:
  GET  /health                    → Health check (liveness + readiness)
  GET  /metrics                   → Prometheus metrics
  POST /v1/segment/predict        → Return segment ID for a customer record
  POST /v1/segment/batch          → Batch segment prediction
  GET  /v1/model/info             → Current model version and metadata
  POST /v1/model/reload           → Hot-swap model from MLflow registry (admin)

Design:
- Model is loaded once at startup from MLflow Model Registry
- Hot-swap: PATCH /v1/model/reload atomically replaces the loaded model
- OpenTelemetry tracing on all routes
- Prometheus metrics: request count, latency histograms, model version gauge
"""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, Field

from serving.model_loader import ModelLoader

# ─── Pydantic Schemas ─────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    record_id: str = Field(..., description="Unique customer/record identifier")
    text: str = Field(..., min_length=1, max_length=10_000, description="Customer feedback text")
    metadata: dict[str, Any] = Field(default_factory=dict)

    class Config:
        json_schema_extra = {
            "example": {
                "record_id": "cust_12345",
                "text": "Great product, fast shipping! Would definitely buy again.",
            }
        }


class BatchPredictRequest(BaseModel):
    records: list[PredictRequest] = Field(..., min_length=1, max_length=500)


class SegmentResponse(BaseModel):
    record_id: str
    segment_id: int
    segment_label: str
    confidence: float
    model_version: str
    latency_ms: float


class BatchSegmentResponse(BaseModel):
    results: list[SegmentResponse]
    total: int
    model_version: str
    latency_ms: float


class ModelInfoResponse(BaseModel):
    model_name: str
    model_version: str
    stage: str
    run_id: str
    embedding_model: str
    n_segments: int
    loaded_at: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str | None
    uptime_seconds: float


# ─── App State ────────────────────────────────────────────────────────────────

_startup_time = time.time()
_model_loader: ModelLoader | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup; cleanup on shutdown."""
    global _model_loader

    logger.info("Starting BMR Serving API...")
    _model_loader = ModelLoader(
        tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"),
        model_name=os.getenv("MLFLOW_MODEL_NAME", "bmr-customer-segmentation"),
        stage=os.getenv("MLFLOW_MODEL_STAGE", "Production"),
    )
    await _model_loader.load()
    logger.info(f"Model loaded | version={_model_loader.version}")

    yield  # App is running

    logger.info("Shutting down BMR Serving API...")


# ─── App Factory ──────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    app = FastAPI(
        title="BMR ML Serving API",
        description=(
            "Customer segmentation and model inference service. "
            "Powers precision behavioral targeting for marketing campaigns."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # ── Middleware ────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET", "POST", "PATCH"],
        allow_headers=["*"],
    )

    # ── Prometheus instrumentation ────────────────────────────────────────────
    Instrumentator(
        should_group_status_codes=True,
        should_ignore_untemplated=True,
        should_respect_env_var=True,
        excluded_handlers=["/health", "/metrics"],
    ).instrument(app).expose(app, endpoint="/metrics")

    # ── Routes ────────────────────────────────────────────────────────────────

    @app.get("/health", response_model=HealthResponse, tags=["ops"])
    async def health():
        """Liveness + readiness probe."""
        return HealthResponse(
            status="healthy" if _model_loader and _model_loader.is_ready else "degraded",
            model_loaded=_model_loader is not None and _model_loader.is_ready,
            model_version=_model_loader.version if _model_loader else None,
            uptime_seconds=time.time() - _startup_time,
        )

    @app.post(
        "/v1/segment/predict",
        response_model=SegmentResponse,
        tags=["inference"],
        summary="Predict customer segment for a single record",
    )
    async def predict_segment(request: PredictRequest):
        """
        Return the behavioral segment ID for a customer record.

        The model embeds the `text` field, then looks up the nearest
        cluster centroid to assign a segment.
        """
        if not _model_loader or not _model_loader.is_ready:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model not loaded",
            )

        t0 = time.perf_counter()
        try:
            result = await _model_loader.predict(
                record_id=request.record_id,
                text=request.text,
            )
        except Exception as exc:
            logger.error(f"Prediction failed for record_id={request.record_id}: {exc}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Prediction error: {exc}",
            )

        latency_ms = (time.perf_counter() - t0) * 1000
        return SegmentResponse(
            record_id=request.record_id,
            segment_id=result["segment_id"],
            segment_label=result["segment_label"],
            confidence=result["confidence"],
            model_version=_model_loader.version,
            latency_ms=latency_ms,
        )

    @app.post(
        "/v1/segment/batch",
        response_model=BatchSegmentResponse,
        tags=["inference"],
        summary="Batch segment prediction (up to 500 records)",
    )
    async def batch_predict(request: BatchPredictRequest):
        """Batch version of /v1/segment/predict — processes up to 500 records in one call."""
        if not _model_loader or not _model_loader.is_ready:
            raise HTTPException(status_code=503, detail="Model not loaded")

        t0 = time.perf_counter()
        results = []

        for record in request.records:
            try:
                result = await _model_loader.predict(
                    record_id=record.record_id,
                    text=record.text,
                )
                results.append(SegmentResponse(
                    record_id=record.record_id,
                    segment_id=result["segment_id"],
                    segment_label=result["segment_label"],
                    confidence=result["confidence"],
                    model_version=_model_loader.version,
                    latency_ms=0.0,
                ))
            except Exception as exc:
                logger.warning(f"Skipping {record.record_id}: {exc}")

        total_latency_ms = (time.perf_counter() - t0) * 1000
        return BatchSegmentResponse(
            results=results,
            total=len(results),
            model_version=_model_loader.version,
            latency_ms=total_latency_ms,
        )

    @app.get(
        "/v1/model/info",
        response_model=ModelInfoResponse,
        tags=["model-registry"],
        summary="Get current model version and metadata",
    )
    async def model_info():
        """Return metadata about the currently loaded model."""
        if not _model_loader:
            raise HTTPException(status_code=503, detail="Model not loaded")
        return _model_loader.info()

    @app.post(
        "/v1/model/reload",
        tags=["model-registry"],
        summary="Hot-swap model from MLflow registry (zero-downtime)",
    )
    async def reload_model(stage: str = "Production"):
        """
        Atomically reload the model from the MLflow registry.
        Enables zero-downtime model updates — the old model continues
        serving requests until the new one is fully loaded.
        """
        if not _model_loader:
            raise HTTPException(status_code=503, detail="Loader not initialized")

        old_version = _model_loader.version
        await _model_loader.reload(stage=stage)
        new_version = _model_loader.version

        logger.info(f"Model hot-swapped | {old_version} → {new_version}")
        return {
            "status": "reloaded",
            "previous_version": old_version,
            "new_version": new_version,
        }

    return app


app = create_app()


if __name__ == "__main__":
    uvicorn.run(
        "serving.api.main:app",
        host=os.getenv("SERVING_HOST", "0.0.0.0"),
        port=int(os.getenv("SERVING_PORT", "8000")),
        workers=int(os.getenv("SERVING_WORKERS", "1")),
        log_level=os.getenv("SERVING_LOG_LEVEL", "info"),
        reload=os.getenv("ENVIRONMENT", "local") == "local",
    )
