"""
Segment API — FastAPI service for customer segment lookup
=========================================================
Exposes the clustering results for real-time and batch segment queries.

Endpoints:
  GET  /health                → liveness probe
  GET  /v1/segment/{record_id} → lookup segment for a known record ID
  POST /v1/segment/bulk        → lookup segments for multiple record IDs
  GET  /v1/segments/summary    → cluster sizes and labels
  GET  /v1/drift/report        → latest drift report
"""

from __future__ import annotations

import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ─── Schemas ──────────────────────────────────────────────────────────────────

class SegmentLookupResponse(BaseModel):
    record_id: str
    segment_id: int
    segment_label: str
    found: bool


class BulkLookupRequest(BaseModel):
    record_ids: list[str]


class BulkLookupResponse(BaseModel):
    results: list[SegmentLookupResponse]
    found_count: int
    not_found_count: int
    latency_ms: float


class SegmentSummary(BaseModel):
    segment_id: int
    segment_label: str
    size: int
    fraction: float


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

# ─── App State ────────────────────────────────────────────────────────────────

_id_to_segment: dict[str, int] = {}
_cluster_summary: dict = {}


def _load_segments(segment_dir: str) -> None:
    """Load segment labels into memory from persisted clustering output."""
    global _id_to_segment, _cluster_summary
    labels_path = Path(segment_dir) / "segment_labels.json"
    summary_path = Path(segment_dir) / "clustering_summary.json"

    if labels_path.exists():
        _id_to_segment = json.loads(labels_path.read_text())

    if summary_path.exists():
        _cluster_summary = json.loads(summary_path.read_text())


@asynccontextmanager
async def lifespan(app: FastAPI):
    segment_dir = os.getenv("SEGMENT_OUTPUT_DIR", "data/segments/")
    _load_segments(segment_dir)
    yield


# ─── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="BMR Segment Lookup API",
    description="Real-time customer segment lookup service",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health():
    return {"status": "healthy", "segments_loaded": len(_id_to_segment)}


@app.get("/v1/segment/{record_id}", response_model=SegmentLookupResponse)
async def get_segment(record_id: str):
    """Lookup the segment for a single known record ID."""
    if record_id in _id_to_segment:
        seg_id = _id_to_segment[record_id]
        return SegmentLookupResponse(
            record_id=record_id,
            segment_id=seg_id,
            segment_label=SEGMENT_LABELS.get(seg_id, f"segment_{seg_id}"),
            found=True,
        )
    return SegmentLookupResponse(
        record_id=record_id,
        segment_id=-1,
        segment_label="unknown",
        found=False,
    )


@app.post("/v1/segment/bulk", response_model=BulkLookupResponse)
async def bulk_segment_lookup(request: BulkLookupRequest):
    """Batch lookup for multiple record IDs."""
    t0 = time.perf_counter()
    results = []
    found = 0
    not_found = 0

    for rid in request.record_ids:
        if rid in _id_to_segment:
            seg_id = _id_to_segment[rid]
            results.append(SegmentLookupResponse(
                record_id=rid,
                segment_id=seg_id,
                segment_label=SEGMENT_LABELS.get(seg_id, f"segment_{seg_id}"),
                found=True,
            ))
            found += 1
        else:
            results.append(SegmentLookupResponse(
                record_id=rid, segment_id=-1, segment_label="unknown", found=False
            ))
            not_found += 1

    return BulkLookupResponse(
        results=results,
        found_count=found,
        not_found_count=not_found,
        latency_ms=(time.perf_counter() - t0) * 1000,
    )


@app.get("/v1/segments/summary")
async def segments_summary():
    """Return cluster sizes and labels from the latest clustering run."""
    if not _cluster_summary:
        raise HTTPException(status_code=404, detail="No clustering results loaded")

    profiles = _cluster_summary.get("profiles", [])
    total = sum(p["size"] for p in profiles)
    return {
        "algorithm": _cluster_summary.get("algorithm"),
        "n_segments": _cluster_summary.get("n_segments"),
        "silhouette_score": _cluster_summary.get("silhouette"),
        "run_timestamp": _cluster_summary.get("run_timestamp"),
        "segments": [
            SegmentSummary(
                segment_id=p["segment_id"],
                segment_label=SEGMENT_LABELS.get(p["segment_id"], f"segment_{p['segment_id']}"),
                size=p["size"],
                fraction=round(p["size"] / total, 4) if total > 0 else 0,
            )
            for p in profiles
        ],
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=True)
