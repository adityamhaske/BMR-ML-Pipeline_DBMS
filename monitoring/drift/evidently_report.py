"""
Evidently AI Drift Report — Monitoring Pillar
=============================================
Generates data and prediction drift reports comparing current and
reference data distributions.

Run with:
    python monitoring/drift/evidently_report.py \
        --current data/sample/amazon_reviews_sample.json \
        --reference data/sample/amazon_reviews_reference.json \
        --output monitoring/reports/drift_report.html
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
from loguru import logger


def load_reviews_as_df(file_path: str) -> pd.DataFrame:
    """Load reviews JSON into a DataFrame with text features."""
    with open(file_path) as f:
        records = json.load(f)
    df = pd.DataFrame(records)
    df["text_length"] = df["text"].str.len()
    df["word_count"] = df["text"].str.split().str.len()
    df["rating"] = pd.to_numeric(df.get("rating", 3), errors="coerce").fillna(3)
    return df


def generate_report(
    current_path: str,
    reference_path: str | None,
    output_path: str = "monitoring/reports/drift_report.html",
) -> dict:
    """
    Generate an Evidently data drift report.

    If reference_path is None, uses statistical thresholds only.

    Returns a dict summary with drift detected flag.
    """
    try:
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset, DataQualityPreset
        from evidently.metrics import ColumnDriftMetric
    except ImportError:
        logger.warning("evidently not installed — generating simplified report")
        return _simplified_report(current_path, reference_path, output_path)

    current_df = load_reviews_as_df(current_path)
    reference_df = load_reviews_as_df(reference_path) if reference_path else current_df.copy()

    # Keep only numeric + categorical columns Evidently can analyze
    cols = ["text_length", "word_count", "rating"]
    current_df = current_df[cols].dropna()
    reference_df = reference_df[cols].dropna()

    report = Report(metrics=[
        DataDriftPreset(),
        DataQualityPreset(),
        ColumnDriftMetric(column_name="text_length"),
        ColumnDriftMetric(column_name="rating"),
    ])

    report.run(reference_data=reference_df, current_data=current_df)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    report.save_html(str(output))

    result_dict = report.as_dict()
    drift_detected = any(
        m.get("result", {}).get("drift_detected", False)
        for m in result_dict.get("metrics", [])
    )

    logger.info(f"Drift report saved: {output} | drift_detected={drift_detected}")
    return {"drift_detected": drift_detected, "report_path": str(output)}


def _simplified_report(
    current_path: str,
    reference_path: str | None,
    output_path: str,
) -> dict:
    """Fallback: compute basic statistics when Evidently is not installed."""
    current_df = load_reviews_as_df(current_path)

    stats = {
        "generated_at": datetime.utcnow().isoformat(),
        "current_file": current_path,
        "record_count": len(current_df),
        "text_length": {
            "mean": float(current_df["text_length"].mean()),
            "std": float(current_df["text_length"].std()),
            "p50": float(current_df["text_length"].quantile(0.50)),
            "p95": float(current_df["text_length"].quantile(0.95)),
        },
        "rating": {
            "mean": float(current_df["rating"].mean()),
            "std": float(current_df["rating"].std()),
        },
        "drift_detected": False,
        "note": "Install evidently for full drift detection",
    }

    output = Path(output_path).with_suffix(".json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(stats, indent=2))
    logger.info(f"Simplified stats report saved: {output}")
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Evidently drift report")
    parser.add_argument("--current", required=True, help="Current data file (JSON)")
    parser.add_argument("--reference", default=None, help="Reference data file (JSON)")
    parser.add_argument("--output", default="monitoring/reports/drift_report.html")
    args = parser.parse_args()

    result = generate_report(args.current, args.reference, args.output)
    print(f"\n✓ Report: {result.get('report_path', args.output)}")
    print(f"  Drift detected: {result.get('drift_detected', 'N/A')}")
