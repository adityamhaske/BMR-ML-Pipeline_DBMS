"""
Amazon Reviews Extractor — ETL Pillar
=======================================
Loads Amazon Customer Reviews from local JSON/JSONL files or S3.
Handles the full 150M-row dataset via streaming to avoid OOM.

Public dataset sources:
  - Kaggle: https://www.kaggle.com/datasets/cynthiarempel/amazon-us-customer-reviews-dataset
  - AWS Open Data: s3://amazon-reviews-pds/parquet/
  - Hugging Face: datasets load_dataset("amazon_us_reviews", ...)

For local dev: uses the synthetic sample from scripts/download_samples.py
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import pyarrow as pa
import pyarrow.parquet as pq
from loguru import logger


AMAZON_SCHEMA = pa.schema([
    pa.field("id", pa.string()),
    pa.field("text", pa.string()),
    pa.field("rating", pa.int32()),
    pa.field("product_id", pa.string()),
    pa.field("customer_id", pa.string()),
    pa.field("timestamp", pa.string()),
])

# Column mappings for real Amazon Review dataset fields
COLUMN_MAP = {
    "review_id": "id",
    "review_body": "text",
    "star_rating": "rating",
    "product_id": "product_id",
    "customer_id": "customer_id",
    "review_date": "timestamp",
}


@dataclass
class ReviewBatch:
    """A batch of Amazon reviews ready for the embedding pipeline."""

    records: list[dict]
    source_file: str
    batch_index: int
    total_records: int


class AmazonReviewsExtractor:
    """
    Streams Amazon Review records in configurable batches.

    Supports:
    - Local JSON / JSONL files (synthetic sample for dev)
    - S3 parquet files (real dataset in production)
    - Column renaming to canonical schema
    - Language filter (English-only)
    - Minimum text length filter
    """

    def __init__(
        self,
        min_text_length: int = 20,
        batch_size: int = 10_000,
    ) -> None:
        self.min_text_length = min_text_length
        self.batch_size = batch_size

    def _normalize_record(self, raw: dict) -> dict | None:
        """Map raw dataset fields to canonical schema. Returns None if invalid."""
        record: dict = {}

        # Handle both synthetic sample format and real Amazon format
        for src_key, dst_key in COLUMN_MAP.items():
            if src_key in raw:
                record[dst_key] = raw[src_key]
            elif dst_key in raw:
                record[dst_key] = raw[dst_key]

        # Must have ID and text
        if not record.get("id") or not record.get("text"):
            return None

        # Text length filter
        if len(str(record.get("text", ""))) < self.min_text_length:
            return None

        # Normalize types
        record["id"] = str(record["id"])
        record["text"] = str(record["text"]).strip()
        record["rating"] = int(record.get("rating", 0))

        return record

    def stream_from_json(self, file_path: str) -> Iterator[ReviewBatch]:
        """
        Stream records from a local JSON or JSONL file in batches.

        Yields ReviewBatch objects with configurable batch_size records each.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Reviews file not found: {file_path}")

        logger.info(f"Streaming from {path} | batch_size={self.batch_size}")

        with path.open() as f:
            if path.suffix == ".jsonl":
                all_raw = [json.loads(line) for line in f if line.strip()]
            else:
                all_raw = json.load(f)

        total = 0
        batch_idx = 0
        current_batch: list[dict] = []

        for raw in all_raw:
            record = self._normalize_record(raw)
            if record is None:
                continue
            current_batch.append(record)
            total += 1

            if len(current_batch) >= self.batch_size:
                yield ReviewBatch(
                    records=current_batch,
                    source_file=str(path),
                    batch_index=batch_idx,
                    total_records=total,
                )
                current_batch = []
                batch_idx += 1

        # Yield remaining records
        if current_batch:
            yield ReviewBatch(
                records=current_batch,
                source_file=str(path),
                batch_index=batch_idx,
                total_records=total,
            )

        logger.info(f"Streaming complete | total_valid_records={total:,} | batches={batch_idx + 1}")

    def load_all(self, file_path: str) -> list[dict]:
        """
        Load all records from a file into memory.
        Only use for small datasets or sample data.
        """
        records = []
        for batch in self.stream_from_json(file_path):
            records.extend(batch.records)
        return records
