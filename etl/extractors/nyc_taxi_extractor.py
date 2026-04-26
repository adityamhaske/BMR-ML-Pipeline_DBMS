"""
NYC Taxi Extractor — ETL Pillar
================================
Handles downloading, validating, and staging NYC Yellow Taxi parquet files
from NYC Open Data. Designed for monthly batch ingestion.

Design:
- Retryable with exponential backoff (tenacity)
- Validates parquet schema before returning
- Returns metadata for downstream tasks
"""

from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq
from loguru import logger
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)


NYC_TAXI_BASE_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data"

REQUIRED_COLUMNS = {
    "tpep_pickup_datetime",
    "tpep_dropoff_datetime",
    "passenger_count",
    "trip_distance",
    "fare_amount",
    "total_amount",
    "PULocationID",
    "DOLocationID",
}


@dataclass
class ExtractionResult:
    """Metadata about a completed extraction."""

    source_url: str
    local_path: str
    s3_key: str
    year: str
    month: str
    row_count: int
    file_size_bytes: int
    schema_valid: bool


class NYCTaxiExtractor:
    """
    Downloads and validates NYC Yellow Taxi monthly parquet files.

    Supports:
    - Yellow Taxi (tpep_* timestamps)
    - Configurable download directory
    - Schema validation against required columns
    - File-level deduplication (skip already-downloaded files)
    """

    def __init__(self, download_dir: str = "/tmp/bmr-etl/nyc-taxi") -> None:
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)

    def _build_url(self, year: int, month: int) -> str:
        return f"{NYC_TAXI_BASE_URL}/yellow_tripdata_{year}-{month:02d}.parquet"

    def _build_s3_key(self, year: int, month: int, filename: str) -> str:
        """Hive-compatible S3 partition key."""
        return f"raw/nyc_taxi/year={year}/month={month:02d}/{filename}"

    @retry(
        retry=retry_if_exception_type((urllib.error.URLError, OSError)),
        wait=wait_exponential(multiplier=1, min=10, max=300),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    def _download(self, url: str, local_path: Path) -> None:
        """Download with retry and progress logging."""
        logger.info(f"Downloading: {url}")
        urllib.request.urlretrieve(url, str(local_path))
        logger.info(f"Download complete: {local_path} ({local_path.stat().st_size / 1e6:.1f} MB)")

    def validate_schema(self, local_path: Path) -> tuple[bool, list[str]]:
        """
        Validate parquet schema against required columns.

        Returns (is_valid, missing_columns).
        """
        metadata = pq.read_metadata(str(local_path))
        schema = pq.read_schema(str(local_path))
        actual_cols = set(schema.names)
        missing = REQUIRED_COLUMNS - actual_cols
        return len(missing) == 0, list(missing)

    def extract(self, year: int, month: int) -> ExtractionResult:
        """
        Download and validate one month of NYC Taxi data.

        Args:
            year: e.g. 2024
            month: 1–12

        Returns:
            ExtractionResult with file metadata
        """
        url = self._build_url(year, month)
        filename = f"yellow_tripdata_{year}-{month:02d}.parquet"
        local_path = self.download_dir / filename

        # Idempotency — skip if already downloaded
        if local_path.exists():
            logger.info(f"File already exists, skipping download: {local_path}")
        else:
            self._download(url, local_path)

        # Validate
        is_valid, missing_cols = self.validate_schema(local_path)
        if not is_valid:
            raise ValueError(
                f"Schema validation failed for {filename}. "
                f"Missing columns: {missing_cols}"
            )

        meta = pq.read_metadata(str(local_path))
        row_count = meta.num_rows
        logger.info(f"Extracted {row_count:,} rows | {filename}")

        return ExtractionResult(
            source_url=url,
            local_path=str(local_path),
            s3_key=self._build_s3_key(year, month, filename),
            year=str(year),
            month=f"{month:02d}",
            row_count=row_count,
            file_size_bytes=local_path.stat().st_size,
            schema_valid=True,
        )
