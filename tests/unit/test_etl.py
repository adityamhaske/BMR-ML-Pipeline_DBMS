"""
ETL Unit Tests — Extractors, Transformers, Loaders
====================================================
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch, MagicMock

import pandas as pd
import numpy as np
import pytest

from etl.extractors.amazon_reviews_extractor import AmazonReviewsExtractor
from etl.transformers.feature_engineer import NYCTaxiFeatureEngineer, ReviewFeatureEngineer


# ─── Amazon Reviews Extractor Tests ───────────────────────────────────────────

class TestAmazonReviewsExtractor:

    @pytest.mark.unit
    def test_load_all_returns_records(self, sample_reviews_file: str) -> None:
        extractor = AmazonReviewsExtractor(min_text_length=10, batch_size=50)
        records = extractor.load_all(sample_reviews_file)
        assert len(records) > 0

    @pytest.mark.unit
    def test_normalizes_record_fields(self, sample_reviews_file: str) -> None:
        extractor = AmazonReviewsExtractor(min_text_length=1)
        records = extractor.load_all(sample_reviews_file)
        assert all("id" in r for r in records)
        assert all("text" in r for r in records)

    @pytest.mark.unit
    def test_filters_short_texts(self) -> None:
        import json, tempfile
        short_records = [{"id": "r1", "text": "Hi"}, {"id": "r2", "text": "Great product overall!"}]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(short_records, f)
            fname = f.name

        extractor = AmazonReviewsExtractor(min_text_length=10)
        records = extractor.load_all(fname)
        assert len(records) == 1
        assert records[0]["id"] == "r2"

    @pytest.mark.unit
    def test_streams_in_batches(self, sample_reviews_file: str) -> None:
        extractor = AmazonReviewsExtractor(min_text_length=1, batch_size=10)
        batches = list(extractor.stream_from_json(sample_reviews_file))
        assert len(batches) > 1  # Should be multiple batches for 100 records

    @pytest.mark.unit
    def test_file_not_found_raises(self) -> None:
        extractor = AmazonReviewsExtractor()
        with pytest.raises(FileNotFoundError):
            extractor.load_all("/nonexistent/path/reviews.json")


# ─── Feature Engineer Tests ───────────────────────────────────────────────────

class TestNYCTaxiFeatureEngineer:

    @pytest.fixture
    def sample_taxi_df(self) -> pd.DataFrame:
        return pd.DataFrame({
            "tpep_pickup_datetime": ["2024-01-15 08:00:00", "2024-01-15 17:30:00"],
            "tpep_dropoff_datetime": ["2024-01-15 08:30:00", "2024-01-15 18:00:00"],
            "passenger_count": [2, 1],
            "trip_distance": [5.2, 8.1],
            "PULocationID": [132, 100],   # 132 = JFK
            "DOLocationID": [50, 200],
            "fare_amount": [20.0, 30.0],
            "tip_amount": [4.0, 5.0],
            "tolls_amount": [0.0, 1.5],
            "total_amount": [25.0, 37.5],
        })

    @pytest.mark.unit
    def test_adds_trip_duration(self, sample_taxi_df: pd.DataFrame) -> None:
        fe = NYCTaxiFeatureEngineer()
        df = fe.transform(sample_taxi_df)
        assert "trip_duration_minutes" in df.columns
        assert df["trip_duration_minutes"].iloc[0] == pytest.approx(30.0)

    @pytest.mark.unit
    def test_adds_airport_flag(self, sample_taxi_df: pd.DataFrame) -> None:
        fe = NYCTaxiFeatureEngineer()
        df = fe.transform(sample_taxi_df)
        assert "is_airport_trip" in df.columns
        assert df["is_airport_trip"].iloc[0] == 1   # PULocationID=132 (JFK)
        assert df["is_airport_trip"].iloc[1] == 0   # no airport

    @pytest.mark.unit
    def test_adds_rush_hour_flag(self, sample_taxi_df: pd.DataFrame) -> None:
        fe = NYCTaxiFeatureEngineer()
        df = fe.transform(sample_taxi_df)
        assert "is_rush_hour" in df.columns
        assert df["is_rush_hour"].iloc[0] == 1  # 08:00 is rush hour
        assert df["is_rush_hour"].iloc[1] == 1  # 17:30 is rush hour

    @pytest.mark.unit
    def test_tip_rate_calculation(self, sample_taxi_df: pd.DataFrame) -> None:
        fe = NYCTaxiFeatureEngineer()
        df = fe.transform(sample_taxi_df)
        assert "tip_rate" in df.columns
        expected_tip_rate = 4.0 / 20.0
        assert df["tip_rate"].iloc[0] == pytest.approx(expected_tip_rate, abs=0.001)

    @pytest.mark.unit
    def test_is_valid_flag(self, sample_taxi_df: pd.DataFrame) -> None:
        fe = NYCTaxiFeatureEngineer()
        df = fe.transform(sample_taxi_df)
        assert "is_valid" in df.columns
        assert df["is_valid"].all()  # All sample records should be valid


class TestReviewFeatureEngineer:

    @pytest.mark.unit
    def test_adds_text_features(self) -> None:
        df = pd.DataFrame({
            "id": ["r1", "r2"],
            "text": ["This is a great product! Amazing quality!", "Terrible. Broken on arrival."],
            "rating": [5, 1],
        })
        fe = ReviewFeatureEngineer()
        result = fe.transform(df)

        assert "text_length" in result.columns
        assert "word_count" in result.columns
        assert "has_exclamation" in result.columns
        assert "is_positive" in result.columns
        assert "is_negative" in result.columns
        assert "sentiment_proxy" in result.columns

    @pytest.mark.unit
    def test_positive_negative_flags(self) -> None:
        df = pd.DataFrame({
            "text": ["Great product", "Terrible product"],
            "rating": [5, 1],
        })
        fe = ReviewFeatureEngineer()
        result = fe.transform(df)
        assert result["is_positive"].iloc[0] == 1
        assert result["is_negative"].iloc[1] == 1

    @pytest.mark.unit
    def test_sentiment_proxy_range(self) -> None:
        df = pd.DataFrame({
            "text": ["This is amazing excellent perfect", "This is terrible awful horrible"],
            "rating": [5, 1],
        })
        fe = ReviewFeatureEngineer()
        result = fe.transform(df)
        assert result["sentiment_proxy"].iloc[0] > 0   # positive words
        assert result["sentiment_proxy"].iloc[1] < 0   # negative words


# ─── DuckDB Loader Tests ──────────────────────────────────────────────────────

class TestDuckDBLoader:

    @pytest.mark.unit
    def test_load_dataframe(self, duckdb_loader) -> None:
        df = pd.DataFrame({"id": ["a", "b"], "value": [1, 2]})
        rows = duckdb_loader.load_dataframe(df, table="test_table")
        assert rows == 2

    @pytest.mark.unit
    def test_load_then_query(self, duckdb_loader) -> None:
        df = pd.DataFrame({"name": ["Alice", "Bob"], "score": [95, 87]})
        duckdb_loader.load_dataframe(df, table="scores")
        result = duckdb_loader.query("SELECT COUNT(*) as cnt FROM scores")
        assert result["cnt"].iloc[0] == 2

    @pytest.mark.unit
    def test_idempotent_load_with_conflict_key(self, duckdb_loader) -> None:
        df1 = pd.DataFrame({"id": ["r1", "r2"], "val": [10, 20]})
        df2 = pd.DataFrame({"id": ["r1", "r3"], "val": [99, 30]})  # r1 again

        duckdb_loader.load_dataframe(df1, table="idem_test", conflict_key="id")
        duckdb_loader.load_dataframe(df2, table="idem_test", conflict_key="id")

        result = duckdb_loader.query("SELECT COUNT(*) as cnt FROM idem_test")
        # r1 deleted + reinserted, r2 stays, r3 added = 3 total
        assert result["cnt"].iloc[0] == 3
