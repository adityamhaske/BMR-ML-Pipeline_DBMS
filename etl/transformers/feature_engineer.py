"""
Feature Engineer — ETL Transformer
=====================================
Transforms raw NYC Taxi and review data into ML-ready feature sets.

Derived features:
  NYC Taxi:
    - trip_duration_minutes
    - speed_mph
    - is_airport_trip
    - hour_of_day, day_of_week, is_weekend
    - fare_per_mile
    - tip_rate (tip_amount / fare_amount)

  Reviews:
    - text_length
    - word_count
    - is_positive (rating >= 4)
    - has_exclamation
    - sentiment_proxy (simple polarity score)
"""

from __future__ import annotations

import re

import pandas as pd
from loguru import logger


# NYC Location IDs that correspond to JFK and LaGuardia airports
AIRPORT_LOCATION_IDS = {132, 138}  # JFK = 132, LaGuardia = 138


class NYCTaxiFeatureEngineer:
    """Transform raw NYC Taxi data into ML-ready features."""

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all feature engineering transformations.

        Args:
            df: Raw NYC Taxi DataFrame with standard column names

        Returns:
            DataFrame with original columns + derived features
        """
        logger.info(f"Transforming {len(df):,} NYC Taxi records")
        df = df.copy()

        # ── Time features ─────────────────────────────────────────────────────
        df["tpep_pickup_datetime"] = pd.to_datetime(df["tpep_pickup_datetime"], errors="coerce")
        df["tpep_dropoff_datetime"] = pd.to_datetime(df["tpep_dropoff_datetime"], errors="coerce")

        df["trip_duration_minutes"] = (
            (df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"])
            .dt.total_seconds()
            .div(60)
            .round(2)
        )
        df["pickup_hour"] = df["tpep_pickup_datetime"].dt.hour
        df["pickup_day_of_week"] = df["tpep_pickup_datetime"].dt.dayofweek  # 0=Mon, 6=Sun
        df["is_weekend"] = df["pickup_day_of_week"].isin([5, 6]).astype(int)
        df["is_rush_hour"] = df["pickup_hour"].isin([7, 8, 9, 16, 17, 18, 19]).astype(int)

        # ── Trip features ─────────────────────────────────────────────────────
        df["is_airport_trip"] = (
            df["PULocationID"].isin(AIRPORT_LOCATION_IDS) |
            df["DOLocationID"].isin(AIRPORT_LOCATION_IDS)
        ).astype(int)

        # Speed (mph) — guard against division by zero and invalid durations
        valid_duration = df["trip_duration_minutes"] > 0
        df["speed_mph"] = 0.0
        df.loc[valid_duration, "speed_mph"] = (
            df.loc[valid_duration, "trip_distance"] /
            (df.loc[valid_duration, "trip_duration_minutes"] / 60)
        ).clip(0, 100).round(2)

        # ── Fare features ─────────────────────────────────────────────────────
        valid_distance = df["trip_distance"] > 0.1
        df["fare_per_mile"] = 0.0
        df.loc[valid_distance, "fare_per_mile"] = (
            df.loc[valid_distance, "fare_amount"] /
            df.loc[valid_distance, "trip_distance"]
        ).clip(0, 50).round(2)

        valid_fare = df["fare_amount"] > 0
        df["tip_rate"] = 0.0
        df.loc[valid_fare, "tip_rate"] = (
            df.loc[valid_fare, "tip_amount"] /
            df.loc[valid_fare, "fare_amount"]
        ).clip(0, 2).round(3)

        # ── Data quality flags ────────────────────────────────────────────────
        df["is_valid"] = (
            (df["trip_duration_minutes"] > 0) &
            (df["trip_duration_minutes"] < 240) &       # < 4 hours
            (df["trip_distance"] > 0) &
            (df["trip_distance"] < 200) &               # < 200 miles
            (df["fare_amount"] > 0) &
            (df["passenger_count"] >= 1) &
            (df["passenger_count"] <= 8)
        ).astype(int)

        valid_count = df["is_valid"].sum()
        logger.info(
            f"Feature engineering complete | "
            f"valid={valid_count:,}/{len(df):,} ({valid_count/len(df):.1%})"
        )
        return df


class ReviewFeatureEngineer:
    """Transform raw review text into ML-ready metadata features."""

    _POSITIVE_WORDS = {"great", "excellent", "amazing", "love", "perfect", "awesome", "fantastic"}
    _NEGATIVE_WORDS = {"terrible", "awful", "horrible", "worst", "broken", "defective", "useless"}

    def transform(self, df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
        """
        Add text-level features to a reviews DataFrame.

        Args:
            df: DataFrame with at least a `text_col` column
            text_col: Name of the text column

        Returns:
            DataFrame with added feature columns
        """
        logger.info(f"Transforming {len(df):,} review records")
        df = df.copy()

        texts = df[text_col].fillna("").astype(str)

        df["text_length"] = texts.str.len()
        df["word_count"] = texts.str.split().str.len()
        df["has_exclamation"] = texts.str.contains("!", regex=False).astype(int)
        df["has_question"] = texts.str.contains("?", regex=False).astype(int)
        df["is_positive"] = (df.get("rating", 3) >= 4).astype(int)
        df["is_negative"] = (df.get("rating", 3) <= 2).astype(int)

        # Simple lexicon-based sentiment proxy (before embedding)
        def sentiment_proxy(text: str) -> float:
            words = set(re.findall(r"\b\w+\b", text.lower()))
            pos = len(words & self._POSITIVE_WORDS)
            neg = len(words & self._NEGATIVE_WORDS)
            total = pos + neg
            return (pos - neg) / total if total > 0 else 0.0

        df["sentiment_proxy"] = texts.map(sentiment_proxy).round(3)

        logger.info(f"Review feature engineering complete")
        return df
