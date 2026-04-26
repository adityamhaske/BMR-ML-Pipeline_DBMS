"""
Sample Data Downloader — Local Development
==========================================
Downloads small samples of each public dataset for local dev and testing.
Run with: python scripts/download_samples.py
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

SAMPLE_DIR = Path("data/sample")
SAMPLE_DIR.mkdir(parents=True, exist_ok=True)


def download_nyc_taxi_sample() -> None:
    """Download 1 month of NYC Yellow Taxi data (parquet ~50MB)."""
    url = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet"
    out = SAMPLE_DIR / "nyc_taxi_2024_01.parquet"
    if out.exists():
        print(f"  ✓ {out} already exists")
        return
    print(f"  Downloading NYC Taxi 2024-01...")
    urllib.request.urlretrieve(url, str(out))
    print(f"  ✓ Saved: {out}")


def create_amazon_reviews_sample() -> None:
    """Create a synthetic Amazon reviews sample for local testing."""
    out = SAMPLE_DIR / "amazon_reviews_sample.json"
    if out.exists():
        print(f"  ✓ {out} already exists")
        return

    from faker import Faker
    fake = Faker()

    reviews = [
        {
            "id": f"review_{i:06d}",
            "text": fake.paragraph(nb_sentences=3),
            "rating": fake.random_int(min=1, max=5),
            "product_id": f"ASIN_{fake.bothify('??########')}",
            "customer_id": f"CUST_{fake.random_int(min=10000, max=99999)}",
            "timestamp": fake.date_time_this_year().isoformat(),
        }
        for i in range(5000)  # 5K records for local testing
    ]

    with open(out, "w") as f:
        json.dump(reviews, f)

    print(f"  ✓ Created synthetic Amazon reviews sample: {out} ({len(reviews)} records)")


if __name__ == "__main__":
    print("Downloading sample datasets for local development...\n")

    print("1. NYC Taxi (structured ETL data):")
    download_nyc_taxi_sample()

    print("\n2. Amazon Reviews (synthetic NLP data for local dev):")
    create_amazon_reviews_sample()

    print("\n✓ Sample data ready in data/sample/")
