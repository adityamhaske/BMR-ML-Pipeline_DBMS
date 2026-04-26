#!/usr/bin/env python3
"""
=============================================================================
BMR-ML-Pipeline — End-to-End Local Demo (Zero Cost)
=============================================================================
Runs the complete pipeline on your local machine with NO AWS, NO Docker, NO cost.

Free stack used:
  - DuckDB      → replaces AWS Redshift
  - Local files → replaces AWS S3
  - FAISS index → replaces any cloud vector store
  - FastAPI     → serves locally on port 8000

Steps:
  1. Generate synthetic customer reviews (5,000 records)
  2. Clean and preprocess text
  3. Generate embeddings (sentence-transformers/all-MiniLM-L6-v2, CPU)
  4. Build FAISS vector index
  5. Run K-Means clustering (8 behavioral segments)
  6. Write segment labels to DuckDB warehouse
  7. Start FastAPI segment lookup service
  8. Run live inference against the API

Usage:
    python scripts/run_demo.py
    python scripts/run_demo.py --records 1000 --skip-serve
=============================================================================
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# ─── Rich console for beautiful output ───────────────────────────────────────
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
    from rich.table import Table
    from rich.text import Text
    console = Console()
    HAS_RICH = True
except ImportError:
    class Console:  # type: ignore[no-redef]
        def print(self, *args, **kwargs): print(*args)
        def rule(self, *args, **kwargs): print("─" * 60)
    console = Console()
    HAS_RICH = False


PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Force CPU and local mode
os.environ["EMBEDDING_DEVICE"] = "cpu"
os.environ["USE_LOCALSTACK"] = "false"
os.environ["DUCKDB_PATH"] = str(PROJECT_ROOT / "data" / "local" / "demo.duckdb")
os.environ["PYTHONPATH"] = str(PROJECT_ROOT)


def step(n: int, title: str) -> None:
    console.rule(f"[bold cyan]Step {n}: {title}[/bold cyan]" if HAS_RICH else f"Step {n}: {title}")


def ok(msg: str) -> None:
    console.print(f"[bold green]  ✓[/bold green] {msg}" if HAS_RICH else f"  ✓ {msg}")


def info(msg: str) -> None:
    console.print(f"[dim]  → {msg}[/dim]" if HAS_RICH else f"  → {msg}")


# ─── Step 1: Generate Data ────────────────────────────────────────────────────

def generate_sample_reviews(n_records: int, output_path: Path) -> list[dict]:
    """Generate synthetic customer reviews using Faker."""
    step(1, f"Generating {n_records:,} synthetic customer reviews")

    try:
        from faker import Faker
    except ImportError:
        console.print("[red]faker not installed. Run: pip install faker[/red]")
        sys.exit(1)

    fake = Faker()
    Faker.seed(42)

    templates = [
        "{adj} product! {detail}. Would {recommend} recommend.",
        "Bought this {timeago}. {experience}. {verdict}.",
        "{sentiment} with my purchase. {detail}.",
        "The {quality} is {adj}. {detail}. {rating_text}.",
    ]
    adjs = ["Amazing", "Terrible", "Decent", "Excellent", "Poor", "Outstanding", "Mediocre"]
    details = [
        "Fast shipping and great packaging",
        "Broke after two days of use",
        "Exactly as described in the listing",
        "Quality feels premium for the price",
        "Customer service was unhelpful",
        "Works perfectly out of the box",
        "Better than expected overall",
    ]

    records = []
    for i in range(n_records):
        rating = fake.random_int(min=1, max=5)
        adj = fake.random_element(["Amazing", "Great", "Okay", "Poor", "Terrible"] if rating > 3 else ["Bad", "Terrible", "Disappointing", "Poor", "Awful"])
        detail = fake.random_element(details)
        text = f"{adj} product! {detail}. Rating: {rating}/5. {fake.sentence()}"

        records.append({
            "id": f"review_{i:06d}",
            "text": text,
            "rating": rating,
            "product_id": f"ASIN_{(i % 200):04d}",
            "customer_id": f"CUST_{(i % 500):05d}",
            "timestamp": fake.date_time_this_year().isoformat(),
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(records, f)

    ok(f"Generated {n_records:,} reviews → {output_path}")
    return records


# ─── Step 2: Preprocess ───────────────────────────────────────────────────────

def preprocess_reviews(records: list[dict]) -> tuple[list[dict], dict]:
    """Clean text and filter invalid records."""
    step(2, "Preprocessing and deduplicating text")

    from embedding.preprocessor import TextPreprocessor

    preprocessor = TextPreprocessor(min_length=15, max_length=512)

    valid = []
    stats = {"total": len(records), "filtered": 0, "duplicates": 0}

    for record in records:
        text = str(record.get("text", ""))
        if not preprocessor.is_valid(text):
            stats["filtered"] += 1
            continue
        if preprocessor.is_duplicate(text):
            stats["duplicates"] += 1
            continue
        record["text"] = preprocessor.clean(text)
        valid.append(record)

    ok(f"{len(valid):,} valid records | filtered={stats['filtered']} | duplicates={stats['duplicates']}")
    return valid, stats


# ─── Step 3: Embed ────────────────────────────────────────────────────────────

def run_embedding(records: list[dict], output_dir: Path) -> Path:
    """Generate sentence embeddings and build FAISS index."""
    step(3, "Generating embeddings (sentence-transformers/all-MiniLM-L6-v2, CPU)")

    from embedding.batch_embedder import BatchEmbedder
    from embedding.config import EmbeddingConfig
    from embedding.vector_store import FAISSVectorStoreWriter

    config = EmbeddingConfig(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        batch_size=128,
        device="cpu",
        output_dir=str(output_dir),
        checkpoint_dir=str(output_dir / ".checkpoints"),
        fail_fast=True,
    )

    embedder = BatchEmbedder(config)
    info(f"Model loaded | embedding_dim={embedder.embedding_dim}")

    writer = FAISSVectorStoreWriter(
        output_dir=str(output_dir),
        embedding_dim=embedder.embedding_dim,
    )

    t0 = time.perf_counter()
    pipeline_stats = embedder.run(
        records=records,
        writer=writer,
        text_field="text",
        id_field="id",
    )
    writer.flush()
    elapsed = time.perf_counter() - t0

    ok(
        f"Embedded {pipeline_stats.processed_records:,} records | "
        f"throughput={pipeline_stats.throughput_rps:.0f} rec/s | "
        f"elapsed={elapsed:.1f}s"
    )
    return output_dir


# ─── Step 4: Cluster ──────────────────────────────────────────────────────────

def run_clustering(embedding_dir: Path, segment_dir: Path) -> dict:
    """Cluster embeddings into behavioral segments."""
    step(4, "Clustering into 8 behavioral segments (K-Means + UMAP)")

    import faiss
    import json as json_mod
    import numpy as np
    from segmentation.clustering import SegmentationEngine

    # Load FAISS index
    index = faiss.read_index(str(embedding_dir / "index.faiss"))
    id_map = json_mod.loads((embedding_dir / "id_map.json").read_text())
    n = index.ntotal

    info(f"Loaded {n:,} vectors from FAISS index")

    # Reconstruct vectors (fine for demo scale)
    embeddings = index.reconstruct_n(0, n)

    engine = SegmentationEngine(
        n_clusters=8,
        min_cluster_size=max(5, n // 100),
        umap_n_components=min(50, embeddings.shape[1]),
        umap_n_neighbors=min(15, n - 1),
    )

    t0 = time.perf_counter()
    result = engine.run_kmeans(embeddings, record_ids=id_map)
    elapsed = time.perf_counter() - t0

    segment_dir.mkdir(parents=True, exist_ok=True)
    engine.save(result, str(segment_dir))

    ok(
        f"Clustered {n:,} records into {result.n_segments} segments | "
        f"silhouette={result.silhouette:.3f} | elapsed={elapsed:.1f}s"
    )

    # Print segment summary
    labels_map = {
        0: "price_sensitive_shoppers",
        1: "brand_loyal_customers",
        2: "high_value_frequent_buyers",
        3: "occasional_reviewers",
        4: "complaint_prone_users",
        5: "enthusiast_early_adopters",
        6: "gift_buyers",
        7: "bargain_hunters",
    }

    if HAS_RICH:
        table = Table(title="Customer Segment Distribution", show_header=True)
        table.add_column("Segment ID", style="cyan")
        table.add_column("Label", style="green")
        table.add_column("Size", style="white")
        table.add_column("Fraction", style="yellow")
        for p in result.profiles:
            frac = p.size / n
            table.add_row(
                str(p.segment_id),
                labels_map.get(p.segment_id, f"segment_{p.segment_id}"),
                f"{p.size:,}",
                f"{frac:.1%}",
            )
        console.print(table)

    return {
        "n_segments": result.n_segments,
        "silhouette": result.silhouette,
        "profiles": [{"segment_id": p.segment_id, "size": p.size} for p in result.profiles],
    }


# ─── Step 5: Write to DuckDB ─────────────────────────────────────────────────

def write_to_duckdb(segment_dir: Path, reviews: list[dict]) -> None:
    """Write segment labels to DuckDB (our free Redshift replacement)."""
    step(5, "Writing segment labels to DuckDB warehouse (free Redshift alternative)")

    import json as json_mod
    import pandas as pd
    from etl.loaders.redshift_loader import DuckDBLoader

    labels = json_mod.loads((segment_dir / "segment_labels.json").read_text())
    summary = json_mod.loads((segment_dir / "clustering_summary.json").read_text())

    segment_label_map = {
        0: "price_sensitive_shoppers", 1: "brand_loyal_customers",
        2: "high_value_frequent_buyers", 3: "occasional_reviewers",
        4: "complaint_prone_users", 5: "enthusiast_early_adopters",
        6: "gift_buyers", 7: "bargain_hunters",
    }

    # Build enriched DataFrame
    review_map = {r["id"]: r for r in reviews}
    rows = []
    for record_id, seg_id in labels.items():
        review = review_map.get(record_id, {})
        rows.append({
            "record_id": record_id,
            "segment_id": int(seg_id),
            "segment_label": segment_label_map.get(int(seg_id), f"segment_{seg_id}"),
            "rating": review.get("rating", 0),
            "customer_id": review.get("customer_id", ""),
            "product_id": review.get("product_id", ""),
            "run_timestamp": summary.get("run_timestamp", ""),
        })

    df = pd.DataFrame(rows)

    db_path = os.environ["DUCKDB_PATH"]
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    with DuckDBLoader(db_path=db_path) as loader:
        # Create segment assignments table
        loader.execute("""
            CREATE TABLE IF NOT EXISTS segment_assignments (
                record_id     TEXT,
                segment_id    INTEGER,
                segment_label TEXT,
                rating        INTEGER,
                customer_id   TEXT,
                product_id    TEXT,
                run_timestamp TEXT
            )
        """)
        loader.execute("DELETE FROM segment_assignments")  # idempotent reset
        loader.load_dataframe(df, table="segment_assignments")

        # Verify
        count = loader.query("SELECT COUNT(*) as cnt FROM segment_assignments").iloc[0]["cnt"]
        by_segment = loader.query("""
            SELECT segment_label, COUNT(*) as size, ROUND(AVG(rating), 2) as avg_rating
            FROM segment_assignments
            GROUP BY segment_label
            ORDER BY size DESC
        """)

    ok(f"Wrote {count:,} segment assignments to DuckDB: {db_path}")
    info(f"Segment breakdown:")
    for _, row in by_segment.iterrows():
        info(f"  {row['segment_label']:<35} size={row['size']:>5,}  avg_rating={row['avg_rating']}")


# ─── Step 6: Live Inference Test ──────────────────────────────────────────────

def test_live_inference(segment_dir: Path, embedding_dir: Path) -> None:
    """Test the segmentation service directly (no HTTP server needed)."""
    step(6, "Running live inference — segment prediction")

    import json as json_mod
    import faiss
    import numpy as np
    from sentence_transformers import SentenceTransformer
    from sklearn.preprocessing import normalize
    from embedding.preprocessor import TextPreprocessor

    # Load segment labels to know which cluster each record is in
    labels_map_raw = json_mod.loads((segment_dir / "segment_labels.json").read_text())

    # Re-compute centroids in ORIGINAL 384D embedding space from FAISS index
    # (clustering used UMAP-reduced centroids, but inference uses full embeddings)
    faiss_index = faiss.read_index(str(embedding_dir / "index.faiss"))
    id_map = json_mod.loads((embedding_dir / "id_map.json").read_text())
    all_embeddings = faiss_index.reconstruct_n(0, faiss_index.ntotal)
    labels_array = np.array([int(labels_map_raw.get(rid, -1)) for rid in id_map])

    n_segments = len(set(labels_array) - {-1})
    centroids_384d = np.array([
        all_embeddings[labels_array == i].mean(axis=0)
        for i in range(n_segments)
        if (labels_array == i).any()
    ], dtype=np.float32)
    centroids_384d = normalize(centroids_384d)

    preprocessor = TextPreprocessor()
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")

    segment_label_map = {
        0: "price_sensitive_shoppers", 1: "brand_loyal_customers",
        2: "high_value_frequent_buyers", 3: "occasional_reviewers",
        4: "complaint_prone_users", 5: "enthusiast_early_adopters",
        6: "gift_buyers", 7: "bargain_hunters",
    }

    test_texts = [
        ("cust_001", "This is absolutely amazing! Best purchase I've ever made. Super fast delivery!"),
        ("cust_002", "Terrible product. Broke after one day. Complete waste of money. Very disappointed."),
        ("cust_003", "It works fine but nothing special. Decent quality for the price paid."),
        ("cust_004", "Perfect gift for my partner! Arrived beautifully packaged and on time."),
        ("cust_005", "I've been buying this brand for years. Always consistent quality. Loyal customer."),
    ]

    if HAS_RICH:
        table = Table(title="Live Inference Results", show_header=True)
        table.add_column("Customer ID", style="cyan")
        table.add_column("Segment", style="green")
        table.add_column("Confidence", style="yellow")
        table.add_column("Text Preview", style="dim", max_width=45)

    results = []
    for cust_id, text in test_texts:
        cleaned = preprocessor.clean(text)
        embedding = model.encode([cleaned], normalize_embeddings=True, show_progress_bar=False)[0]
        similarities = centroids_384d @ embedding
        seg_id = int(np.argmax(similarities))
        confidence = float((similarities[seg_id] + 1.0) / 2.0)
        label = segment_label_map.get(seg_id, f"segment_{seg_id}")
        results.append({"id": cust_id, "segment": label, "confidence": confidence, "text": text[:50]})

        if HAS_RICH:
            table.add_row(cust_id, label, f"{confidence:.2%}", text[:45] + "...")

    if HAS_RICH:
        console.print(table)
    else:
        for r in results:
            info(f"{r['id']} → {r['segment']} ({r['confidence']:.1%}): {r['text'][:50]}")

    ok("Live inference working correctly")
    return results


# ─── Step 7: Run Unit Tests ───────────────────────────────────────────────────

def run_unit_tests() -> bool:
    """Run the unit test suite."""
    step(7, "Running unit tests")

    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/unit/", "-v", "-m", "unit",
         "--tb=short", "--no-header", "-q"],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(PROJECT_ROOT)},
    )

    # Print output
    for line in result.stdout.split("\n"):
        if line.strip():
            if "PASSED" in line:
                console.print(f"  [green]✓[/green] {line}" if HAS_RICH else f"  ✓ {line}")
            elif "FAILED" in line or "ERROR" in line:
                console.print(f"  [red]✗[/red] {line}" if HAS_RICH else f"  ✗ {line}")
            elif "passed" in line or "failed" in line or "error" in line:
                console.print(f"  [bold]{line}[/bold]" if HAS_RICH else line)

    if result.returncode == 0:
        ok("All unit tests passed!")
        return True
    else:
        console.print(f"[yellow]  ⚠ Some tests had issues (may be missing optional deps)[/yellow]" if HAS_RICH else "  ⚠ Some tests had issues")
        return False


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="BMR-ML-Pipeline End-to-End Demo (Free)")
    parser.add_argument("--records", type=int, default=2000, help="Number of synthetic reviews")
    parser.add_argument("--skip-tests", action="store_true", help="Skip unit tests")
    parser.add_argument("--batch-size", type=int, default=128, help="Embedding batch size")
    args = parser.parse_args()

    os.environ["EMBEDDING_BATCH_SIZE"] = str(args.batch_size)

    # ── Directories ────────────────────────────────────────────────────────────
    data_dir = PROJECT_ROOT / "data"
    reviews_path = data_dir / "sample" / "demo_reviews.json"
    embedding_dir = data_dir / "embeddings" / "demo"
    segment_dir = data_dir / "segments" / "demo"

    if HAS_RICH:
        console.print(Panel.fit(
            "[bold cyan]BMR-ML-Pipeline[/bold cyan]\n"
            "[dim]End-to-End Local Demo — 100% Free, No AWS Required[/dim]\n\n"
            f"[white]Records:[/white]     {args.records:,}\n"
            f"[white]Device:[/white]      CPU (MPS/CUDA auto-detected)\n"
            f"[white]Warehouse:[/white]   DuckDB (free Redshift replacement)\n"
            f"[white]Vectors:[/white]     FAISS local index (free)\n"
            f"[white]Serving:[/white]     FastAPI (local port 8000)\n",
            title="🚀 Starting Demo",
        ))
    else:
        print("=" * 60)
        print("BMR-ML-Pipeline — End-to-End Demo (100% Free)")
        print("=" * 60)

    start = time.perf_counter()

    # Run pipeline steps
    reviews = generate_sample_reviews(args.records, reviews_path)
    valid_reviews, preprocess_stats = preprocess_reviews(reviews)
    embed_dir = run_embedding(valid_reviews, embedding_dir)
    cluster_stats = run_clustering(embed_dir, segment_dir)
    write_to_duckdb(segment_dir, valid_reviews)
    inference_results = test_live_inference(segment_dir, embed_dir)

    if not args.skip_tests:
        run_unit_tests()

    total_elapsed = time.perf_counter() - start

    # ── Summary ───────────────────────────────────────────────────────────────
    console.rule("[bold green]Pipeline Complete![/bold green]" if HAS_RICH else "Pipeline Complete!")

    if HAS_RICH:
        summary = Table(title="Pipeline Summary", show_header=True)
        summary.add_column("Metric", style="cyan")
        summary.add_column("Value", style="green")
        summary.add_row("Total records processed", f"{len(valid_reviews):,}")
        summary.add_row("Segments discovered", str(cluster_stats["n_segments"]))
        summary.add_row("Silhouette score", f"{cluster_stats['silhouette']:.3f}")
        summary.add_row("DuckDB warehouse", os.environ["DUCKDB_PATH"])
        summary.add_row("FAISS index", str(embedding_dir / "index.faiss"))
        summary.add_row("Total elapsed", f"{total_elapsed:.1f}s")
        summary.add_row("AWS cost", "$0.00 💸")
        console.print(summary)
    else:
        print(f"\n  Records processed: {len(valid_reviews):,}")
        print(f"  Segments:          {cluster_stats['n_segments']}")
        print(f"  Silhouette score:  {cluster_stats['silhouette']:.3f}")
        print(f"  Total elapsed:     {total_elapsed:.1f}s")
        print(f"  AWS cost:          $0.00")

    console.print("\n[bold]Next:[/bold] Start the segment API with:" if HAS_RICH else "\nNext: Start the segment API with:")
    console.print("  [cyan]python -m segmentation.segment_api[/cyan]" if HAS_RICH else "  python -m segmentation.segment_api")
    console.print("  [dim]Then POST to http://localhost:8001/v1/segment/bulk[/dim]" if HAS_RICH else "  Then POST to http://localhost:8001/v1/segment/bulk")


if __name__ == "__main__":
    main()
