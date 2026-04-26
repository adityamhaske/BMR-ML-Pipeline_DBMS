"""
Amazon Reviews Embedding DAG — Pillar 1 + 2
============================================
Orchestrates the full embedding pipeline for Amazon review data.

Schedule: Weekly (re-embed to catch model updates and new data)

Tasks:
  check_new_data          → Check if new review data is available in S3
  preprocess_reviews      → Clean, deduplicate, language filter
  run_batch_embedding     → Invoke batch embedder (chunked, idempotent)
  build_faiss_index       → Flush vectors, build/update FAISS index
  run_clustering          → HDBSCAN + K-Means on new embeddings
  write_segments_to_dw    → Write segment assignments to DuckDB/Redshift
  compute_drift_report    → Compare centroids vs. last week (alert if >15%)
  notify_completion       → Slack/SNS notification
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator, ShortCircuitOperator
from airflow.utils.dates import days_ago

DAG_ID = "amazon_reviews_embedding"
OWNER = "ml-engineering"


def check_new_data(**context) -> bool:
    """
    ShortCircuit: return True if new review data is available.
    Returns False to skip the rest of the pipeline if nothing to process.
    """
    # In production: check S3 for new files since last successful run
    # For dev: always return True if sample file exists
    sample_path = os.path.join(
        os.getenv("AIRFLOW_HOME", "/opt/airflow"),
        "data/sample/amazon_reviews_sample.json"
    )
    has_data = os.path.exists(sample_path)
    context["task_instance"].log.info(
        f"New data check: {'found' if has_data else 'none'}"
    )
    return has_data


def preprocess_reviews(**context) -> str:
    """
    Task: Load reviews, apply text cleaning and deduplication.
    Returns the path to the cleaned records (via XCom).
    """
    import json
    from pathlib import Path

    from etl.extractors.amazon_reviews_extractor import AmazonReviewsExtractor

    input_path = os.path.join(
        os.getenv("AIRFLOW_HOME", "/opt/airflow"),
        "data/sample/amazon_reviews_sample.json"
    )
    extractor = AmazonReviewsExtractor(min_text_length=20, batch_size=5000)
    records = extractor.load_all(input_path)

    output_path = "/tmp/bmr-etl/preprocessed_reviews.json"
    Path("/tmp/bmr-etl").mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(records, f)

    context["task_instance"].log.info(
        f"Preprocessed {len(records):,} reviews → {output_path}"
    )
    return output_path


def run_batch_embedding(**context) -> str:
    """
    Task: Run the batch embedding pipeline.
    Chunked, idempotent. Returns embedding output directory via XCom.
    """
    input_path = context["task_instance"].xcom_pull(task_ids="preprocess_reviews")

    from embedding.batch_embedder import BatchEmbedder
    from embedding.config import EmbeddingConfig
    from embedding.vector_store import FAISSVectorStoreWriter
    import json

    output_dir = "/tmp/bmr-etl/embeddings/"
    config = EmbeddingConfig(
        model_name=os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"),
        batch_size=int(os.getenv("EMBEDDING_BATCH_SIZE", "512")),
        device=os.getenv("EMBEDDING_DEVICE", "cpu"),
        output_dir=output_dir,
        checkpoint_dir="/tmp/bmr-etl/checkpoints/",
    )

    with open(input_path) as f:
        records = json.load(f)

    embedder = BatchEmbedder(config)
    writer = FAISSVectorStoreWriter(
        output_dir=output_dir,
        embedding_dim=embedder.embedding_dim,
    )
    stats = embedder.run(records=records, writer=writer, text_field="text", id_field="id")
    writer.flush()

    context["task_instance"].log.info(
        f"Embedding complete | processed={stats.processed_records:,} | "
        f"throughput={stats.throughput_rps:.0f} rec/s"
    )
    return output_dir


def run_clustering(**context) -> str:
    """Task: Run HDBSCAN + K-Means clustering on embeddings."""
    embedding_dir = context["task_instance"].xcom_pull(task_ids="run_batch_embedding")
    import json

    from embedding.vector_store import FAISSVectorStoreReader
    from segmentation.clustering import SegmentationEngine
    import numpy as np

    reader = FAISSVectorStoreReader(embedding_dir)
    id_map = json.loads(open(f"{embedding_dir}/id_map.json").read())
    n = reader.total_vectors

    # Reconstruct all vectors from FAISS (small dataset in dev)
    import faiss
    index = faiss.read_index(f"{embedding_dir}/index.faiss")
    embeddings = index.reconstruct_n(0, n)

    engine = SegmentationEngine(n_clusters=8, min_cluster_size=10)
    result = engine.run_kmeans(embeddings, record_ids=id_map)

    output_dir = "/tmp/bmr-etl/segments/"
    engine.save(result, output_dir)

    context["task_instance"].log.info(
        f"Clustering complete | segments={result.n_segments} | "
        f"silhouette={result.silhouette:.3f}"
    )
    return output_dir


def write_segments_to_dw(**context) -> int:
    """Task: Write segment labels to data warehouse."""
    segment_dir = context["task_instance"].xcom_pull(task_ids="run_clustering")
    import json

    from etl.loaders.redshift_loader import get_loader
    import pandas as pd

    with open(f"{segment_dir}/segment_labels.json") as f:
        labels = json.load(f)

    df = pd.DataFrame([
        {"record_id": rid, "segment_id": seg, "run_timestamp": datetime.utcnow().isoformat()}
        for rid, seg in labels.items()
    ])

    with get_loader() as loader:
        loader.execute("""
            CREATE TABLE IF NOT EXISTS segment_assignments (
                record_id TEXT, segment_id INTEGER, run_timestamp TEXT
            )
        """)
        rows = loader.load_dataframe(df, table="segment_assignments", conflict_key="record_id")

    context["task_instance"].log.info(f"Wrote {rows:,} segment assignments to DW")
    return rows


default_args = {
    "owner": OWNER,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": True,
}

with DAG(
    dag_id=DAG_ID,
    description="Weekly: Amazon Reviews → Embeddings → Clustering → Segment Labels",
    default_args=default_args,
    schedule_interval="0 3 * * 1",  # 03:00 UTC every Monday
    start_date=days_ago(7),
    catchup=False,
    max_active_runs=1,
    tags=["embedding", "segmentation", "weekly", "pillar-1"],
    doc_md=__doc__,
) as dag:

    check_data = ShortCircuitOperator(
        task_id="check_new_data",
        python_callable=check_new_data,
    )

    preprocess = PythonOperator(
        task_id="preprocess_reviews",
        python_callable=preprocess_reviews,
        execution_timeout=timedelta(minutes=30),
    )

    embed = PythonOperator(
        task_id="run_batch_embedding",
        python_callable=run_batch_embedding,
        execution_timeout=timedelta(hours=4),
    )

    cluster = PythonOperator(
        task_id="run_clustering",
        python_callable=run_clustering,
        execution_timeout=timedelta(hours=1),
    )

    write_dw = PythonOperator(
        task_id="write_segments_to_dw",
        python_callable=write_segments_to_dw,
        execution_timeout=timedelta(minutes=30),
    )

    check_data >> preprocess >> embed >> cluster >> write_dw
