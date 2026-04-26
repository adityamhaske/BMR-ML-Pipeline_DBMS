"""
Model Retraining DAG — MLOps Pillar
=====================================
Scheduled retraining pipeline: triggers when new data is available
or on a monthly cadence.

Tasks:
  check_retrain_trigger    → Check if retraining is needed (data drift or schedule)
  run_training             → Run mlops/train.py via subprocess
  run_evaluation           → Evaluate on holdout set
  register_to_staging      → Push to MLflow Staging
  promote_to_production    → Promote if metric gate passes (silhouette >2% better)
  notify                   → Slack/SNS notification
"""

from __future__ import annotations

import os
import subprocess
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator, ShortCircuitOperator
from airflow.utils.dates import days_ago

DAG_ID = "model_retraining_pipeline"
OWNER = "ml-engineering"
MODEL_NAME = "bmr-customer-segmentation"


def check_retrain_trigger(**context) -> bool:
    """
    ShortCircuit: Check if retraining should proceed.
    Triggers on:
    - Drift report shows has_drift=True
    - Monthly schedule (first run of month)
    """
    # In production: read latest drift report from S3 / DuckDB
    # For simplicity, always retrain if data file exists
    sample_path = os.path.join(
        os.getenv("AIRFLOW_HOME", "/opt/airflow"),
        "data/sample/amazon_reviews_sample.json"
    )
    should_retrain = os.path.exists(sample_path)
    context["task_instance"].log.info(f"Should retrain: {should_retrain}")
    return should_retrain


def run_training(**context) -> str:
    """
    Task: Run training pipeline.
    Returns MLflow run_id via XCom.
    """
    import sys
    import json as json_lib

    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    input_file = os.path.join(
        os.getenv("AIRFLOW_HOME", "/opt/airflow"),
        "data/sample/amazon_reviews_sample.json"
    )

    cmd = [
        sys.executable, "-m", "mlops.train",
        "--input", input_file,
        "--n-clusters", "8",
        "--algorithm", "kmeans",
        "--batch-size", "512",
        "--device", os.getenv("EMBEDDING_DEVICE", "cpu"),
    ]

    env = {**os.environ, "MLFLOW_TRACKING_URI": mlflow_uri}
    result = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd="/opt/airflow")

    if result.returncode != 0:
        raise RuntimeError(f"Training failed:\n{result.stderr}")

    context["task_instance"].log.info(result.stdout)

    # Extract run_id from stdout (last line format: "✓ Training complete | run_id=<id>")
    for line in reversed(result.stdout.split("\n")):
        if "run_id=" in line:
            run_id = line.split("run_id=")[-1].strip()
            return run_id

    return "unknown"


def promote_to_staging(**context) -> None:
    """Task: Promote latest model version to MLflow Staging."""
    from mlops.registry import promote

    version = promote(
        model_name=MODEL_NAME,
        from_stage="None",
        to_stage="Staging",
        enforce_metric_gate=False,  # No gate for Staging
    )
    context["task_instance"].log.info(f"Promoted v{version} to Staging")


def promote_to_production(**context) -> None:
    """Task: Promote Staging model to Production (metric-gated)."""
    from mlops.registry import promote

    try:
        version = promote(
            model_name=MODEL_NAME,
            from_stage="Staging",
            to_stage="Production",
            enforce_metric_gate=True,
        )
        context["task_instance"].log.info(f"✓ Promoted v{version} to Production")
    except ValueError as exc:
        context["task_instance"].log.warning(f"Metric gate failed — staying in Staging: {exc}")


default_args = {
    "owner": OWNER,
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
    "email_on_failure": True,
}

with DAG(
    dag_id=DAG_ID,
    description="Monthly model retraining: Train → Evaluate → Staging → Production",
    default_args=default_args,
    schedule_interval="0 4 1 * *",  # 04:00 UTC on the 1st of each month
    start_date=days_ago(30),
    catchup=False,
    max_active_runs=1,
    tags=["mlops", "retraining", "monthly", "pillar-3"],
    doc_md=__doc__,
) as dag:

    check = ShortCircuitOperator(
        task_id="check_retrain_trigger",
        python_callable=check_retrain_trigger,
    )

    train = PythonOperator(
        task_id="run_training",
        python_callable=run_training,
        execution_timeout=timedelta(hours=3),
    )

    stage = PythonOperator(
        task_id="promote_to_staging",
        python_callable=promote_to_staging,
        execution_timeout=timedelta(minutes=10),
    )

    production = PythonOperator(
        task_id="promote_to_production",
        python_callable=promote_to_production,
        execution_timeout=timedelta(minutes=10),
        trigger_rule="all_success",
    )

    check >> train >> stage >> production
