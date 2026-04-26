"""
MLflow Model Registry — Promotion & Rollback
=============================================
Manages model lifecycle in the MLflow Model Registry.

Stages: None → Staging → Production → Archived

Operations:
  promote   — move model from one stage to another
  rollback  — revert Production to the previous version
  list      — show all versions and their stages
  compare   — compare metrics between two versions

Usage:
    python mlops/registry.py promote \
        --model-name bmr-customer-segmentation \
        --from-stage Staging \
        --to-stage Production

    python mlops/registry.py rollback \
        --model-name bmr-customer-segmentation
"""

from __future__ import annotations

import argparse
import os
from typing import Optional

import mlflow
from mlflow.tracking import MlflowClient
from loguru import logger


VALID_STAGES = ["None", "Staging", "Production", "Archived"]
METRIC_GATE = "silhouette_score"
METRIC_IMPROVEMENT_THRESHOLD = 0.02  # new model must beat current by 2%


def get_client() -> MlflowClient:
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    return MlflowClient()


def list_versions(model_name: str) -> None:
    """Print all versions and their current stages."""
    client = get_client()
    versions = client.search_model_versions(f"name='{model_name}'")

    print(f"\nModel: {model_name}")
    print(f"{'Version':<10} {'Stage':<15} {'Status':<12} {'Run ID':<36} {'Created'}")
    print("-" * 90)
    for v in sorted(versions, key=lambda x: int(x.version), reverse=True):
        print(
            f"{v.version:<10} {v.current_stage:<15} {v.status:<12} "
            f"{v.run_id:<36} {v.creation_timestamp}"
        )


def promote(
    model_name: str,
    from_stage: str,
    to_stage: str,
    enforce_metric_gate: bool = True,
) -> str:
    """
    Promote the latest model in `from_stage` to `to_stage`.

    If promoting to Production and enforce_metric_gate=True,
    the new model must have a silhouette_score > current Production model's score.

    Returns the promoted version number.
    """
    client = get_client()

    # Find latest version in from_stage
    candidates = client.get_latest_versions(model_name, stages=[from_stage])
    if not candidates:
        raise ValueError(f"No model in stage '{from_stage}' for model '{model_name}'")
    candidate = candidates[0]

    logger.info(
        f"Promoting: {model_name} v{candidate.version} "
        f"{from_stage} → {to_stage}"
    )

    # Metric gate: only enforce when promoting to Production
    if to_stage == "Production" and enforce_metric_gate:
        _check_metric_gate(client, model_name, candidate)

    # Archive current Production before promoting
    if to_stage == "Production":
        current_prod = client.get_latest_versions(model_name, stages=["Production"])
        for cv in current_prod:
            logger.info(f"Archiving current Production: v{cv.version}")
            client.transition_model_version_stage(
                name=model_name,
                version=cv.version,
                stage="Archived",
                archive_existing_versions=False,
            )

    # Promote
    client.transition_model_version_stage(
        name=model_name,
        version=candidate.version,
        stage=to_stage,
        archive_existing_versions=False,
    )

    logger.info(f"✓ Promoted v{candidate.version} to {to_stage}")
    return candidate.version


def rollback(model_name: str) -> str:
    """
    Rollback Production to the most recent Archived version.

    The current Production model is moved to Archived.
    The most recent Archived version is promoted to Production.

    Returns the version that was restored.
    """
    client = get_client()

    # Archive current Production
    current_prod = client.get_latest_versions(model_name, stages=["Production"])
    if current_prod:
        logger.info(f"Archiving current Production: v{current_prod[0].version}")
        client.transition_model_version_stage(
            name=model_name,
            version=current_prod[0].version,
            stage="Archived",
        )

    # Find the most recent archived version
    archived = client.get_latest_versions(model_name, stages=["Archived"])
    if not archived:
        raise RuntimeError("No archived model version available for rollback")

    restore = max(archived, key=lambda v: int(v.version))
    client.transition_model_version_stage(
        name=model_name,
        version=restore.version,
        stage="Production",
    )

    logger.warning(f"⚠ ROLLBACK: Restored v{restore.version} to Production")
    return restore.version


def _check_metric_gate(
    client: MlflowClient,
    model_name: str,
    candidate,  # ModelVersion
) -> None:
    """
    Check that candidate model's silhouette_score exceeds current Production by 2%.
    Raises ValueError if gate fails.
    """
    current_prod = client.get_latest_versions(model_name, stages=["Production"])
    if not current_prod:
        logger.info("No current Production model — skipping metric gate")
        return

    prod_run = client.get_run(current_prod[0].run_id)
    candidate_run = client.get_run(candidate.run_id)

    prod_score = prod_run.data.metrics.get(METRIC_GATE, 0)
    cand_score = candidate_run.data.metrics.get(METRIC_GATE, 0)

    improvement = cand_score - prod_score
    logger.info(
        f"Metric gate | {METRIC_GATE}: "
        f"candidate={cand_score:.4f} vs production={prod_score:.4f} "
        f"(Δ={improvement:+.4f})"
    )

    if improvement < METRIC_IMPROVEMENT_THRESHOLD:
        raise ValueError(
            f"Metric gate FAILED: candidate {METRIC_GATE}={cand_score:.4f} "
            f"does not beat production {prod_score:.4f} "
            f"by required {METRIC_IMPROVEMENT_THRESHOLD:.2f}"
        )


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MLflow Model Registry Operations")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # list
    p_list = subparsers.add_parser("list", help="List all model versions")
    p_list.add_argument("--model-name", default="bmr-customer-segmentation")

    # promote
    p_promote = subparsers.add_parser("promote", help="Promote model stage")
    p_promote.add_argument("--model-name", default="bmr-customer-segmentation")
    p_promote.add_argument("--from-stage", default="Staging")
    p_promote.add_argument("--to-stage", default="Production")
    p_promote.add_argument("--skip-metric-gate", action="store_true")

    # rollback
    p_rollback = subparsers.add_parser("rollback", help="Rollback Production model")
    p_rollback.add_argument("--model-name", default="bmr-customer-segmentation")

    args = parser.parse_args()

    if args.command == "list":
        list_versions(args.model_name)
    elif args.command == "promote":
        version = promote(
            args.model_name,
            from_stage=args.from_stage,
            to_stage=args.to_stage,
            enforce_metric_gate=not args.skip_metric_gate,
        )
        print(f"\n✓ Promoted to {args.to_stage}: v{version}")
    elif args.command == "rollback":
        version = rollback(args.model_name)
        print(f"\n⚠ Rolled back to: v{version}")
