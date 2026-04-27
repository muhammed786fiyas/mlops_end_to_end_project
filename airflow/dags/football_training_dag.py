"""
Football Match Outcome Prediction — Training Pipeline DAG

Orchestrates the end-to-end retraining workflow:
1. Wait for a trigger file (FileSensor) — simulates new data arrival
2. Validate the project state (data files exist)
3. Run DVC stage: build_features (pool-constrained)
4. Run DVC stage: train (pool-constrained)
5. Check the resulting F1 score meets minimum threshold

A separate dry_pipeline_alert task fires only if the sensor times out
without seeing a trigger file.

Concurrency is controlled by the `training_pool` (1 slot) so that heavy
training jobs cannot run in parallel, preventing OOM on a single machine.

Schedule: Weekly on Sunday at 02:00 UTC, paused on creation — manually 
triggered for demos. To trigger, drop a file at:
/opt/airflow/project/data/triggers/retrain_trigger.txt
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

from airflow.sdk import DAG, task
from airflow.providers.standard.operators.bash import BashOperator
from airflow.providers.standard.sensors.filesystem import FileSensor

PROJECT_ROOT = "/opt/airflow/project"
TRIGGER_DIR = f"{PROJECT_ROOT}/data/triggers"
TRIGGER_FILE = f"{TRIGGER_DIR}/retrain_trigger.txt"
METRICS_FILE = f"{PROJECT_ROOT}/models/metrics.json"

default_args = {
    "owner": "football-mlops",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=1),
    "retry_exponential_backoff": True,
    "max_retry_delay": timedelta(minutes=10),
}


@task
def validate_data(trigger_file: str) -> dict:
    """Sanity-check the project state before training.

    Performs three checks:
    1. Trigger file exists (set by upstream data pipeline)
    2. Trigger file is recent (< 24 hours old) — stale triggers
       suggest the upstream pipeline failed to refresh
    3. Raw database is present and non-empty (>1MB sanity check
       against partial/corrupted downloads)

    Returns metadata for downstream tasks via XCom.
    """
    import os
    import time

    # Check 1: trigger file presence (sensor already did this, but defensive)
    if not os.path.isfile(trigger_file):
        raise FileNotFoundError(f"Trigger file not found: {trigger_file}")

    # Check 2: trigger file freshness
    trigger_age_seconds = time.time() - os.path.getmtime(trigger_file)
    trigger_age_hours = trigger_age_seconds / 3600
    if trigger_age_hours > 24:
        raise ValueError(
            f"Trigger file is {trigger_age_hours:.1f}h old (max 24h). "
            f"Upstream data pipeline may have failed to refresh."
        )
    print(f"Trigger file age: {trigger_age_hours:.2f}h (within 24h window)")

    # Check 3: raw data presence and size
    raw_db = Path(PROJECT_ROOT) / "data" / "raw" / "database.sqlite"
    if not raw_db.exists():
        raise FileNotFoundError(f"Raw database missing: {raw_db}")

    size_bytes = raw_db.stat().st_size
    size_mb = size_bytes / (1024 * 1024)
    if size_bytes < 1_000_000:  # less than 1MB suggests partial download
        raise ValueError(
            f"Raw database suspiciously small: {size_mb:.2f}MB. "
            f"Expected at least 1MB. Possible corrupt download."
        )
    print(f"Raw DB OK: {raw_db} ({size_mb:.1f} MB)")

    return {
        "raw_db_size_mb": round(size_mb, 1),
        "trigger_age_hours": round(trigger_age_hours, 2),
        "validated_at": datetime.utcnow().isoformat(),
    }


@task
def check_metrics(min_f1: float = 0.40) -> dict:
    """Read test F1 from metrics.json and gate deployment."""
    with open(METRICS_FILE) as f:
        metrics = json.load(f)
    test_f1 = metrics["test"]["macro_f1"]
    test_accuracy = metrics["test"]["accuracy"]
    test_auc = metrics["test"]["roc_auc_ovr_macro"]
    print(f"Test Macro F1:  {test_f1:.4f}")
    print(f"Test Accuracy:  {test_accuracy:.4f}")
    print(f"Test ROC-AUC:   {test_auc:.4f}")
    print(f"Minimum F1 threshold: {min_f1}")
    if test_f1 < min_f1:
        raise ValueError(
            f"Model F1 ({test_f1:.4f}) below minimum threshold ({min_f1}). "
            f"Blocking promotion."
        )
    print(f"Model passed quality gate (F1 >= {min_f1})")
    return {"test_f1": test_f1, "test_accuracy": test_accuracy, "test_auc": test_auc}


@task(trigger_rule="one_failed", retries=0)
def dry_pipeline_alert() -> str:
    """Fires when the FileSensor times out without a trigger file."""
    import logging
    logging.warning(
        "DRY PIPELINE ALERT: No trigger file detected within the sensor "
        "timeout window. Skipping this run — no new data to process."
    )
    return "alerted"


with DAG(
    dag_id="football_training_pipeline",
    description="Weekly retraining of the football match outcome predictor",
    default_args=default_args,
    schedule="0 2 * * 0",
    start_date=datetime(2026, 4, 1),
    catchup=False,
    max_active_runs=1,
    tags=["football", "mlops", "training", "dvc", "mlflow"],
) as dag:

    data_sensor = FileSensor(
        task_id="data_sensor",
        filepath=TRIGGER_FILE,
        poke_interval=30,
        timeout=300,
        mode="reschedule",
    )

    validation = validate_data(trigger_file=TRIGGER_FILE)

    build_features = BashOperator(
        task_id="build_features",
        bash_command=f"cd {PROJECT_ROOT} && dvc repro build_features",
        pool="training_pool",
    )

    train_model = BashOperator(
        task_id="train_model",
        bash_command=f"cd {PROJECT_ROOT} && dvc repro train",
        pool="training_pool",
        env={"MLFLOW_TRACKING_URI": "http://mlflow:5000"},
        append_env=True,
    )

    quality_gate = check_metrics(min_f1=0.40)
    dry_alert = dry_pipeline_alert()

    data_sensor >> validation >> build_features >> train_model >> quality_gate
    data_sensor >> dry_alert
