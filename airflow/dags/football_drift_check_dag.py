"""
Football Match Outcome Prediction — Production Drift Check DAG

Runs daily to detect data drift on the deployed model. Decoupled from the
weekly training DAG because drift detection answers a different question:
"is the currently-deployed model still good?" — which you need to know
between retrains, not just after them.

Schedule: Daily at 03:00 UTC (1 hour after the weekly training slot, so
the two DAGs never overlap). In a real production system with streaming
data, drift would be measured even more frequently.

Quality gate: fails the DAG run if drift verdict is `severe_drift`,
making it visible in the Airflow UI and triggering AlertManager via the
backend's downstream alerting (in production this would page on-call;
here it surfaces in the Airflow UI as a failed task).

This project's production CSV is static (the 2015/16 season), so drift
won't change between checks. The DAG nonetheless demonstrates the
production pattern — when real production data arrives, this DAG catches
drift the same day rather than waiting until the next training cycle.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

from airflow.sdk import DAG, task
from airflow.providers.standard.operators.bash import BashOperator

PROJECT_ROOT = "/opt/airflow/project"
DRIFT_REPORT_FILE = f"{PROJECT_ROOT}/models/production_drift_report.json"


default_args = {
    "owner": "football-mlops",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
    "retry_exponential_backoff": False,
}


@task
def parse_drift_verdict() -> dict:
    """Read the drift report produced by the upstream stage and gate on severity.

    Reads `production_drift_report.json` written by `evaluate_production.py`
    and inspects the drift_severity tag indirectly via the verdict text.
    Fails the DAG if drift is severe enough to warrant immediate retraining.

    The script's own MLflow logging is the permanent record; this task
    surfaces severity to Airflow's UI so operators see drift in the same
    place they see training failures.
    """
    if not Path(DRIFT_REPORT_FILE).exists():
        raise FileNotFoundError(
            f"Drift report not found at {DRIFT_REPORT_FILE}. "
            f"The evaluate_production stage may have failed silently."
        )

    with open(DRIFT_REPORT_FILE) as f:
        report = json.load(f)

    metrics = report.get("metrics", {})
    prod_f1 = metrics.get("production_macro_f1")
    timestamp = report.get("evaluation_timestamp_utc", "unknown")
    model_version = report.get("model_version", "unknown")

    # Read canonical test F1 for in-DAG drift comparison
    metrics_file = Path(f"{PROJECT_ROOT}/models/metrics.json")
    if metrics_file.exists():
        with open(metrics_file) as f:
            canonical = json.load(f)
        test_f1 = canonical.get("test", {}).get("macro_f1")
    else:
        test_f1 = None

    if test_f1 is None or prod_f1 is None:
        print(f"Cannot compute drift: test_f1={test_f1}, prod_f1={prod_f1}")
        return {"drift": None, "verdict": "unknown"}

    drift = prod_f1 - test_f1
    print(f"Model version evaluated:  {model_version}")
    print(f"Evaluation timestamp:     {timestamp}")
    print(f"Test F1:                  {test_f1:.4f}")
    print(f"Production F1:            {prod_f1:.4f}")
    print(f"Drift:                    {drift:+.4f}")

    # Same severity bands as evaluate_production.py
    if drift >= 0.02:
        verdict = "improved"
    elif drift >= -0.01:
        verdict = "stable"
    elif drift >= -0.03:
        verdict = "mild_drift"
    elif drift >= -0.05:
        verdict = "drift"
    else:
        verdict = "severe_drift"

    print(f"Verdict: {verdict.upper().replace('_', ' ')}")

    # Quality gate: fail loudly on severe drift
    # mild_drift / drift / improved / stable all pass — only severe_drift fails the DAG
    if verdict == "severe_drift":
        raise ValueError(
            f"SEVERE DRIFT detected (drift={drift:+.4f}). "
            f"Production F1 dropped >5 points below test F1. "
            f"Schedule retraining IMMEDIATELY. "
            f"This DAG run is being failed to surface the issue to operators."
        )

    return {
        "drift": round(drift, 4),
        "verdict": verdict,
        "test_f1": round(test_f1, 4),
        "production_f1": round(prod_f1, 4),
    }


with DAG(
    dag_id="football_drift_check",
    description="Daily drift detection on the deployed model vs unseen 2015/16 data",
    default_args=default_args,
    schedule="0 3 * * *",   # daily at 03:00 UTC (1h after training DAG slot on Sundays)
    start_date=datetime(2026, 4, 1),
    catchup=False,
    max_active_runs=1,
    tags=["football", "mlops", "drift-detection", "monitoring"],
) as dag:

    evaluate_drift = BashOperator(
        task_id="evaluate_production_drift",
        bash_command=f"cd {PROJECT_ROOT} && dvc repro evaluate_production",
        pool="training_pool",   # share with training so they never conflict
        env={"MLFLOW_TRACKING_URI": "http://mlflow:5000"},
        append_env=True,
    )

    verdict_check = parse_drift_verdict()

    evaluate_drift >> verdict_check
