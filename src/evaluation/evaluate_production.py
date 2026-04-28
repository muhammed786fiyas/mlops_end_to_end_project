"""
Production drift evaluation 

Loads the canonical model (LightGBM v17 from MLflow, or local pickle fallback)
and evaluates it on the 2015/16 season production CSV. The model was never
exposed to this season during training or test (which used 2008-2014 + 2014/15).

This simulates how the model would perform if deployed to production at the
start of 2015/16. If F1 is comparable to test set F1 (~0.45), the model
generalizes well to a fresh season. If F1 drops noticeably, that's measurable
drift.

Logs results to MLflow as a "production_drift_check" run for permanent record.

Usage:
    docker compose exec airflow-scheduler bash -c \
        "cd /opt/airflow/project && python -m src.evaluation.evaluate_production"
"""

import json
import os
import socket
import subprocess
from datetime import datetime
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, roc_auc_score
)

from src.utils.logger import get_logger

logger = get_logger(__name__)


# Must match train.py / feature_service.py FEATURE_COLUMNS
FEATURE_COLUMNS = [
    'home_form_wins', 'home_form_draws', 'home_form_losses',
    'home_form_gs_avg', 'home_form_gc_avg',
    'away_form_wins', 'away_form_draws', 'away_form_losses',
    'away_form_gs_avg', 'away_form_gc_avg',
    'h2h_home_wins', 'h2h_draws', 'h2h_away_wins', 'h2h_n_meetings',
    'home_buildUpPlaySpeed', 'home_buildUpPlayDribbling', 'home_buildUpPlayPassing',
    'home_chanceCreationPassing', 'home_chanceCreationCrossing', 'home_chanceCreationShooting',
    'home_defencePressure', 'home_defenceAggression', 'home_defenceTeamWidth',
    'away_buildUpPlaySpeed', 'away_buildUpPlayDribbling', 'away_buildUpPlayPassing',
    'away_chanceCreationPassing', 'away_chanceCreationCrossing', 'away_chanceCreationShooting',
    'away_defencePressure', 'away_defenceAggression', 'away_defenceTeamWidth',
    'home_elo', 'away_elo', 'elo_diff',
]
TARGET_COLUMN = 'outcome_encoded'
TARGET_NAMES = ['H', 'D', 'A']
MLFLOW_EXPERIMENT_NAME = "football-prediction"
PRODUCTION_CSV_NAME = "season_2015_16.csv"


def _project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in [current, *current.parents]:
        if (parent / "config.yaml").exists():
            return parent
    raise FileNotFoundError("Could not find project root (no config.yaml)")


def _git_hash() -> str:
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def load_canonical_model(local_path: Path):
    """Load the production-canonical model.

    Tries MLflow registry first (matches what backend does); falls back to
    local pickle. Same graceful-degradation pattern as the API.
    """
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)

    model_uri = "models:/football-outcome-predictor/latest"
    try:
        logger.info(f"Loading model from MLflow registry: {model_uri}")
        model = mlflow.lightgbm.load_model(model_uri)
        logger.info("  Loaded from MLflow registry")
        return model, "mlflow:football-outcome-predictor@latest"
    except Exception as e:
        logger.warning(f"MLflow load failed ({type(e).__name__}: {e})")
        logger.info(f"Falling back to local pickle at {local_path}")
        if not local_path.exists():
            raise RuntimeError(
                f"Neither MLflow registry ({model_uri}) nor local pickle "
                f"({local_path}) is reachable. Cannot evaluate."
            )
        model = joblib.load(local_path)
        logger.info("  Loaded from local pickle")
        return model, f"local:{local_path.name}"


def evaluate(model, X, y, split_name: str) -> dict:
    """Same metrics as train.py — apples-to-apples comparison."""
    logger.info(f"Evaluating on {split_name} set ({len(X):,} rows)")
    proba = model.predict_proba(X)
    pred = np.argmax(proba, axis=1)

    metrics = {
        f"{split_name}_macro_f1": f1_score(y, pred, average='macro'),
        f"{split_name}_roc_auc":  roc_auc_score(y, proba, multi_class='ovr', average='macro'),
        f"{split_name}_accuracy": accuracy_score(y, pred),
    }
    per_class_f1 = f1_score(y, pred, average=None, labels=[0, 1, 2])
    for cls_name, score in zip(TARGET_NAMES, per_class_f1):
        metrics[f"{split_name}_f1_{cls_name}"] = float(score)

    logger.info(f"  Macro F1:  {metrics[f'{split_name}_macro_f1']:.4f}")
    logger.info(f"  ROC-AUC:   {metrics[f'{split_name}_roc_auc']:.4f}")
    logger.info(f"  Accuracy:  {metrics[f'{split_name}_accuracy']:.4f}")
    logger.info(f"  Per-class F1: H={metrics[f'{split_name}_f1_H']:.3f}, "
                f"D={metrics[f'{split_name}_f1_D']:.3f}, "
                f"A={metrics[f'{split_name}_f1_A']:.3f}")
    return metrics


def main():
    logger.info("=" * 60)
    logger.info("PRODUCTION DRIFT EVALUATION — START")
    logger.info("=" * 60)

    project_root = _project_root()
    with open(project_root / "config.yaml") as f:
        config = yaml.safe_load(f)

    production_dir = project_root / config['data']['production_dir']
    models_dir = project_root / "models"
    local_model_path = models_dir / "lightgbm_baseline.pkl"

    # MLflow setup
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    git_hash = _git_hash()
    run_name = f"production-drift-check-{git_hash}"
    logger.info(f"Run name: {run_name}")

    with mlflow.start_run(run_name=run_name):
        run_id = mlflow.active_run().info.run_id
        logger.info(f"MLflow run_id: {run_id}")

        # Tags for permanent record
        mlflow.set_tags({
            "git_commit": git_hash,
            "stage": "production_drift_check",
            "purpose": "evaluate trained model on unseen 2015/16 season",
            "registered_to_production": "false",
        })

        # Load model
        model, model_version = load_canonical_model(local_model_path)
        mlflow.log_param("evaluated_model_version", model_version)

        # Load production CSV
        prod_csv_path = production_dir / PRODUCTION_CSV_NAME
        if not prod_csv_path.exists():
            raise FileNotFoundError(
                f"Production CSV not found at {prod_csv_path}. "
                f"Run `dvc repro build_features` to regenerate."
            )

        logger.info(f"Loading production data from {prod_csv_path}")
        prod_df = pd.read_csv(prod_csv_path)
        logger.info(f"  Loaded {len(prod_df):,} matches from production CSV")

        # Verify all 35 features are present
        missing = [c for c in FEATURE_COLUMNS if c not in prod_df.columns]
        if missing:
            raise ValueError(
                f"Production CSV is missing {len(missing)} required features: "
                f"{missing[:5]}{'...' if len(missing) > 5 else ''}. "
                f"Run `dvc repro build_features` to regenerate with current schema."
            )

        X_prod = prod_df[FEATURE_COLUMNS]
        y_prod = prod_df[TARGET_COLUMN].astype(int)

        # Sanity: any nulls would skew the result silently
        if X_prod.isnull().any().any():
            null_cols = X_prod.columns[X_prod.isnull().any()].tolist()
            raise ValueError(f"Production data has nulls in: {null_cols}")

        # Log dataset stats for the record
        mlflow.log_param("production_n_matches", len(prod_df))
        mlflow.log_param("production_season", "2015/2016")

        # Evaluate
        prod_metrics = evaluate(model, X_prod, y_prod, "production")
        for k, v in prod_metrics.items():
            mlflow.log_metric(k, v)

        # Compare against the canonical test set metrics from models/metrics.json
        canonical_metrics_path = models_dir / "metrics.json"
        if canonical_metrics_path.exists():
            with open(canonical_metrics_path) as f:
                canonical_metrics = json.load(f)

            test_f1 = canonical_metrics.get('test', {}).get('macro_f1', None)
            prod_f1 = prod_metrics['production_macro_f1']

            if test_f1 is not None:
                drift = prod_f1 - test_f1
                mlflow.log_metric("drift_macro_f1", drift)
                logger.info("=" * 60)
                logger.info("DRIFT COMPARISON")
                logger.info("=" * 60)
                logger.info(f"  Test F1 (2014/15):       {test_f1:.4f}")
                logger.info(f"  Production F1 (2015/16): {prod_f1:.4f}")
                logger.info(f"  Drift (prod - test):     {drift:+.4f}")

                # Drift severity bands tuned for tabular ML on noisy domain (~F1=0.45 baseline).
                # These thresholds reflect what's typical inter-season variance vs actionable drift.
                if drift >= 0.02:
                    verdict = f"IMPROVED by {drift:+.3f} F1 — model generalizes better than test (unusual)"
                    severity = "improved"
                elif drift >= -0.01:
                    verdict = "STABLE — within typical inter-season variance"
                    severity = "stable"
                elif drift >= -0.03:
                    verdict = f"MILD DRIFT detected ({drift:+.3f} F1) — monitor over next 2 seasons"
                    severity = "mild_drift"
                elif drift >= -0.05:
                    verdict = f"DRIFT detected ({drift:+.3f} F1) — schedule retraining within 30 days"
                    severity = "drift"
                else:
                    verdict = f"SEVERE DRIFT ({drift:+.3f} F1) — retrain immediately"
                    severity = "severe_drift"
                logger.info(f"  Verdict: {verdict}")
                mlflow.set_tag("drift_verdict", verdict)
                mlflow.set_tag("drift_severity", severity)
        else:
            logger.warning(f"Canonical metrics file not found at {canonical_metrics_path}; "
                           f"cannot compute drift delta")

        # Save drift report
        drift_report = {
            "evaluation_timestamp_utc": datetime.utcnow().isoformat(),
            "model_version": model_version,
            "production_season": "2015/2016",
            "n_matches": int(len(prod_df)),
            "metrics": prod_metrics,
            "git_commit": git_hash,
        }
        drift_report_path = models_dir / "production_drift_report.json"
        with open(drift_report_path, "w") as f:
            json.dump(drift_report, f, indent=2)
        mlflow.log_artifact(str(drift_report_path))
        logger.info(f"Saved drift report: {drift_report_path}")

        logger.info("=" * 60)
        logger.info(f"PRODUCTION DRIFT EVALUATION — COMPLETE | "
                    f"Production F1 = {prod_metrics['production_macro_f1']:.4f}")
        logger.info(f"MLflow run ID: {run_id}")
        logger.info("=" * 60)


if __name__ == "__main__":
    main()
