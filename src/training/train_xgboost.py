"""
Day 7 experiment: XGBoost training for comparison vs LightGBM baseline.

Trains an XGBoost classifier on the same 35-feature set and chronological
split as the LightGBM baseline. Logs to the same MLflow experiment under a
distinct run name so the comparison lives in one place.

This script does NOT register the model to the registry — LightGBM v16
remains the production-served model. This is purely an experimentation run
to validate the algorithm choice empirically.

Usage:
    docker compose exec airflow-scheduler bash -c \
        "cd /opt/airflow/project && python -m src.training.train_xgboost"
"""

import json
import os
import socket
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
import xgboost as xgb
import yaml
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, roc_auc_score
)

from src.utils.logger import get_logger

logger = get_logger(__name__)


# Constants — must match train.py to keep the comparison apples-to-apples
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


def load_train_test(processed_dir: Path):
    train_df = pd.read_csv(processed_dir / "train.csv")
    test_df = pd.read_csv(processed_dir / "test.csv")

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET_COLUMN].astype(int)
    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[TARGET_COLUMN].astype(int)

    logger.info(f"  Train: {len(X_train):,} rows, {len(FEATURE_COLUMNS)} features")
    logger.info(f"  Test:  {len(X_test):,} rows, {len(FEATURE_COLUMNS)} features")
    return X_train, y_train, X_test, y_test


def chronological_train_val_split(X, y, val_fraction: float):
    n = len(X)
    cutoff = int(n * (1 - val_fraction))
    X_fit, X_val = X.iloc[:cutoff], X.iloc[cutoff:]
    y_fit, y_val = y.iloc[:cutoff], y.iloc[cutoff:]
    logger.info(f"  Carved validation set: {len(X_val):,} rows ({int(val_fraction*100)}% of train)")
    return X_fit, y_fit, X_val, y_val


def evaluate(model, X, y, split_name: str) -> dict:
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
    logger.info("XGBOOST TRAINING (Day 7 experiment) — START")
    logger.info("=" * 60)

    project_root = _project_root()
    with open(project_root / "config.yaml") as f:
        config = yaml.safe_load(f)
    with open(project_root / "params.yaml") as f:
        params = yaml.safe_load(f)

    processed_dir = project_root / config['data']['processed_dir']
    models_dir = project_root / "models"
    models_dir.mkdir(exist_ok=True)

    # MLflow setup
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    logger.info(f"MLflow tracking URI: {tracking_uri}")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    git_hash = _git_hash()
    run_name = f"xgboost-md{params['algorithms']['xgboost']['max_depth']}-{git_hash}"
    logger.info(f"Run name: {run_name}")

    with mlflow.start_run(run_name=run_name):
        run_id = mlflow.active_run().info.run_id
        logger.info(f"MLflow run_id: {run_id}")

        # Tag with provenance
        mlflow.set_tags({
            "git_commit": git_hash,
            "stage": "experiment",
            "framework": "xgboost",
            "dataset": "kaggle-european-soccer-2008-2016",
            "experiment_purpose": "algorithm comparison vs LightGBM baseline",
            "registered_to_production": "false",
        })

        # Load data
        X_train, y_train, X_test, y_test = load_train_test(processed_dir)

        # Apply XGBoost params
        xgb_params = dict(params['algorithms']['xgboost'])
        # XGBoost needs num_class for multiclass
        xgb_params['num_class'] = 3
        # Remove keys that aren't xgboost constructor args
        use_label_encoder = xgb_params.pop('use_label_encoder', False)

        # Carve val split
        X_fit, y_fit, X_val, y_val = chronological_train_val_split(
            X_train, y_train,
            val_fraction=params['training']['val_fraction'],
        )

        # Train
        logger.info("Training XGBoost classifier")
        logger.info(f"  max_depth={xgb_params['max_depth']}, "
                    f"learning_rate={xgb_params['learning_rate']}, "
                    f"n_estimators={xgb_params['n_estimators']}")

        # Log params to MLflow
        mlflow.log_params(xgb_params)
        mlflow.log_param("val_fraction", params['training']['val_fraction'])
        mlflow.log_param("early_stopping_rounds", params['training']['early_stopping_rounds'])
        mlflow.log_param("n_features", len(FEATURE_COLUMNS))

        start = datetime.now()
        model = xgb.XGBClassifier(
            **xgb_params,
            early_stopping_rounds=params['training']['early_stopping_rounds'],
        )
        model.fit(
            X_fit, y_fit,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        train_time = (datetime.now() - start).total_seconds()
        logger.info(f"  Training complete in {train_time:.1f}s")
        logger.info(f"  Best iteration: {model.best_iteration} "
                    f"(of {xgb_params['n_estimators']} max)")

        mlflow.log_metric("training_time_seconds", train_time)
        mlflow.log_metric("best_iteration", model.best_iteration)

        # Evaluate on all splits
        all_metrics = {}
        all_metrics.update(evaluate(model, X_fit, y_fit, "train"))
        all_metrics.update(evaluate(model, X_val, y_val, "val"))
        all_metrics.update(evaluate(model, X_test, y_test, "test"))

        for k, v in all_metrics.items():
            mlflow.log_metric(k, v)

        # Save model + metrics locally for reference (NOT registering to production)
        model_path = models_dir / "xgboost_experiment.pkl"
        joblib.dump(model, model_path)
        logger.info(f"Saved model: {model_path}")

        metrics_path = models_dir / "xgboost_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(all_metrics, f, indent=2)
        logger.info(f"Saved metrics: {metrics_path}")

        # Feature importance plot
        importance = model.feature_importances_
        order = np.argsort(importance)[::-1][:20]
        plt.figure(figsize=(8, 6))
        plt.barh(range(len(order)), importance[order][::-1])
        plt.yticks(range(len(order)), [FEATURE_COLUMNS[i] for i in order][::-1])
        plt.xlabel("Importance")
        plt.title("XGBoost — Top 20 Features (Day 7 experiment)")
        plt.tight_layout()
        importance_path = models_dir / "xgboost_feature_importance.png"
        plt.savefig(importance_path, dpi=120)
        plt.close()
        logger.info(f"Saved feature importance plot: {importance_path}")

        # Log artifacts to MLflow
        mlflow.log_artifact(str(metrics_path))
        mlflow.log_artifact(str(importance_path))

        # Log model artifact (but DO NOT register)
        mlflow.xgboost.log_model(
            xgb_model=model,
            name="model",
            input_example=X_test.head(3),
        )

        logger.info("=" * 60)
        logger.info(f"XGBOOST TRAINING — COMPLETE | Test Macro F1 = {all_metrics['test_macro_f1']:.4f}")
        logger.info(f"MLflow run ID: {run_id}")
        logger.info(f"NOT registered to production — LightGBM v16 remains canonical.")
        logger.info("=" * 60)


if __name__ == "__main__":
    main()
