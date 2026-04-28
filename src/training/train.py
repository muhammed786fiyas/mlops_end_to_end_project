"""
Training pipeline for the football-mlops project.

Loads engineered features, trains a regularized LightGBM classifier, 
evaluates on a held-out test set, logs everything to MLflow, and 
persists artifacts to disk.

Usage:
    python -m src.training.train                    # standalone
    python -m src.training.train --learning-rate 0.05 --num-leaves 31
    mlflow run . --env-manager=local                # via MLproject
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import mlflow
import mlflow.lightgbm
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, roc_auc_score
)

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# MLflow configuration
# ---------------------------------------------------------------------------
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_EXPERIMENT_NAME = "football-prediction"
MLFLOW_REGISTERED_MODEL_NAME = "football-outcome-predictor"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

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
    # ELO (3) — Day 7 experiment
    'home_elo', 'away_elo', 'elo_diff',
]
TARGET_COLUMN = 'outcome_encoded'
TARGET_NAMES = ['H', 'D', 'A']


# ---------------------------------------------------------------------------
# Pipeline functions
# ---------------------------------------------------------------------------

def load_train_test(processed_dir: Path) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    logger.info(f"Loading train/test from {processed_dir}")

    train_df = pd.read_csv(processed_dir / "train.csv")
    test_df = pd.read_csv(processed_dir / "test.csv")
    logger.info(f"  Train: {len(train_df):,} rows, {len(train_df.columns)} cols")
    logger.info(f"  Test:  {len(test_df):,} rows, {len(test_df.columns)} cols")

    missing = [c for c in FEATURE_COLUMNS if c not in train_df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    if TARGET_COLUMN not in train_df.columns:
        raise ValueError(f"Missing target column: {TARGET_COLUMN}")

    X_train = train_df[FEATURE_COLUMNS].copy()
    y_train = train_df[TARGET_COLUMN].copy()
    X_test = test_df[FEATURE_COLUMNS].copy()
    y_test = test_df[TARGET_COLUMN].copy()

    assert X_train.isna().sum().sum() == 0, "Nulls in X_train"
    assert X_test.isna().sum().sum() == 0, "Nulls in X_test"
    assert set(y_train.unique()).issubset({0, 1, 2}), f"Unexpected target values: {y_train.unique()}"

    logger.info(f"  Features: {len(FEATURE_COLUMNS)} columns, no nulls")
    return X_train, y_train, X_test, y_test


def chronological_train_val_split(
    X: pd.DataFrame, y: pd.Series, val_fraction: float
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    n_val = int(len(X) * val_fraction)
    X_train_fit = X.iloc[:-n_val]
    y_train_fit = y.iloc[:-n_val]
    X_val = X.iloc[-n_val:]
    y_val = y.iloc[-n_val:]
    logger.info(f"  Carved validation set: {len(X_val):,} rows ({val_fraction*100:.0f}% of train)")
    return X_train_fit, y_train_fit, X_val, y_val


def train_model(
    X_train: pd.DataFrame, y_train: pd.Series,
    X_val: pd.DataFrame, y_val: pd.Series,
    model_params: dict, early_stopping_rounds: int,
) -> lgb.LGBMClassifier:
    logger.info("Training LightGBM classifier")
    logger.info(f"  num_leaves={model_params['num_leaves']}, "
                f"learning_rate={model_params['learning_rate']}, "
                f"n_estimators={model_params['n_estimators']}")

    lgb_params = {k: v for k, v in model_params.items() if k != 'algorithm'}
    model = lgb.LGBMClassifier(**lgb_params)

    start = datetime.now()
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False)],
    )
    elapsed = (datetime.now() - start).total_seconds()

    logger.info(f"  Training complete in {elapsed:.1f}s")
    logger.info(f"  Best iteration: {model.best_iteration_} "
                f"(of {model_params['n_estimators']} max)")
    return model


def evaluate(model: lgb.LGBMClassifier, X: pd.DataFrame, y: pd.Series, split_name: str) -> dict:
    logger.info(f"Evaluating on {split_name} set ({len(X):,} rows)")

    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)

    macro_f1 = f1_score(y, y_pred, average='macro')
    accuracy = accuracy_score(y, y_pred)
    roc_auc = roc_auc_score(y, y_proba, multi_class='ovr', average='macro')
    per_class_f1 = f1_score(y, y_pred, average=None).tolist()
    cm = confusion_matrix(y, y_pred)

    metrics = {
        'macro_f1': float(macro_f1),
        'roc_auc_ovr_macro': float(roc_auc),
        'accuracy': float(accuracy),
        'per_class_f1': dict(zip(TARGET_NAMES, [float(x) for x in per_class_f1])),
        'confusion_matrix': cm.tolist(),
    }

    logger.info(f"  Macro F1:  {macro_f1:.4f}")
    logger.info(f"  ROC-AUC:   {roc_auc:.4f}")
    logger.info(f"  Accuracy:  {accuracy:.4f}")
    logger.info(f"  Per-class F1: H={per_class_f1[0]:.3f}, D={per_class_f1[1]:.3f}, A={per_class_f1[2]:.3f}")
    return metrics


def save_artifacts(
    model: lgb.LGBMClassifier, metrics: dict, model_params: dict,
    n_train: int, n_val: int, n_test: int, models_dir: Path,
) -> None:
    models_dir.mkdir(parents=True, exist_ok=True)

    try:
        git_hash = subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            cwd=models_dir.parent, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        git_hash = "unknown"

    model_path = models_dir / "lightgbm_baseline.pkl"
    joblib.dump(model, model_path)
    logger.info(f"Saved model: {model_path}")

    metrics_path = models_dir / "metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved metrics: {metrics_path}")

    metadata = {
        'training_date': datetime.now().isoformat(),
        'git_commit': git_hash,
        'algorithm': 'lightgbm',
        'n_features': len(FEATURE_COLUMNS),
        'feature_names': FEATURE_COLUMNS,
        'n_train': n_train,
        'n_val': n_val,
        'n_test': n_test,
        'best_iteration': int(model.best_iteration_),
        'model_params': model_params,
        'final_test_macro_f1': metrics['test']['macro_f1'],
        'final_test_roc_auc': metrics['test']['roc_auc_ovr_macro'],
    }
    metadata_path = models_dir / "training_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved metadata: {metadata_path}")

    fig, ax = plt.subplots(figsize=(10, 8))
    importances = pd.Series(model.feature_importances_, index=FEATURE_COLUMNS).sort_values(ascending=True)
    importances.plot(kind='barh', ax=ax)
    ax.set_title('LightGBM Feature Importance — Baseline Model')
    ax.set_xlabel('Importance (split count)')
    plt.tight_layout()
    fi_path = models_dir / "feature_importance.png"
    plt.savefig(fi_path, dpi=100, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved feature importance plot: {fi_path}")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def _project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in [current, *current.parents]:
        if (parent / "config.yaml").exists():
            return parent
    raise FileNotFoundError("Could not find project root")


def main(
    learning_rate: Optional[float] = None,
    num_leaves: Optional[int] = None,
    n_estimators: Optional[int] = None,
):
    """Run the training pipeline end-to-end with MLflow tracking."""
    logger.info("=" * 60)
    logger.info("TRAINING PIPELINE — START")
    logger.info("=" * 60)

    # --- MLflow setup ---
    # If called via `mlflow run`, MLflow already configured tracking + started a run.
    # Otherwise, we set it up ourselves for standalone execution.
    invoked_via_mlflow_run = "MLFLOW_RUN_ID" in os.environ
    if not invoked_via_mlflow_run:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    logger.info(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
    logger.info(f"Invoked via mlflow run: {invoked_via_mlflow_run}")

    project_root = _project_root()
    with open(project_root / "config.yaml") as f:
        config = yaml.safe_load(f)
    with open(project_root / "params.yaml") as f:
        params = yaml.safe_load(f)

    processed_dir = project_root / config['data']['processed_dir']
    models_dir = project_root / "models"

    # Apply CLI overrides
    model_params = dict(params['model'])
    if learning_rate is not None:
        model_params['learning_rate'] = learning_rate
    if num_leaves is not None:
        model_params['num_leaves'] = num_leaves
    if n_estimators is not None:
        model_params['n_estimators'] = n_estimators

    try:
        git_hash = subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            cwd=project_root, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        git_hash = "unknown"

    # --- Start MLflow run ---
    run_name = (
        f"lgbm-lr{model_params['learning_rate']}"
        f"-leaves{model_params['num_leaves']}"
        f"-window{params['features']['rolling_window_size']}"
        f"-{git_hash}"
    )
    # If `mlflow run` started a run, attach to it. Otherwise, create a new one.
    start_run_kwargs = {"run_name": run_name}

    with mlflow.start_run(**start_run_kwargs) as run:
        run_id = run.info.run_id
        logger.info(f"MLflow run_id: {run_id}")

        mlflow.set_tags({
            "git_commit": git_hash,
            "stage": "dvc_train",
            "framework": "lightgbm",
            "dataset": "european_soccer",
            "run_label": run_name,
        })

        mlflow.log_params(model_params)
        mlflow.log_params({f"training.{k}": v for k, v in params['training'].items()})
        mlflow.log_params({f"features.{k}": v for k, v in params['features'].items()})

        X_train, y_train, X_test, y_test = load_train_test(processed_dir)
        X_train_fit, y_train_fit, X_val, y_val = chronological_train_val_split(
            X_train, y_train, val_fraction=params['training']['val_fraction']
        )
        model = train_model(
            X_train_fit, y_train_fit, X_val, y_val,
            model_params=model_params,
            early_stopping_rounds=params['training']['early_stopping_rounds'],
        )

        mlflow.log_params({
            "n_train": len(X_train_fit),
            "n_val": len(X_val),
            "n_test": len(X_test),
            "n_features": len(FEATURE_COLUMNS),
        })
        mlflow.log_metric("best_iteration", model.best_iteration_)

        train_metrics = evaluate(model, X_train_fit, y_train_fit, 'train')
        val_metrics = evaluate(model, X_val, y_val, 'val')
        test_metrics = evaluate(model, X_test, y_test, 'test')
        all_metrics = {'train': train_metrics, 'val': val_metrics, 'test': test_metrics}

        for split_name, m in all_metrics.items():
            mlflow.log_metrics({
                f"{split_name}_macro_f1": m['macro_f1'],
                f"{split_name}_roc_auc": m['roc_auc_ovr_macro'],
                f"{split_name}_accuracy": m['accuracy'],
                f"{split_name}_f1_H": m['per_class_f1']['H'],
                f"{split_name}_f1_D": m['per_class_f1']['D'],
                f"{split_name}_f1_A": m['per_class_f1']['A'],
            })

        save_artifacts(
            model=model, metrics=all_metrics, model_params=model_params,
            n_train=len(X_train_fit), n_val=len(X_val), n_test=len(X_test),
            models_dir=models_dir,
        )

        mlflow.log_artifact(str(models_dir / "metrics.json"))
        mlflow.log_artifact(str(models_dir / "training_metadata.json"))
        mlflow.log_artifact(str(models_dir / "feature_importance.png"))

        mlflow.lightgbm.log_model(
            lgb_model=model,
            name="model",
            registered_model_name=MLFLOW_REGISTERED_MODEL_NAME,
            input_example=X_test.head(3),
        )
        logger.info(f"Model registered as '{MLFLOW_REGISTERED_MODEL_NAME}'")

    logger.info("=" * 60)
    logger.info(f"TRAINING PIPELINE — COMPLETE | Test Macro F1 = {test_metrics['macro_f1']:.4f}")
    logger.info(f"MLflow run ID: {run_id}")
    logger.info("=" * 60)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LightGBM classifier")
    parser.add_argument('--learning-rate', type=float, default=None)
    parser.add_argument('--num-leaves', type=int, default=None)
    parser.add_argument('--n-estimators', type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        n_estimators=args.n_estimators,
    )