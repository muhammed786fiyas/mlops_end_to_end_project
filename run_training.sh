#!/bin/bash
# Wrapper for `mlflow run .` that sets the experiment name
# Usage:
#   ./run_training.sh                                    # uses defaults
#   ./run_training.sh -P learning_rate=0.05              # override params
#   ./run_training.sh -e build_features -P rolling_window=7

mlflow run . \
  --experiment-name football-prediction \
  --env-manager=local \
  "$@"