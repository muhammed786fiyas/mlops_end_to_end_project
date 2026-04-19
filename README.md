# Football Match Outcome Prediction — End-to-End MLOps Pipeline

Production-grade AI application predicting football match outcomes (Home Win / Draw / Away Win) using the European Soccer Database.

**Tech stack:** Python, LightGBM, MLflow, Apache Airflow, DVC, FastAPI, Docker Compose, Prometheus, Grafana.

## Targets
- Macro F1-score ≥ 0.50
- ROC-AUC ≥ 0.70
- Inference latency < 200ms
- Pipeline runtime < 5 minutes

## Setup
```bash
conda env create -f environment.yml
conda activate football-mlops
```

## Project Structure
See `docs/` for architecture diagram, HLD, and LLD.

## Course
DA5402 — MLOps Final Project