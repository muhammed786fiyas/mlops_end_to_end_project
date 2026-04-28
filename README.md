# Football Match Outcome Prediction — End-to-End MLOps Pipeline

> Production-shape MLOps stack predicting Home / Draw / Away outcomes for European football matches.     
> **12 Docker containers · 4 DVC stages · 3 experiments compared in MLflow · 14/14 tests passing**  
> Test macro F1 = 0.4521 on held-out 2014/15 season; mild measured drift on unseen 2015/16 season.   

[![Tests](https://img.shields.io/badge/tests-14%2F14%20passing-brightgreen)](docs/test_plan/TEST_PLAN.md)
[![DVC pipeline](https://img.shields.io/badge/dvc-4%20stages-blue)](dvc.yaml)
[![MLflow](https://img.shields.io/badge/mlflow-2.18-orange)](http://localhost:5000)
[![Python](https://img.shields.io/badge/python-3.11-yellow)]()
[![Docker](https://img.shields.io/badge/docker-compose-2496ED)](docker-compose.yml)

---

## Demo

| | |
|---|---|
| **Operations Console** at `localhost:8080/dashboard.html` — one landing page linking all 8 system UIs | **Predict a match** at `localhost:8080` — pick two teams, see probability bars + container ID + latency |

> Screenshots and a 5-minute walkthrough video live in `demo/`. The compiled project report (LaTeX + PDF) lives in `report/`.

---

## Quick Start

Clone, configure, run. Three commands:

```bash
git clone git@github.com:muhammed786fiyas/mlops_end_to_end_project.git
cd mlops_end_to_end_project
cp .env.example .env    # fill in MailTrap creds (optional, only for email alerts)
docker compose up -d
```

Then open **http://localhost:8080/dashboard.html** for the operations console.

For full setup, prerequisites, and troubleshooting see the [User Manual](docs/user_manual/USER_MANUAL.md).

---

## What's running

| URL | Service | Purpose |
|---|---|---|
| http://localhost:8080/dashboard.html | **Operations Console** | Single navigation page linking all 8 UIs |
| http://localhost:8080 | Frontend | Make a prediction |
| http://localhost:8000/docs | Backend Swagger | Interactive API explorer |
| http://localhost:5000 | MLflow | Experiment runs + model registry |
| http://localhost:8081 | Airflow (`admin`/`admin`) | DAG status + manual trigger |
| http://localhost:9090 | Prometheus | Metrics + alert rule status |
| http://localhost:3000 | Grafana (`admin`/`admin`) | Service health dashboard |
| http://localhost:9093 | AlertManager | Active alerts |

---

## Architecture

12-container stack with deliberate separation of concerns: data lifecycle (DVC), ML lifecycle (MLflow), orchestration (Airflow), observability (Prometheus + Grafana + AlertManager + email relay), inference (FastAPI + nginx).

```
                    ┌──────────────┐  REST API  ┌─────────────┐
        Browser ──▶ │   frontend   │ ─────────▶ │   backend   │ ──▶ /predict, /metrics
                    │   (nginx)    │            │  (FastAPI)  │
                    └──────────────┘            └──────┬──────┘
                                                       │ loads model from
                                                       ▼
                          ┌──────────────────────────────────────┐
                          │  MLflow registry (registered models) │
                          └──────────────────────────────────────┘
                                          ▲
                                          │ logs runs from
                                          │
            ┌────────────────┐    ┌───────┴────────┐    ┌────────────────┐
            │ build_features │───▶│      train     │───▶│ evaluate_      │
            │  (DVC stage)   │    │ (DVC, LightGBM)│    │ production     │
            └────────────────┘    └────────────────┘    └────────────────┘
                                          ▲                     ▲
                                          │ runs weekly         │ runs daily
                                          │                     │
                                  ┌───────┴─────────┐  ┌────────┴────────┐
                                  │ training DAG    │  │ drift_check DAG │
                                  │ (Airflow)       │  │ (Airflow)       │
                                  └─────────────────┘  └─────────────────┘

Observability:  Prometheus → Grafana (dashboards) + AlertManager → mailtrap-relay → email
```

For full architectural detail with C4 diagrams, see [HLD](docs/hld/HLD.md).
For module-level design with API specs, see [LLD](docs/lld/LLD.md).

---

## Results

Three models compared via MLflow on the same chronological train/test split (2008–2014 train, 2014/15 test, 2015/16 held-out production).

| Model | Test Macro F1 | Test Accuracy | Status |
|---|---|---|---|
| Baseline LightGBM (32 features) | ~0.418 | ~0.45 | Day 2 baseline |
| **LightGBM + ELO (35 features)** | **0.4521** | 0.4716 | **Canonical, registered as v17** |
| XGBoost + ELO (35 features) | 0.3824 | 0.5150 | Experiment, not registered |

**Key finding:** Adding 3 ELO-rating features (home_elo, away_elo, elo_diff) improved test macro F1 by +0.034 over baseline. Per-class F1 improved across all three classes (H 0.55→0.56, D 0.27→0.30, A 0.45→0.50). XGBoost without explicit class weighting underfits the minority Draw class (F1 0.03) — empirically validates the LightGBM choice.

### Production drift evaluation

Same canonical model evaluated on the held-out 2015/16 season (3,262 matches the model never saw during training or test):

| Metric | 2014/15 (test) | 2015/16 (production) | Delta |
|---|---|---|---|
| Macro F1 | 0.4521 | 0.4379 | **−0.0142** |

Verdict (per the script's 5-band severity classification): **MILD DRIFT — monitor over next 2 seasons.** Small enough to keep serving, large enough to be worth watching. Empirically justifies the weekly retrain cadence.

---

## DVC pipeline

Four stages with parallel branches (`dvc dag`):

```
                      build_features
                       /     |      \
                      /      |       \
                     v       v        v
                  train  train_xgboost evaluate_production
                  (canonical) (experiment) (drift check)
```

Reproducible end-to-end:
```bash
dvc repro                # rebuild any stale stage
dvc repro train          # train only
dvc repro evaluate_production   # measure drift
./run_training.sh -e train      # canonical training via MLproject wrapper
```

---

## Repository layout

```
src/                    # Python source
├── api/                # FastAPI app + runtime feature service
├── features/           # build_features (DVC stage)
├── training/           # train.py + train_xgboost.py (DVC stages)
├── evaluation/         # evaluate_production.py (DVC stage)
└── utils/              # rotating-file logger

airflow/dags/           # 2 DAGs: training_pipeline + drift_check
monitoring/             # Prometheus + Grafana + AlertManager + mailtrap-relay
frontend/               # nginx static — index.html + dashboard.html
webhook_logger/         # alert receiver stub
tests/                  # 14 pytest cases (smoke + validation + contract)
docs/                   # HLD + LLD + USER_MANUAL + TEST_PLAN + daily_log
demo/                   # 5-minute project demo video
report/                 # LaTeX project report (source + PDF)
```

Full repo tree in [LLD §2](docs/lld/LLD.md#2-repository-layout).

---

## Documentation

| Document | Audience | Purpose |
|---|---|---|
| [User Manual](docs/user_manual/USER_MANUAL.md) | Anyone running the system | Install, run, troubleshoot, demo |
| [HLD](docs/hld/HLD.md) | Architects, graders | System context, container diagram, design rationale |
| [LLD](docs/lld/LLD.md) | Developers | Module design, API specs, schemas, sequences |
| [API Reference](docs/lld/api_endpoints.md) | API consumers | Endpoint details with I/O specs |
| [Test Plan](docs/test_plan/TEST_PLAN.md) | QA / graders | Test strategy, cases, results |
| [Daily logs](docs/daily_log/) | Curious readers | Per-day build journal |
| [Q&A study guide](football_mlops_qa_guide.tex) | Viva prep | Anticipated questions + answers |

---

## Milestones (git tags)

| Tag | Milestone | Test F1 |
|---|---|---|
| `data-v1` | Raw data DVC-locked | — |
| `model-baseline` | Original LightGBM trained | ~0.418 |
| `v0.4.0` | Airflow orchestration complete | ~0.418 |
| `v0.5.0` | Monitoring stack (Prometheus + Grafana + AlertManager) | ~0.418 |
| `v0.6.0` | Software engineering documentation (HLD + LLD + Test Plan + User Manual) | ~0.418 |
| `v0.7.0-elo` | ELO ratings as features | 0.4521 |
| `v0.7.1-experiments` | XGBoost comparison run | 0.4521 (LightGBM still wins) |
| `v0.7.2-drift-eval` | Production drift evaluation script | — |
| `v0.7.3-alerting` | Drift DAG + email alerts | — |

Each tag is annotated with the rationale for that milestone (`git show <tag>` to read).

---

## Tech stack

**ML:** Python 3.11, LightGBM 4.5, XGBoost 3.0, scikit-learn 1.4, pandas 2.2, numpy 1.26
**Serving:** FastAPI, uvicorn, Pydantic, prometheus-client
**Pipelines:** DVC 3.50, MLflow 2.18, Apache Airflow 3.1
**Observability:** Prometheus 2.55, Grafana 11, AlertManager 0.27, MailTrap REST API
**Infrastructure:** Docker Engine 20.10+, Docker Compose v2, Postgres 16

---

## Targets vs achieved

| Target | Goal | Achieved | Notes |
|---|---|---|---|
| Macro F1-score | ≥ 0.50 | 0.4521 | Below ambition target but in line with published research on this dataset |
| ROC-AUC | ≥ 0.70 | 0.6471 | Football outcomes are inherently noisy |
| Inference latency | < 200 ms | ~50 ms (p50) | Comfortably under target |
| Pipeline runtime | < 5 min | ~5 sec end-to-end | Includes feature engineering + training |

The F1 target was deliberately ambitious. F1=0.45 is **strong on this dataset** — Kaggle leaderboards on the same dataset hover around F1 0.42–0.50, with most academic papers reporting in the same range. The point of the project was end-to-end MLOps, not chasing F1.

---

## Course

DA5402 — MLOps Final Project (Indian Institute of Technology Madras)    
Author: **Muhammed Fiyas**  
Roll number : **DA25M018**
