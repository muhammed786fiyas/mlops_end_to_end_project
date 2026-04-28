# User Manual — Football MLOps End-to-End Project

**Document version:** 1.1   
**Last updated:** 2026-04-28 (Day 7)    
**Project:** DA5402 Final Project — Football Match Outcome Prediction   
**Author:** Muhammed Fiyas  
**Companion documents:** [HLD](../hld/HLD.md) · [LLD](../lld/LLD.md) · [Test Plan](../test_plan/TEST_PLAN.md)

---

## 1. What This System Does

This is an end-to-end MLOps system that predicts the outcome of European football matches — Home Win, Draw, or Away Win — given two competing teams. Open the web UI, pick two teams, click Predict, see the prediction with class probabilities and inference latency.

Behind the simple UI is a production-shape MLOps stack: data versioning (DVC), experiment tracking (MLflow), scheduled retraining (Airflow), monitoring (Prometheus + Grafana), and alerting (AlertManager). The full stack is 11 Docker containers.

This manual tells you how to install, run, and operate the system. For architectural detail see the [HLD](../hld/HLD.md); for code-level detail see the [LLD](../lld/LLD.md).

---

## 2. Quick Start

If you just want to see it work, three commands:

```bash
git clone git@github.com:muhammed786fiyas/mlops_end_to_end_project.git
cd mlops_end_to_end_project
docker compose up -d
```

Wait ~60 seconds for all services to come up, then open **http://localhost:8080/dashboard.html** in a browser for the operations console (one-stop entry to every UI in the system), or **http://localhost:8080** to make a prediction directly.

If anything fails, see [Section 9 — Troubleshooting](#9-troubleshooting).

---

## 3. Prerequisites

### 3.1 Required

| Tool | Minimum version | Why |
|---|---|---|
| **Docker Engine** | 20.10+ | Runs all 11 services |
| **Docker Compose** | v2 (`docker compose` — note the space) | Stack orchestration |
| **Git** | any modern version | Clone the repo |
| **8 GB RAM free** | — | All 11 containers fit comfortably under 4 GB; 8 GB allows headroom |
| **5 GB free disk** | — | Docker images + DVC remote + database |

Verify with:

```bash
docker --version          # Docker version 20.10.x or newer
docker compose version    # Docker Compose version v2.x or newer
git --version
free -h                   # check available RAM
df -h .                   # check disk space
```

### 3.2 Optional (for development, not required for demo)

| Tool | Why |
|---|---|
| **Conda** | To run the Python code outside Docker (training, tests) |
| **VS Code** | Editing markdown / Mermaid diagrams |
| **DVC CLI** | Version-controlling data outside Docker |
| **kaggle CLI** | Downloading the source dataset (only needed first time, and the SQLite is already DVC-tracked in this repo) |

### 3.3 Operating system

Tested on:

- **Ubuntu 24.04** (primary development environment)
- **Linux in general** (any distribution with Docker Engine)

Not tested but expected to work:

- **macOS with Docker Desktop**
- **Windows with WSL2 + Docker Desktop**

Native Windows Docker (without WSL2) is not supported — Airflow's volume permissions don't translate cleanly.

---

## 4. Installation

### 4.1 Clone the repository

```bash
git clone git@github.com:muhammed786fiyas/mlops_end_to_end_project.git
cd mlops_end_to_end_project
```

If you don't have SSH keys set up:

```bash
git clone https://github.com/muhammed786fiyas/mlops_end_to_end_project.git
cd mlops_end_to_end_project
```

### 4.2 Verify the data is present

The Kaggle SQLite database is tracked by DVC. To pull it:

```bash
dvc pull         # if you have DVC installed
```

Or download manually if DVC isn't available:

```bash
# Place database.sqlite in data/raw/
ls data/raw/database.sqlite    # should exist, ~300 MB
```

If the database is missing, the backend container will fail to start with a clear error — see [Section 9.2](#92-backend-container-fails-to-start).

### 4.3 Build and start the stack

```bash
docker compose up -d
```

What `up -d` does:

- Builds custom images (`backend`, `frontend`, `airflow-*`, `webhook-logger`) — first run takes ~10 minutes
- Pulls public images (`mlflow`, `postgres`, `prometheus`, `grafana`, `alertmanager`)
- Creates the `football-net` Docker bridge network
- Creates persistent named volumes (`prometheus-data`, `grafana-data`, etc.)
- Starts all containers in dependency order

On a fresh machine, the first `docker compose up` takes 10–15 minutes (mostly downloading base images and pip-installing into the backend image). Subsequent restarts take ~30 seconds.

### 4.4 Verify all containers are healthy

```bash
docker compose ps
```

Expected output (10 running containers + 1 exited init container):

```
NAME                         STATUS
football-airflow-init        Exited (0)             ← correct: one-shot DB migrator
football-airflow-scheduler   Up (healthy)
football-airflow-webserver   Up (healthy)
football-alertmanager        Up
football-backend             Up (healthy)
football-frontend            Up
football-grafana             Up
football-mlflow              Up
football-postgres            Up (healthy)
football-prometheus          Up
football-webhook-logger      Up
```

If any container shows `Restarting` or `Exited (1)`, see [Section 9](#9-troubleshooting).

### 4.5 Verify the API is responding

```bash
curl http://localhost:8000/health
```

Expected:

```json
{
  "status": "ok",
  "model_loaded": true,
  "container_id": "abc123def456",
  "timestamp": "2026-04-28T..."
}
```

`status: "ok"` confirms the model loaded successfully (either from MLflow or the local pickle fallback).

---

## 5. The Eight Web UIs

Each container exposing a UI is reachable on a different localhost port. The **operations console** at `/dashboard.html` provides a single landing page that links out to the other seven; each tool itself is independently useful with its own UI depth.

| URL | Service | Login | Purpose |
|---|---|---|---|
| http://localhost:8080/dashboard.html | **Operations Console** | — | **Single navigation page linking to all UIs** (Day 7) |
| http://localhost:8080 | Frontend | — | **Make a prediction** |
| http://localhost:8000/docs | Backend Swagger | — | Interactive API explorer |
| http://localhost:5000 | MLflow | — | Experiment runs + model registry |
| http://localhost:8081 | Airflow | `admin` / `admin` | DAG status + manual trigger |
| http://localhost:9090 | Prometheus | — | Metrics queries + alert rule status |
| http://localhost:3000 | Grafana | `admin` / `admin` | **Service health dashboard** |
| http://localhost:9093 | AlertManager | — | Active alerts + silencing |

---

## 6. Common Workflows

### 6.1 Make a prediction (the primary user flow)

1. Open **http://localhost:8080**
2. Select a team in the **Home** dropdown (e.g., "Manchester United")
3. Select a different team in the **Away** dropdown (e.g., "Liverpool")
4. Optionally pick a **match date** (defaults to today)
5. Click **Predict**

You'll see:

- The predicted outcome (Home Win / Draw / Away Win)
- Three colored probability bars (one per class)
- Inference latency in ms
- Container ID (proves it came from the Docker container, not a local fallback)
- Model version (which MLflow model produced this prediction)

### 6.2 View experiment history

1. Open **http://localhost:5000**
2. Click **Experiments** → `football-prediction`
3. See the table of all training runs with metrics
4. Click any run to see params, metrics, artifacts (including `feature_importance.png`)
5. Click **Models** in the top nav → `football-outcome-predictor` to see all registered model versions

### 6.3 Trigger a retraining manually

The DAG runs automatically every Sunday at 02:00 UTC, but you can trigger it manually:

```bash
# Create a new trigger file
date > data/triggers/retrain_trigger.txt
```

Then:

1. Open **http://localhost:8081** (Airflow UI; login `admin/admin`)
2. Find the `football_training_pipeline` DAG
3. Click the **▶** (Play) button → **Trigger DAG**
4. Watch the task graph fill in green as each task completes
5. The full DAG should complete in ~5 seconds (because DVC caches stage outputs by hash; nothing actually changed)

To **force** an actual retrain (even when DVC says nothing changed):

```bash
docker compose exec airflow-scheduler bash -c "cd /opt/airflow/project && dvc repro --force train"
```

This bypasses DVC's cache and produces a new MLflow model version.

### 6.4 Run the test suite

```bash
# From the host (requires a local conda env)
conda activate football-mlops
pip install pytest httpx
pytest tests/ -v
```

Expected: 14 tests pass in under 5 seconds. The tests run against the live Docker stack (specifically against `localhost:8000`), so the stack must be `up -d` before running.

### 6.5 View live metrics

1. Open **http://localhost:3000** (Grafana; login `admin/admin`, skip password change on first login)
2. Click **Dashboards** in left sidebar
3. Open **Football MLOps → Football MLOps — Service Overview**

You'll see 11 panels:

- **Top row:** Backend status, total predictions, CPU%, memory MB
- **Middle row:** Prediction rate by class (H/D/A), latency p50/p95/p99
- **Bottom row:** Error rate, errors by status code, predicted-class distribution, currently loaded model
- **Active Alerts table** at the very bottom

Generate some traffic to populate the panels:

```bash
for i in {1..30}; do
  curl -s -X POST http://localhost:8000/predict \
    -H "Content-Type: application/json" \
    -d '{"home_team_id": 8350, "away_team_id": 9880}' \
    -o /dev/null
  sleep 0.3
done
```

Refresh the dashboard. You should see traffic spikes in the rate panels.

### 6.6 Trigger a test alert

Stop the backend container and watch the BackendDown alert fire:

```bash
docker compose stop backend

# Wait ~75 seconds (15s scrape lag + 60s alert suppression window)
sleep 75

# Check the webhook receiver received the alert
docker compose logs webhook-logger --tail 20
```

You should see a formatted alert payload with `alert: BackendDown`, `severity: critical`, etc. Then bring backend back up:

```bash
docker compose start backend
```

### 6.7 Run production drift evaluation (Day 7)

```bash
docker compose exec airflow-scheduler bash -c "cd /opt/airflow/project && python -m src.evaluation.evaluate_production"
```

Evaluates the canonical model on the held-out 2015/16 season (3,262 unseen matches) and reports drift severity vs the canonical test set:

```
DRIFT COMPARISON
  Test F1 (2014/15):       0.4521
  Production F1 (2015/16): 0.4379
  Drift (prod - test):     -0.0142
  Verdict: MILD DRIFT detected (-0.014 F1) — monitor over next 2 seasons
```

The verdict uses 5 severity bands (STABLE / MILD DRIFT / DRIFT / SEVERE DRIFT / IMPROVED). Results are logged to MLflow as a `production-drift-check-<git_hash>` run for permanent tracking, and a JSON drift report is saved to `models/production_drift_report.json`.

This script runs inside the airflow-scheduler container because that's where the production DAG would invoke it. The script falls back to the local model pickle if the MLflow model registry is unreachable — same graceful degradation pattern the FastAPI backend uses.

### 6.8 Stop the stack

```bash
docker compose down              # stops and removes containers; volumes persist
docker compose down -v           # also removes volumes (full reset)
```

`docker compose down` is safe — restarting takes ~30 seconds and your MLflow runs, Grafana dashboards, and Prometheus history are preserved in named volumes.

---

## 7. Configuration

### 7.1 Path config — `config.yaml`

If you've moved files around, edit the paths here. Out-of-the-box defaults work for the standard repo layout:

```yaml
data:
  raw_db: "data/raw/database.sqlite"
  processed_dir: "data/processed"
  production_dir: "data/production"
```

### 7.2 Hyperparameters — `params.yaml`

DVC-tracked, with a nested `algorithms.{lightgbm,xgboost}` structure (Day 7 refactor) so different model types can coexist without naming collisions. To experiment with different settings:

1. Edit `params.yaml` (e.g., change `algorithms.lightgbm.num_leaves` from 15 to 31)
2. Run `dvc repro` from the host or inside the airflow-scheduler container
3. DVC re-runs only the affected pipeline stages (`train`, since lightgbm params changed)
4. A new MLflow run appears with the updated params and metrics

Two parallel training stages are defined: `train` (canonical LightGBM, registered as production) and `train_xgboost` (XGBoost comparison experiment, not registered). Run a specific stage with `dvc repro <stage_name>`.

### 7.3 Authentication

Out-of-the-box credentials:

- **Airflow:** `admin` / `admin` (set in `airflow/config/simple_auth_manager_passwords.json.generated`)
- **Grafana:** `admin` / `admin` (set as env vars in `docker-compose.yml`)
- **MLflow:** no authentication (academic deployment)
- **Prometheus / AlertManager:** no authentication (academic)

For a production deployment, replace these with secrets injected via `.env` and use a reverse proxy (Traefik or Nginx) for HTTPS termination.

---

## 8. Day-to-Day Operations

### 8.1 Restart a single service

If one container is misbehaving:

```bash
docker compose restart <service-name>
# e.g.
docker compose restart backend
```

### 8.2 View logs

```bash
# All services, live tail
docker compose logs -f

# Single service
docker compose logs backend

# Last 50 lines, no follow
docker compose logs --tail 50 prometheus
```

The application also writes its own logs to `logs/football_mlops.log` (rotating, 10 MB × 5 backups). View with:

```bash
tail -f logs/football_mlops.log
```

### 8.3 Update the model in production

The backend always loads `models:/football-outcome-predictor/latest` from MLflow's registry at startup. To deploy a new version:

1. Train a new model (manually or via Airflow DAG)
2. Verify in MLflow UI that a new version was registered
3. Restart the backend: `docker compose restart backend`
4. Check `/health` to confirm the new `model_version` is loaded

### 8.4 Roll back to a previous model

1. Open MLflow UI → **Models** → `football-outcome-predictor`
2. Note the version number you want to roll back to (e.g., v12)
3. Edit `docker-compose.yml`, set `MLFLOW_MODEL_STAGE=12` for the backend service
4. `docker compose up -d backend` to restart with the new env var

Alternatively, use MLflow's stage-based aliases (`@production`, `@staging`) for non-numerical rollback.

---

## 9. Troubleshooting

### 9.1 "docker compose: command not found"

You have Docker Compose v1 (`docker-compose` with hyphen) but the project uses v2 (`docker compose` with space). Either:

- Install Docker Compose v2 (bundled with modern Docker Engine)
- Or replace every `docker compose` in the commands above with `docker-compose`

### 9.2 Backend container fails to start

```bash
docker compose logs backend
```

Common errors:

**`FileNotFoundError: SQLite database not found at ...`**
The Kaggle dataset is missing. Run `dvc pull` or download manually to `data/raw/database.sqlite`.

**`OSError: No such file or directory: '/mlflow/mlflow_artifacts/...'`**
The MLflow artifact mount is missing. Check `docker-compose.yml` — backend should have `./mlflow_artifacts:/mlflow/mlflow_artifacts` in its `volumes:` block. This was a Day 4 bug we fixed; if you cloned a fresh copy the fix is already in place.

**`MLflow load failed (..) Falling back to local pickle`**
Not an error — backend tried MLflow first, fell back to the on-disk pickle. Service is functional. Investigate by checking that MLflow container is up: `docker compose ps mlflow`.

### 9.3 Grafana dashboard shows "No data"

The dashboard is loaded but Prometheus has no metrics yet.

- Check Prometheus targets: **http://localhost:9090/targets** — `football-backend` should be `UP`
- If `DOWN`, the backend isn't reachable from Prometheus. Check both are on the `football-net` network: `docker network inspect mlops_end_to_end_project_football-net`
- If `UP` but Grafana panels are still empty, generate some traffic (Section 6.5 has a curl loop)

### 9.4 Grafana panels show red triangles

The dashboard's datasource UID doesn't match the provisioned Prometheus datasource. This was fixed in Day 5 by pinning `uid: prometheus` in `monitoring/grafana/provisioning/datasources/prometheus.yml`. If you're seeing this on a fresh clone, verify that file hasn't been corrupted.

### 9.5 Airflow webserver login rejected

The Simple Auth Manager regenerates passwords on every restart unless `airflow/config/simple_auth_manager_passwords.json.generated` is present. The repo includes this file pre-populated with `{"admin": "admin"}`. If login still fails:

```bash
cat airflow/config/simple_auth_manager_passwords.json.generated
# should contain: {"admin": "admin"}
```

If empty or missing, recreate it:

```bash
echo '{"admin": "admin"}' > airflow/config/simple_auth_manager_passwords.json.generated
docker compose restart airflow-webserver
```

### 9.6 DAG shows green but no new MLflow model appeared

This is **DVC working as designed**, not a bug. `dvc repro train` checks input hashes; if nothing changed, the stage is skipped silently. Force a re-run:

```bash
docker compose exec airflow-scheduler bash -c "cd /opt/airflow/project && dvc repro --force train"
```

Or modify a tracked input (e.g., bump `algorithms.lightgbm.num_leaves` in `params.yaml`).

### 9.7 Resolved alert notification didn't arrive at webhook-logger

This is documented AlertManager behavior, not a bug. AlertManager respects `group_interval: 5m` — alerts that resolve faster than 5 minutes are intentionally not resent as "resolved" to reduce notification spam. In production, real outages last longer than 5 minutes and you'll see both notifications.

### 9.8 Out-of-memory errors during `docker compose up`

The 11-container stack uses ~3 GB RAM at peak. If your machine has less than 4 GB free:

- Close other applications
- Or reduce the stack: bring up only the core services (`mlflow`, `backend`, `frontend`) and skip the observability stack temporarily

```bash
# Minimal stack — just inference, no monitoring
docker compose up -d mlflow backend frontend
```

### 9.9 The frontend loads but predictions fail with CORS errors

Open browser DevTools → Network tab. If you see CORS-blocked requests, the backend's CORS middleware isn't configured correctly. Check `src/api/main.py` for:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    ...
)
```

This is open by default for local demo. Locked down in production.

---

## 10. Demo Walkthrough (5-minute version)

For a quick demonstration of every rubric area:

| Time | Action | URL | What to point out |
|---|---|---|---|
| 0:00 | `docker compose up -d` | terminal | "11 containers, one command" |
| 0:30 | Open operations console, click into prediction UI | localhost:8080/dashboard.html | "One landing page; each tool keeps its native UI depth" |
| 0:45 | Predict Real Madrid vs Granada | localhost:8080 | "Probability bars + container ID + latency" |
| 1:15 | Open MLflow (3 runs to compare) | localhost:5000 | "Baseline, ELO experiment, XGBoost experiment all logged" |
| 1:45 | Open Airflow | localhost:8081 | "Scheduled DAG with sensor + pool + retries + quality gate" |
| 2:15 | Open Grafana | localhost:3000 | "11-panel dashboard following Four Golden Signals" |
| 2:45 | Open Prometheus targets | localhost:9090/targets | "Backend /metrics scraped every 15s" |
| 3:15 | Stop backend, watch BackendDown fire | terminal + localhost:9093 | "Alert state machine: inactive → pending → firing" |
| 3:45 | Restart backend | terminal | "Recovery" |
| 4:00 | Run pytest suite | terminal | "14 tests in under 5 seconds; contract test prevents silent train/serve drift" |
| 4:30 | Run drift evaluation script | terminal | "Mild drift detected on unseen 2015/16 season; F1 −1.4%" |
| 4:45 | Show GitHub repo with daily logs and HLD | github.com | "Full design rationale in version control" |
| 5:00 | Done |

---

## 11. Where to Get Help

- **Architecture questions:** [HLD](../hld/HLD.md)
- **Code-level questions:** [LLD](../lld/LLD.md)
- **What's tested and why:** [Test Plan](../test_plan/TEST_PLAN.md)
- **API spec:** http://localhost:8000/docs (Swagger UI, requires backend running) or [api_endpoints.md](../lld/api_endpoints.md)
- **Daily build journal:** `docs/daily_log/day_NN.md`
- **Q&A study guide (LaTeX):** `football_mlops_qa_guide.tex`

---

## 12. Document History

| Version | Date | Author | Notes |
|---|---|---|---|
| 1.0 | 2026-04-28 | Muhammed Fiyas | Initial user manual covering v0.5.0 milestone |
| 1.1 | 2026-04-28 | Muhammed Fiyas | Day 7: added operations console (8 UIs total), drift evaluation workflow (§6.7), nested algorithms params structure note (§7.2), updated demo walkthrough |