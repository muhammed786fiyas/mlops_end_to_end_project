# Day 4 — Airflow Orchestration in Docker
**Date:** 2026-04-23 (continued 04-27)
**Time spent:** ~6 hours (Day 4 main session) + ~1 hour follow-up (model registry fix)
**Rubric mapping:** MLOps Implementation (Data Engineering with Airflow — sensors, pools, retries, scheduling), Software Engineering (orchestration design), Software Packaging (multi-service Docker stack expanded to 6 containers), Viva (debugging journey + producer/consumer patterns)

## What I shipped today

- **Airflow Docker image** (`docker/Dockerfile.airflow`) — extends `apache/airflow:3.1.8` with git, libgomp1, DVC, LightGBM, MLflow, pandas; ~9 min build
- **Expanded `docker-compose.yml`** from 3 services to 6 using `x-airflow-common` YAML anchor:
  - `mlflow` (existing)
  - `backend` (existing — patched to mount `mlflow_artifacts/`)
  - `frontend` (existing)
  - `postgres:16` — Airflow metadata DB on `football-net`
  - `airflow-init` — one-shot DB migration container
  - `airflow-scheduler` — scheduler with LocalExecutor
  - `airflow-webserver` — UI on port 8081 (note: command is `api-server` in 3.x)
- **Football training DAG** (`airflow/dags/football_training_dag.py`, ~140 lines) with 6 tasks:
  - `data_sensor` (FileSensor, reschedule mode, 30s poke / 5min timeout)
  - `validate_data` (@task — checks trigger freshness <24h + raw DB size >1MB)
  - `build_features` (BashOperator — `dvc repro build_features`, `pool='training_pool'`)
  - `train_model` (BashOperator — `dvc repro train`, `pool='training_pool'`)
  - `check_metrics` (@task — F1 quality gate at 0.40 threshold)
  - `dry_pipeline_alert` (@task — `trigger_rule='one_failed'`, fires when sensor times out)
- **Persistent admin login** — `airflow/config/simple_auth_manager_passwords.json.generated` with `admin/admin`, pointed via `AIRFLOW__CORE__SIMPLE_AUTH_MANAGER_PASSWORDS_FILE`
- **Trigger directory** (`data/triggers/.gitkeep`) — host-side mount for the FileSensor; runtime `retrain_trigger.txt` is gitignored
- **Airflow Pool** `training_pool` with 1 slot — prevents parallel heavy training
- **Airflow Connection** `fs_default` — manually created (not auto-provisioned in 3.x) so FileSensor can resolve filesystem paths
- **Backend mlflow_artifacts mount** (follow-up fix) — added `./mlflow_artifacts:/mlflow/mlflow_artifacts` to the `backend` service so MLflow client can read model files registered by training tasks
- **Tag `v0.4.0`** marking the Day 4 milestone

## Key design decisions (viva-ready)

- **LocalExecutor over CeleryExecutor**
  - Alternatives considered: CeleryExecutor (1 broker + N workers), KubernetesExecutor
  - Why LocalExecutor: same rubric points, half the containers. CeleryExecutor needs Redis/RabbitMQ + at least one celery worker — 3 extra services for a project that runs one DAG. LocalExecutor lets the scheduler spawn task subprocesses directly. Mentions in viva: "I would switch to CeleryExecutor if the workload grew past one worker's capacity, or to KubernetesExecutor in a Kubernetes-native deployment."

- **Airflow tasks call `dvc repro`, not Python directly**
  - Alternatives considered: Airflow tasks import Python modules and call functions; subprocess into `python -m src.training.train`
  - Why `dvc repro`: clean separation of concerns. Airflow controls *when* to check; DVC controls *what* to do. DVC's input-hash caching automatically skips stages whose inputs haven't changed — preventing wasted compute. Each tool plays its proper role.

- **FileSensor on a separate trigger file, not the data file directly**
  - Alternatives: poke `data/raw/database.sqlite` (always exists → always succeeds → degenerate sensor); use HttpSensor or S3KeySensor
  - Why a sentinel file: producer/consumer handshake. Upstream data pipeline writes the trigger as its **last step**, after the data file is fully written. Pattern is identical to Hadoop's `_SUCCESS` file or Spark's `_SUCCESS` marker. Maps cleanly to a real production data-arrival workflow.

- **`mode="reschedule"` on FileSensor (not default `poke`)**
  - Why: `poke` mode holds a worker slot the entire wait period. With `timeout=5min` (or hours in production), that worker is locked doing nothing. `reschedule` releases the slot between polls, freeing capacity for other tasks.

- **Airflow Pool `training_pool` with 1 slot**
  - Alternatives: `max_active_tis_per_dag=1`, no concurrency control
  - Why a pool: explicit, named resource constraint. Multiple DAGs (future Day 6/7 experiments) can share the same pool. Pools are the rubric-explicit answer to "manage resource bottlenecks."

- **`dry_pipeline_alert` with `trigger_rule="one_failed"`**
  - Why: demonstrates branching pattern beyond the happy path. Sensor times out → alert fires. In production this would be EmailOperator or Slack webhook; here it's a log warning so we don't need SMTP setup. Pattern is the value, not the channel.

- **F1 quality gate (`check_metrics` raises if F1 < 0.40)**
  - Alternatives: log only, don't fail
  - Why fail: this is the model-promotion gate. In a richer pipeline this task would block promotion to MLflow's "Production" stage. Today it just fails the DAG, but the **principle** is the rubric-relevant point: ML pipelines need automated quality checks before deployment.

- **Airflow runs as UID 1000 (matching host user) via `${AIRFLOW_UID:-50000}:0` + `.env`**
  - Why: avoids root-owned files in the host's `airflow/dags/` and `airflow/logs/` folders. Folder-level read/write works without `sudo` on the host side. The `:-50000` fallback keeps the compose file portable for graders whose UID differs.

- **6-container stack vs. 4-container minimum**
  - Why 6: Airflow 3.x cleanly separates init, scheduler, webserver. Postgres is non-negotiable for LocalExecutor. We deliberately skipped `triggerer` (only needed for async deferrable operators) and `dag-processor` (only needed at scale). Production-shape stack without unnecessary services.

## Problems faced & how I solved them

This was the heaviest debugging day of the project. Documenting all 17 distinct issues so the viva story is concrete:

### Airflow 3.x migration quirks

- **Problem:** `airflow webserver` command not found
  - **Root cause:** Renamed in 3.x
  - **Fix:** Changed compose `command: webserver` → `command: api-server`

- **Problem:** `airflow users create` removed
  - **Root cause:** 3.x replaced FAB-based auth with pluggable Simple Auth Manager
  - **Fix:** Created `airflow/config/simple_auth_manager_passwords.json.generated` with `{"admin": "admin"}`; wired via `AIRFLOW__CORE__SIMPLE_AUTH_MANAGER_PASSWORDS_FILE` env var

- **Problem:** Healthcheck failed at `/health`
  - **Root cause:** Endpoint moved in 3.x
  - **Fix:** Changed test URL to `/api/v2/monitor/health`

- **Problem:** Task workers crashed with `httpx.ConnectError: Connection refused`
  - **Root cause:** 3.x introduced an internal Task Execution API that workers must reach
  - **Fix:** Added `AIRFLOW__CORE__EXECUTION_API_SERVER_URL=http://airflow-webserver:8080/execution/` and `AIRFLOW__API__BASE_URL=http://airflow-webserver:8080/`

- **Problem:** `Simple Auth Manager USERS` env var format `username:password` produced random passwords on every restart
  - **Root cause:** In 3.x the `USERS` env var format is `username:role` (not username:password). Passwords are auto-generated and persisted to `$AIRFLOW_HOME/simple_auth_manager_passwords.json.generated` only if that file already exists.
  - **Fix:** Pre-created the JSON file with `{"admin": "admin"}` and pointed Airflow at it via the `_PASSWORDS_FILE` env var

- **Problem:** `FileSensor` raised `AirflowNotFoundException: The conn_id 'fs_default' isn't defined`
  - **Root cause:** In 3.x the default filesystem connection is no longer auto-created
  - **Fix:** `airflow connections add fs_default --conn-type fs --conn-extra '{"path": "/"}'`

### Container UID / permission chain

- **Problem:** `entrypoint: /bin/bash` override caused `ModuleNotFoundError: No module named 'airflow'` for UID 1000
  - **Root cause:** The official Airflow image's custom entrypoint (`/entrypoint`) sets up PATH/PYTHONPATH for non-default UIDs. Overriding entrypoint bypasses that setup.
  - **Fix:** Remove `entrypoint: /bin/bash`; pass bash via `command: [bash, -c, ...]` instead so Airflow's entrypoint runs first

- **Problem:** Setting `PYTHONPATH=/opt/airflow/project` env var caused the same `ModuleNotFoundError`
  - **Root cause:** Custom PYTHONPATH override clobbered the entrypoint's path setup
  - **Fix:** Removed PYTHONPATH from env block entirely; rely on Airflow's defaults

- **Problem:** Host couldn't write to `airflow/dags/` after container ran as UID 50000
  - **Fix:** `sudo chown -R 1000:0 airflow/`; pinned compose `user: "${AIRFLOW_UID:-50000}:0"` with `AIRFLOW_UID=1000` in `.env`

### MLflow artifact path chain

- **Problem:** `train_model` failed with `PermissionError: '/home/muhammed786fiyas'`
  - **Root cause:** MLflow `experiments.artifact_location` in `mlflow.db` still pointed to host paths from Day 3 host-side training
  - **Fix:** SQL UPDATE on `experiments`, `runs`, `model_versions` tables to rewrite host paths to container paths (`/home/muhammed786fiyas/.../mlflow_artifacts` → `/mlflow/mlflow_artifacts`)

- **Problem:** Training failed with `PermissionError: '/mlflow'` after path fix
  - **Root cause:** Airflow container had no mount at `/mlflow/mlflow_artifacts/`
  - **Fix:** Added `./mlflow_artifacts:/mlflow/mlflow_artifacts` to `x-airflow-common.volumes`

- **Problem (Day 4 follow-up):** Backend silently fell back to local pickle on every startup with `OSError: No such file or directory: '/mlflow/mlflow_artifacts/.../artifacts/.'`
  - **Root cause:** Same mount missing on backend service. MLflow client reads model files directly from filesystem when given a `models:/` URI; metadata-over-HTTP isn't enough.
  - **Fix:** Added `./mlflow_artifacts:/mlflow/mlflow_artifacts` to backend's volumes too. After fix, log shows `Loaded model from MLflow registry`.

### DVC caching gotcha (the most insidious bug)

- **Problem:** Every Airflow DAG run reported success, but no new MLflow model versions appeared
  - **Root cause:** `dvc repro train` saw no input changes (same data, same code) and reported `Stage 'train' didn't change, skipping`. Training script never executed; no model was logged.
  - **Fix:** Manual one-shot `dvc repro --force train` to register a fresh model with proper artifacts (model version 15)
  - **Going forward:** This is *correct* DVC behavior — training only re-runs when inputs change. Documented as a viva talking point: "Airflow controls *when* to check, DVC controls *what* to do."

### Pip & build issues

- **Problem:** `pip install "lightgbm>=4.0,<5.0"` in Dockerfile failed
  - **Root cause:** Bash interpreted `<` and `>` as redirect operators
  - **Fix:** Quote the version constraints; added `--default-timeout=300 --retries 10` for the slow build

### Stuck DAG runs blocking each other

- **Problem:** Multiple queued DAG runs piled up; new triggers stayed in `queued` state forever
  - **Root cause:** `max_active_runs=1` blocks new runs while an older "running" run is still alive — and our oldest run had a stuck task
  - **Fix:** `airflow dags delete football_training_pipeline --yes` + `airflow dags reserialize` to clean up

## Results / metrics

- **Stack scale:** 3 containers (Day 3) → 6 containers (Day 4). Total Docker disk usage ~7.9 GB
- **Airflow image build time:** ~9 min (one-time)
- **End-to-end DAG runtime (after fixes):** ~4 seconds when all DVC stages are cached; ~6-8 seconds when training actually executes
- **Tasks per DAG:** 6 (5 happy path + 1 failure-branch alert)
- **Pool slots:** 1 (`training_pool`)
- **Sensor timing:** 30s poke interval, 5min timeout, reschedule mode
- **Retries on every task:** 2, with exponential backoff (60s → 120s → 240s, max 600s)
- **Test Macro F1 (forced retrain on Day 4 follow-up):** 0.4236 (unchanged — same data, same hyperparams)
- **MLflow model registry:** Version 15 of `football-outcome-predictor` — first version with all artifacts on disk after backend mount fix
- **Backend log after final fix:** `Loaded model from MLflow registry` (no fallback)

## What I deferred (and why)

- **Triggerer + Dag Processor containers** — both are optional in Airflow 3.x. Triggerer only needed for async deferrable sensors; we use synchronous FileSensor. Dag Processor is for separating DAG parsing from scheduler at scale. For one DAG, scheduler handles parsing inline. Health panel shows them red — informational only, viva-ready talking point.

- **Real email/Slack alerts** — `dry_pipeline_alert` logs a warning instead of sending. Adding SMTP would mean an extra SMTP container and credentials handling for academic-grade marginal benefit. Pattern is the rubric-relevant part.

- **GitHub Actions / CI** — same reasoning as Day 3. Rubric defines CI as DVC's `dvc repro`, not workflow YAML files. Adding a CI workflow is out of grading scope.

- **Cleaning up old broken model versions in MLflow** — versions 1-14 in the registry point to non-existent artifacts (legacy from earlier debugging). They don't break anything (`latest` alias points to v15) and serve as evidence of debugging history. Cleanup can wait.

- **`force_retrain` config option on the DAG** — would let demo runs bypass DVC caching via `airflow dags trigger ... --conf '{"force_retrain": true}'`. Worth adding if needed for live demo, but the standard production behavior (DVC decides) is the more defensible default.

## Commits

Day 4 main session (4 commits + 1 dvc.lock):
- `3457032` — build(airflow): add Airflow 3.x service stack to docker compose
- `6e5d7bf` — feat(airflow): add football_training_pipeline DAG
- `03dc304` — chore(data): add triggers directory for FileSensor
- `76b3858` — chore: remove unused root dags/ folder
- `074df39` — chore(dvc): update lock file from containerized pipeline run

Day 4 follow-up (2026-04-27, 2 commits):
- `00fe82d` — fix(backend): mount mlflow_artifacts so model registry loads work
- `3ef9d64` — feat(airflow): strengthen validate_data with freshness and size checks

Tag: `v0.4.0` (after the main 5 commits)

## For the viva

**Q: Why did you choose Airflow over Ray, Spark, or a custom Python script?**
A: Airflow is the rubric-aligned answer for orchestration, and it's the only one of the three with first-class scheduling, sensors, and pools — the four things the rubric explicitly tests. Ray and Spark are distributed compute frameworks; they shine on big data. My data is 300 MB, well within pandas' memory limits, so distributed compute would add cluster orchestration overhead with no performance benefit. I'm using Airflow's strengths: scheduling, dependency graphs, retry semantics, and resource pools.

**Q: Walk me through one full DAG run end-to-end.**
A: An upstream data pipeline writes `data/raw/database.sqlite` and signals completion by writing `data/triggers/retrain_trigger.txt`. The FileSensor — running in reschedule mode, polling every 30 seconds — detects the trigger and succeeds. `validate_data` confirms the trigger is fresh (under 24h) and the raw DB is at least 1 MB. `build_features` runs `dvc repro build_features` inside the `training_pool`, which has 1 slot to prevent parallel resource contention. `train_model` runs `dvc repro train` in the same pool — DVC's hash-based caching means the script only re-executes if inputs actually changed. `check_metrics` reads `metrics.json` and fails the DAG if Macro F1 dropped below 0.40, blocking promotion of bad models. If the FileSensor times out instead, `dry_pipeline_alert` fires via `trigger_rule='one_failed'` and logs a warning. Each task retries up to 2 times with exponential backoff before the failure path triggers.

**Q: How does your trigger file pattern scale to production?**
A: It's the same producer/consumer pattern Hadoop and Spark use — a "_SUCCESS" sentinel file written as the upstream's last step. The trigger guarantees data is fully written before downstream consumers see it; if the upstream crashes mid-write, no trigger appears, no DAG runs. In our setup the file is local; in production it'd live on S3 or HDFS, and the FileSensor would be replaced by S3KeySensor or HdfsSensor. The Airflow logic doesn't change — just the sensor type.

**Q: Why does updating the trigger file alone not retrain the model?**
A: Two caching layers. Airflow + FileSensor decide *when* to check. DVC decides *what* to do based on input hashes. The trigger file isn't a DVC dependency — DVC tracks `data/raw/database.sqlite`, `src/training/train.py`, and `params.yaml`. If none of those change, `dvc repro train` reports "Stage 'train' didn't change, skipping" and the script never executes. This is by design: we don't waste compute retraining when the result would be byte-identical. To force a retrain for testing, I run `dvc repro --force train` directly. In production, the upstream pipeline updates both the trigger AND the data file, which makes DVC re-execute naturally.

**Q: What was the hardest bug you fixed today?**
A: The DVC caching one. Every Airflow DAG run for several days reported "all 5 tasks green" but no new model versions appeared in MLflow. The pipeline *looked* successful — green ticks across the board — but the actual training script wasn't running. The clue was that the run's artifact folder only contained the JSON metadata files; no model. Once I read the `train_model` task log carefully, I saw `Stage 'train' didn't change, skipping`. DVC was doing exactly what it should — caching deterministic outputs — but the silent skip masquerading as "success" is dangerous in a CI context. Fix was to run `dvc repro --force` once to register a fresh model, and document this as a viva talking point: airflow runs aren't proof of training; MLflow run history is.

**Q: Why is the Triggerer panel red in your Airflow UI?**
A: Triggerer and Dag Processor are optional Airflow 3.x services. Triggerer handles async deferrable sensors; we use synchronous FileSensor so it's not needed. Dag Processor is a separate DAG-parsing service for large deployments; the scheduler handles parsing inline for our single DAG. Red means "not running" — not "broken." Both would be added containers in a production stack with hundreds of DAGs or many async sensors.