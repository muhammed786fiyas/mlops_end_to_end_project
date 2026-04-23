# Day 3 — DVC Pipeline + MLflow + MLproject + Full Docker Integration
**Date:** 2026-04-22
**Time spent:** ~7 hours
**Rubric mapping:** MLOps Implementation (Source Control & CI, Experiment Tracking, Software Packaging, Data Engineering), Viva demonstrations

## What I shipped today

- **DVC initialization** — `.dvc/`, `.dvcignore`, local remote at `../dvc_remote_football`
- **Raw SQLite versioned** — `data/raw/database.sqlite.dvc` (100-byte pointer) tracks the 299 MB file
- **`dvc.yaml` pipeline** — 2 stages (`build_features`, `train`) with deps/params/outs forming a DAG
- **Pipeline reproducibility proven** — changed `rolling_window_size` (5→6→5) and `learning_rate` (0.03→0.04→0.03); confirmed only affected stages re-run
- **Git tags** — `data-v1` (data snapshot) and `model-baseline` (F1=0.4236 baseline)
- **MLflow tracking integration** — `src/training/train.py` wraps training in `mlflow.start_run()`, logs all params, metrics (train/val/test × F1/AUC/accuracy/per-class), artifacts, and tags
- **MLflow Model Registry** — training registers `football-outcome-predictor` as a new version via `mlflow.lightgbm.log_model(registered_model_name=...)`
- **Descriptive run names** — format: `lgbm-lr<X>-leaves<Y>-window<Z>-<git_hash>` for scannability
- **Dual-mode script** — detects `MLFLOW_RUN_ID` env var, works both standalone (for DVC) and via `mlflow run`
- **`MLproject` file** — declares 3 entry points (`train`, `build_features`, `main`) with typed parameters
- **`conda.yaml`** — environment spec for reproducibility
- **`run_training.sh`** — wrapper passing `--experiment-name football-prediction` to avoid landing in Default experiment
- **FastAPI backend wired to MLflow registry** — `mlflow.lightgbm.load_model("models:/football-outcome-predictor/latest")` with graceful pickle fallback
- **Docker Compose with 3 services** — MLflow + backend + frontend, all on `football-net`, with volume mounts for persistent MLflow state and data

## Key design decisions (viva-ready)

- **DVC remote outside project folder (`../dvc_remote_football`)**
  - Why: mimics production separation of infrastructure from code; prevents Git/DVC from touching the same files

- **DVC as CI (not GitHub Actions)**
  - Rubric explicitly asks "how is CI implemented using DVC?" — answer is `dvc repro` with DAG-based smart re-execution

- **Local MLflow tracking server in Docker**
  - Alternatives: file-based (`./mlruns/`), cloud-hosted
  - Why: need the backend container to load models via HTTP; containerized MLflow mirrors production architecture

- **`mlflow.lightgbm.load_model` vs `mlflow.pyfunc.load_model`**
  - Why: LightGBM-specific loader returns raw classifier with `predict_proba()` intact. pyfunc wrapper loses it.

- **Graceful fallback to pickle if MLflow is down**
  - Why: resilience principle; keeps demo working if any one service fails

- **MLflow container uses `--allowed-hosts "*"` for dev**
  - Alternatives: restrict to specific hostnames
  - Why: MLflow 3.11's security middleware rejected all cross-container requests with custom Host headers; wildcard is acceptable for local dev demo

- **Volume mounts for `mlflow.db`, `mlflow_artifacts/`, `data/processed/`, `models/`**
  - Why: persists state across container restarts; allows host-based training + container-based serving to share data

- **Dual-mode training script (standalone vs `mlflow run`)**
  - Script detects `MLFLOW_RUN_ID` env var; attaches to existing run or creates fresh one
  - Why: DVC's `dvc repro` invokes training standalone; MLproject invocation needs different behavior

- **Slim `requirements.txt` with pip-installable versions only**
  - Why: `pip freeze` produced conda-only versions (e.g., `scikit-learn==1.8.0`) that don't exist on PyPI, breaking Docker builds. Rewrote with version ranges (`>=`) for portability.

- **Always train from inside the backend container**
  - Why: MLflow records artifact paths using whatever filesystem the training process sees. Host-side training records host paths invisible to containers; container-side training records `/mlflow/mlflow_artifacts/` paths visible to both backend and MLflow containers via the shared volume. In production this would be solved with S3 as the artifact store.

## Problems faced & how I solved them

- **Problem:** `dvc repro` errored — "output already tracked by SCM"
  - **Root cause:** Day 2 committed `feature_importance.png` + `training_metadata.json` to Git; DVC claims them as stage outputs
  - **Fix:** `git rm --cached` both; files stay on disk, tracked by DVC going forward

- **Problem:** `mlflow run` errored — "active experiment ID does not match"
  - **Root cause:** `mlflow run` creates run in Default experiment before script starts; `set_experiment()` conflict
  - **Fix:** Detect `MLFLOW_RUN_ID` env var; skip `set_experiment` when present

- **Problem:** Runs landed in "Default" experiment instead of "football-prediction"
  - **Root cause:** MLproject YAML doesn't support `experiment_name` top-level; env var prefix in command runs too late
  - **Fix:** Wrapper script `run_training.sh` passing `--experiment-name` to `mlflow run`

- **Problem:** Docker backend couldn't reach MLflow via `host.docker.internal:5000` — got 403 "Invalid Host header"
  - **Root cause:** MLflow 3.11 added DNS rebinding protection that rejects non-localhost Host headers
  - **Fix:** Added MLflow as a proper Docker Compose service on shared bridge network; container-to-container via `http://mlflow:5000`

- **Problem:** After moving MLflow to Docker, DB migration failed
  - **Root cause:** Host-built `mlflow.db` (MLflow 3.11) had schema newer than older container image (2.18)
  - **Fix:** Matched container MLflow version to host (3.11.1)

- **Problem:** Model artifacts had host-machine paths in `mlflow.db`
  - **Root cause:** Training ran on host before MLflow moved to container; paths baked in
  - **Fix:** Re-registered model from inside the backend container via `docker compose exec backend python -m src.training.train`; new paths relative to container filesystem

- **Problem:** Docker `requirements.txt` build failed with `debugpy==1.8.20 not found`
  - **Root cause:** `pip freeze` dumped every package in the conda env, including non-PyPI versions
  - **Fix:** Hand-curated minimal `requirements.txt` with `>=` constraints

## Results / metrics

- DVC remote: 299 MB raw + ~6 MB pipeline outputs
- MLflow tracking: 13+ runs logged; `football-outcome-predictor` has 13+ versions registered
- Pipeline reproducibility: confirmed via repeated `dvc repro` with same params producing identical outputs (cache hits)
- Full Docker stack: 3 containers orchestrated, MLflow → backend → frontend chain working end-to-end
- Backend model load source: `mlflow:football-outcome-predictor@latest` (visible in every API response)
- F1 still 0.4236 (no regression despite MLOps tooling)

## What I deferred (and why)

- **Full env rebuild demo (`mlflow run` without `--env-manager=local`)** — takes ~5 min; Day 7 viva prep
- **Model promotion workflow (Staging → Production stages)** — using `latest` tag, not stage-based aliases; enough for rubric
- **S3-based artifact store** — would eliminate the host-vs-container path issue but adds complexity (MinIO container); training-via-container discipline achieves the same outcome

## Commits

- `e45703e` — chore(dvc): initialize DVC with local remote
- `9f4776d` — feat(dvc): version raw SQLite database with DVC
- `6b92eda` — feat(dvc): add dvc.yaml pipeline with build_features and train stages
- `82111f2` — feat(mlflow): add experiment tracking and model registry to training
- `904e433` — feat(mlflow): add MLproject + conda.yaml + wrapper script for reproducible runs
- `7a849f8` — feat(docker): add MLflow as 3rd service with proper container networking

**Tags:** `data-v1`, `model-baseline`

## For the viva

**Q: How is CI implemented using DVC?**
A: `dvc.yaml` defines two stages with deps (code + data), params (from `params.yaml`), and outputs. `dvc repro` computes hashes of all inputs and re-runs only stages whose hashes changed. Changing `model.learning_rate` re-runs only the `train` stage; changing `features.rolling_window_size` re-runs both. The pipeline DAG IS the CI definition.

**Q: Walk me through reproducibility end-to-end.**
A: `git checkout data-v1` restores the code + `.dvc` pointer files at that tag. `dvc checkout` syncs the actual data files. `dvc repro` re-runs the pipeline (skipped if cached). `random_state: 42` throughout ensures byte-identical outputs.

**Q: What's in MLflow that's not in DVC?**
A: Experiment-level tracking — I can compare 13+ training runs side-by-side in the UI, filter by `metrics.test_macro_f1 > 0.42`, and see which hyperparameter combos won. DVC tracks pipeline state; MLflow tracks experiment history.

**Q: How does the backend get its model?**
A: At startup, the backend calls `mlflow.lightgbm.load_model("models:/football-outcome-predictor/latest")` against `http://mlflow:5000`. The MLflow container shares the `mlflow_artifacts/` volume, so the backend reads the model artifact directly after getting the URI from MLflow. If MLflow is unreachable, it falls back to a local pickle for resilience.

**Q: Why 3 containers instead of 2?**
A: Loose coupling. MLflow handles experiment tracking + model serving; backend is a stateless prediction service; frontend is static UI. Each can be deployed/scaled independently. In production I'd put MLflow behind a persistent database and on dedicated infrastructure; here it's one compose file for the demo.

**Q: Why must training run inside the backend container?**
A: MLflow records artifact paths using the filesystem of the training process. Host-side training writes host paths that the backend container can't access. Container-side training writes `/mlflow/mlflow_artifacts/...` — visible to both containers via the shared volume. In production I'd use S3 as the artifact store, eliminating this host-vs-container path mismatch entirely.