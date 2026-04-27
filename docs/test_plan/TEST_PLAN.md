# Test Plan — Football MLOps End-to-End Project

**Document version:** 1.0
**Last updated:** 2026-04-28 (Day 6)
**Project:** DA5402 Final Project — Football Match Outcome Prediction
**Author:** Muhammed Fiyas

---

## 1. Purpose

This document describes the testing strategy for the Football Match Outcome Prediction MLOps system. It enumerates what is tested, what is deliberately not tested and why, the test environment required, and the pass/fail criteria for each test category.

The plan is designed to demonstrate that the system's **critical paths** behave correctly under both happy-path and degraded conditions. It is **not** designed for full coverage — that would be inappropriate scope for a one-week academic project. Instead, we test where errors would be hardest to detect by manual demo: silent data leakage, cold-start edge cases, and model-input contract drift.

---

## 2. Scope

### 2.1 In scope

The following components are covered by automated tests:

- **REST API endpoints** — `GET /health`, `GET /teams`, `POST /predict`, `GET /metrics`
- **FeatureService** (`src/api/feature_service.py`) — runtime feature builder used by `/predict`
- **Build features pipeline** (`src/features/build_features.py`) — DVC stage that produces train/test/production CSVs
- **Training pipeline** (`src/training/train.py`) — model fitting and MLflow logging
- **Model-input contract** — consistency between `FEATURE_COLUMNS` in `train.py` and `_expected_columns()` in `feature_service.py`

### 2.2 Out of scope

The following are deliberately not covered by automated tests in this project:

- **End-to-end browser tests of the frontend** — the UI is 150 lines of vanilla JS with three interactive elements. Visual smoke testing during the demo is sufficient.
- **Airflow DAG execution** — Airflow's own test infrastructure (`airflow dags test`) is the right tool for this; we rely on manual verification via the Airflow UI and confirmed task graph success in Day 4.
- **Prometheus / Grafana / AlertManager configuration** — already verified end-to-end during Day 5 (alert fired, webhook logged). The configs are declarative YAML; their correctness is validated by the services starting cleanly.
- **Performance / load testing** — out of scope for academic-scale single-host deployment. p95 latency was observed at ~75 ms under burst traffic during Day 5; no formal load test was performed.
- **Security testing** — no authentication is implemented (not required by rubric). API exposes no secrets. CORS is open by design for local demo.
- **Database migration testing** — schema is fixed by the Kaggle dataset; no migrations occur.

### 2.3 Why this scope

The rubric grades whether a test plan exists, what it covers, and whether the choices are defensible. We choose **depth over breadth**: each tested area is a place where a silent bug would be costly. We do not chase coverage metrics because high coverage on trivial getter/setter code is misleading.

---

## 3. Test strategy

### 3.1 Test pyramid

```
                    ╱╲
                   ╱  ╲     E2E (manual: demo + Airflow UI)
                  ╱────╲
                 ╱      ╲    Integration (pytest + httpx — API endpoints)
                ╱────────╲
               ╱          ╲   Unit (pytest — feature/training functions)
              ╱────────────╲
```

We follow the standard test pyramid: many fast unit tests, fewer integration tests, manual end-to-end verification.

### 3.2 Test categories

Six categories of test, mapped to specific risk areas:

| Category | What it catches | Layer |
|---|---|---|
| **C1 — Smoke** | Did the service even start? | Integration |
| **C2 — Input validation** | Are bad inputs rejected with proper status codes? | Integration |
| **C3 — Cold-start handling** | Does the system degrade gracefully when data is incomplete? | Unit |
| **C4 — Target derivation correctness** | Are training labels computed correctly from raw scores? | Unit |
| **C5 — Data leakage prevention** | Do the chronological-split guards actually fire? | Unit |
| **C6 — Model-input contract** | Does the runtime feature vector match what the model expects? | Unit |

### 3.3 Frameworks and tools

- **pytest** as the test runner — industry standard for Python testing
- **httpx.AsyncClient** for testing FastAPI endpoints in-process (no separate server needed)
- **pytest fixtures** for shared setup (loaded model, in-memory test database)
- **No mocking framework needed** — most tests run against real (small) data; mocking adds complexity without much value at this scope

---

## 4. Test cases

The following 12 test cases cover the 6 categories above. Each is implemented in `tests/`.

### C1 — Smoke tests (3 cases)

**TC-1.1 — Health endpoint returns 200 with model loaded**
- **Setup:** Backend running, model loaded from MLflow registry or local pickle
- **Action:** `GET /health`
- **Expected:** Status 200, JSON body has `status="ok"`, `model_loaded=True`, `container_id` is non-empty
- **Pass/fail:** All three assertions must hold

**TC-1.2 — Teams endpoint returns 299 teams**
- **Setup:** Backend running with `team_lookup.csv` available
- **Action:** `GET /teams`
- **Expected:** Status 200, body has `count==299`, `teams` is a list of 299 dicts each with `team_api_id`, `name`, `short_name`
- **Pass/fail:** Status code + count + structure of first item

**TC-1.3 — Metrics endpoint exposes Prometheus format**
- **Setup:** Backend running with prometheus-client installed
- **Action:** `GET /metrics`
- **Expected:** Status 200, body contains `football_predictions_total`, `football_prediction_latency_seconds_bucket`, `football_model_info`
- **Pass/fail:** All three metric names must be present in response text

### C2 — Input validation (3 cases)

**TC-2.1 — Unknown home_team_id returns 400**
- **Setup:** Backend running
- **Action:** `POST /predict` with `home_team_id=99999` (deliberately invalid)
- **Expected:** Status 400, `detail` contains "Unknown home_team_id"
- **Pass/fail:** Both assertions

**TC-2.2 — Same team for home and away returns 400**
- **Setup:** Backend running
- **Action:** `POST /predict` with `home_team_id == away_team_id` (e.g., both 8350)
- **Expected:** Status 400, `detail` contains "must differ"
- **Pass/fail:** Both assertions

**TC-2.3 — Malformed JSON body returns 422**
- **Setup:** Backend running
- **Action:** `POST /predict` with missing required field (no `home_team_id`)
- **Expected:** Status 422 (FastAPI's Pydantic validation)
- **Pass/fail:** Status code

### C3 — Cold-start handling (1 case)

**TC-3.1 — Team with insufficient match history returns zero-form features**
- **Setup:** FeatureService initialized with the project's SQLite database
- **Action:** Call `_compute_team_form()` for a team whose first match is the match_date itself (no prior matches)
- **Expected:** Returns `{form_wins: 0, form_draws: 0, form_losses: 0, form_gs_avg: 0.0, form_gc_avg: 0.0}` and logs a warning
- **Pass/fail:** Returned dict matches exactly + warning is logged

### C4 — Target derivation (2 cases)

**TC-4.1 — derive_target produces correct H/D/A distribution**
- **Setup:** A small fixture DataFrame with hand-computed scores: (2-1) → H, (1-1) → D, (0-3) → A
- **Action:** Call `derive_target(fixture_df)`
- **Expected:** `outcome` column is `["H", "D", "A"]`, `outcome_encoded` is `[0, 1, 2]`
- **Pass/fail:** Both columns match exactly

**TC-4.2 — derive_target raises on null outcomes**
- **Setup:** A fixture DataFrame with one row where `home_team_goal` is NaN
- **Action:** Call `derive_target(fixture_df)`
- **Expected:** Raises `ValueError` with message containing "nulls"
- **Pass/fail:** Exception type + message substring

### C5 — Data leakage prevention (2 cases)

**TC-5.1 — chronological_split raises on overlapping seasons**
- **Setup:** A fixture DataFrame with seasons `2008/2009`, `2009/2010`
- **Action:** Call `chronological_split(df, train_seasons=["2008/2009"], test_seasons=["2008/2009"], production_seasons=["2009/2010"])`
- **Expected:** Raises `ValueError` with message containing "leakage"
- **Pass/fail:** Exception type + message substring

**TC-5.2 — chronological_split raises on unknown seasons**
- **Setup:** A fixture DataFrame with seasons `2008/2009`
- **Action:** Call `chronological_split(df, train_seasons=["1999/2000"], test_seasons=["2008/2009"], production_seasons=[])`
- **Expected:** Raises `ValueError` with message containing "Seasons not found"
- **Pass/fail:** Exception type + message substring

### C6 — Model-input contract (1 case)

**TC-6.1 — FeatureService and train.py declare the same feature columns in the same order**
- **Setup:** Import `FEATURE_COLUMNS` from `src.training.train` and instantiate `FeatureService` with real paths
- **Action:** Compare `train.FEATURE_COLUMNS == FeatureService(...)._expected_columns()`
- **Expected:** Lists are equal, in the same order
- **Pass/fail:** List equality

**Why this matters:** if someone adds a feature to `train.py` and forgets to update `feature_service.py`, the model silently receives the wrong feature vector at inference time and produces nonsense predictions with no exception raised. This is the highest-impact silent bug we can defend against; one line of test code prevents it.

---

## 5. Test environment

### 5.1 Local development

```
Conda environment:  football-mlops (Python 3.11)
OS:                 Ubuntu 24
DB path:            data/raw/database.sqlite (DVC-pulled)
Test framework:     pytest 8+, httpx (already present in FastAPI test stack)
```

### 5.2 Running the tests

```bash
# From project root
conda activate football-mlops
pip install pytest httpx pytest-asyncio
pytest tests/ -v
```

### 5.3 Test data

- **Real data tests** (TC-3.1, TC-6.1) use the project's actual `database.sqlite` and processed CSVs. They are deterministic because the data is fixed.
- **Fixture-based tests** (TC-4.1/4.2, TC-5.1/5.2) use small in-memory DataFrames defined in the test file itself. They are fast (<10 ms each) and have no external dependencies.

### 5.4 Continuous integration

Tests are run **manually** on the developer's machine before commits. We do not have a GitHub Actions CI workflow because:
- The project rubric defines "CI" as DVC's pipeline reproducibility (`dvc repro`), not GitHub Actions
- Adding GitHub Actions earns no rubric points and would be additional unmaintained infrastructure for a one-week project

For a production system we would add a GitHub Actions workflow that runs `pytest` on every push to `develop` and blocks merges to `main` on test failure. This is a logical Day 8+ improvement.

---

## 6. Pass/fail criteria

### 6.1 Per-test

A test passes if all assertions hold and no unexpected exceptions are raised. Test output is captured by pytest's verbose mode.

### 6.2 Overall

The test plan is satisfied when:
- 12/12 cases pass
- No skipped tests (other than environmental skips like "DB unavailable")
- Run completes in under 30 seconds

If a test fails, the failure is investigated before the next commit. We do not allow red CI to be merged.

---

## 7. Risks & assumptions

### 7.1 What this test plan does not catch

We do not test:
- **Model accuracy regression** — handled by Airflow's `check_metrics` quality gate (F1 ≥ 0.40), not by pytest
- **Concurrent request safety** — we use `check_same_thread=False` on the SQLite connection but rely on FastAPI/uvicorn's request handling. A real concurrency test would require `pytest-asyncio` with multiple parallel requests against a stress endpoint.
- **Long-running stability** — the service has been observed running for ~5 hours during Day 5 with no memory growth visible in Grafana, but this is observation, not a test.
- **Disk-full / memory-exhaustion** — out of scope for academic.

### 7.2 Assumptions

We assume:
- The Kaggle SQLite dataset is well-formed (no corrupt rows that would break parsing)
- The MLflow tracking server is reachable from the backend container at startup
- Docker Compose's bridge network correctly resolves service names
- The host machine has sufficient resources (>= 2 GB free RAM) to run all 11 containers

If any assumption is violated, tests may fail with confusing errors. Our file-existence checks at module entry points (added Day 5) help diagnose missing-data cases early.

---

## 8. Test execution log

| Date | Test count | Passed | Failed | Notes |
|---|---|---|---|---|
| 2026-04-28 | 14 | 14 | 0 | All 14 tests pass in 2.78s. 12 from plan + 2 anchor tests for valid happy paths. |

This table is updated whenever the test suite is run. A failing test is fixed (or the test corrected if the test itself was wrong) before the next commit.

---

## 9. Future work

When this project is extended past Day 7:

- Add `pytest-cov` and aim for ≥70% coverage on `src/` (excluding generated code)
- Add property-based testing via `hypothesis` for `compute_rolling_form` (random valid match histories should always produce non-negative form counts that sum to the window size)
- Add `mlflow_tracking` integration test that verifies a training run actually registers a new model version
- Add a GitHub Actions workflow running pytest + ruff lint on every push
- Add Airflow DAG tests via `dag.test()` for the full pipeline
- Add k6 / Locust load tests with p95 latency SLO assertions

---