# Day 6 — Documentation, Testing, and Exception Handling Audit

**Date:** 2026-04-28
**Goal:** Convert 5 days of implementation work into rubric-graded design documentation. Add a real test suite. Audit and harden exception handling.

---

## 1. Where I started

After Day 5 wrap, the technical work was complete: 11-container stack running, all 14/14 implementation rubric points covered (DVC, MLflow, Software Packaging, Airflow, Exporter Instrumentation), v0.5.0 tagged on main. What was missing was the **Software Engineering** section of the rubric — the design documents and test plan that grade documentation maturity rather than code.

Day 6 was always going to be writing-heavy. The shift from "build and debug" to "explain and document" is a different mode of work and required deliberately switching gears.

---

## 2. Plan for the day

Six milestones, in this order:

```
M6.1 — Test plan        (write the doc that shapes the tests)
M6.2 — Pytest suite     (implement the cases the plan enumerates)
M6.3 — HLD              (high-level design with C4 diagrams)
M6.4 — LLD              (low-level design with module interfaces)
M6.5 — User manual      (install / operate / troubleshoot)
M6.6 — Day 6 wrap       (this document, Q&A LaTeX, tag v0.6.0, merge)
```

The order is deliberate: writing the test plan first forced enumeration of test cases, which then took 30 minutes to implement. The HLD and LLD reference each other and the test plan, so writing them after testing avoided forward references. The user manual is least technical and so came last.

---

## 3. M6.1 — Test plan

**Deliverable:** `docs/test_plan/TEST_PLAN.md`

Wrote a 9-section IEEE 829-style test plan covering 6 test categories aligned to specific risk areas:

| Category | What it catches | Why it earns its place |
|---|---|---|
| C1 — Smoke | Did the service even start? | First-line sanity check |
| C2 — Input validation | 400/422 paths work | Confirms boundary handling |
| C3 — Cold-start | Graceful degradation | Defends an A2-style design choice |
| C4 — Target derivation | Training labels correct | Silent data bugs are hardest to catch |
| C5 — Leakage prevention | Train/test season overlap | Highest-impact silent failure mode |
| C6 — Model contract | Train/serve column alignment | Single test prevents whole class of drift |

The plan documents 12 cases. We ended up implementing 14 — two extra "happy-path anchor" tests that prove the failure tests aren't always raising. Updated the execution log table to reflect the actual count.

Defended the explicit out-of-scope decisions: no GitHub Actions CI (rubric defines CI as DVC, not Actions); no browser tests of the frontend (3 interactive elements, manual demo sufficient); no load testing (academic single-host scope).

---

## 4. M6.2 — Pytest suite

**Deliverable:** `tests/` with 5 files, 14 cases, all passing in 2.78 seconds.

Decided to test against the **live Docker stack** (httpx → localhost:8000) rather than loading FastAPI in-process. Three reasons: simpler test code, more realistic (tests the actual HTTP path with the actual model), and cleaner viva narrative ("I run pytest while my Docker stack is up"). Documented in the test plan that `docker compose up -d` is a precondition.

File layout:

```
tests/
├── conftest.py              # api_url, http_client, real_team_ids fixtures
├── test_api_smoke.py        # 3 cases (TC-1.1, 1.2, 1.3)
├── test_api_validation.py   # 4 cases (TC-2.1, 2.2, 2.3 + happy-path anchor)
├── test_feature_service.py  # 2 cases (TC-3.1 cold-start, TC-6.1 contract)
└── test_build_features.py   # 5 cases (TC-4.1, 4.2, 5.1, 5.2 + clean-split anchor)
```

The most valuable test by far is `test_feature_columns_match_train_contract` — it asserts that `FeatureService._expected_columns()` equals `train.FEATURE_COLUMNS` in the same order. If someone adds a feature to training and forgets to update the runtime feature builder, the model silently receives the wrong feature vector and produces nonsense predictions with no exception raised. One assertion blocks a whole class of silent bugs that would survive demo testing.

Initial run produced 14/14 passed. Documented this in the test plan's execution log.

---

## 5. M6.3 — High-Level Design

**Deliverable:** `docs/hld/HLD.md`

12-section HLD following IEEE/ISO 42010 architecture description style: purpose & scope, stakeholders & concerns, system context, layered + container architecture, key design decisions with alternatives, data flow, quality attributes, risks & assumptions, stack summary, related documents, glossary, document history.

Three Mermaid diagrams:

1. **System context** — shows 5 actors (End User, Academic Grader, MLOps Engineer, Kaggle dataset, GitHub) connected to the central Football MLOps System
2. **Container view (C4 Level 2)** — 11 services grouped into Serving / Platform / Observability tiers, with edge labels showing protocol and direction
3. **Sequence diagram** — anatomy of a `/predict` request including the asynchronous Prometheus scrape

The sequence diagram explicitly shows the Pydantic-validation `alt` block separating 400 and happy-path branches — useful both as documentation and as visual confirmation of the layered exception handling.

**Key documentation decision:** captured the 10 most important design choices as section 5 entries, each with alternatives considered + rationale. This is where the daily logs paid dividends — the rationale was already written, just needed assembly.

### Diagram theming

Initial colored versions (pastel fills with custom `style` overrides) had readability issues on dark backgrounds: pale yellow with light gray text was nearly unreadable in VS Code's dark preview. Iterated through three theme strategies:

1. `theme: neutral` — better but still washed out on dark
2. `theme: forest` — sequence diagram OK, container diagram wrong palette
3. **Final:** `theme: base` with explicit `themeVariables` forcing primaryTextColor:#000 and per-classDef `color:#000`

Pragmatic resolution: graders view on GitHub.com, not VS Code preview. GitHub's renderer auto-adapts to viewer's theme (light or dark mode). Verified all three diagrams render readably on GitHub. Stopped fighting VS Code.

---

## 6. M6.4 — Low-Level Design

**Deliverable:** `docs/lld/LLD.md`

13-section LLD covering: repository layout (annotated tree), API specification (links to existing api_endpoints.md + Pydantic models + status code contract), 7 module-level interfaces, data schemas (3 SQLite tables + processed CSV columns + team_lookup), configuration (config.yaml + params.yaml field-by-field), internal sequence diagram for the training pipeline, error handling architecture (the layered approach with diagram), observability hooks in code, coding conventions, build & run, 5 extension points, document history.

**Decision on api_endpoints.md:** retained as a separate companion file rather than absorbing into the LLD. The LLD's section 3 references it. Three audiences benefit from separation: API consumers (frontend devs) want the focused endpoint reference, maintainers want the architectural LLD, runtime consumers can use the Swagger UI at `/docs`.

### Two Mermaid issues fixed during LLD

1. **Exception layers diagram** — same gray-on-pastel issue as HLD; fixed with `themeVariables` directive forcing black text + explicit `color:#000` on each classDef.

2. **Airflow DAG diagram** — Mermaid parse error: `got 'LINK_ID'`. Caused by underscore-containing node IDs paired with `<br/>` tags. Fixed by quoting the node labels: `A["data_sensor<br/>FileSensor"]` instead of unquoted form. Mermaid then treats the label as a string literal.

---

## 7. M6.5 — User manual

**Deliverable:** `docs/user_manual/USER_MANUAL.md`

12-section user manual covering quick start (3 commands), prerequisites, full installation walkthrough, the 7 web UIs (table with URLs + credentials), 7 common workflows, configuration overview, day-to-day operations (restart, logs, model rollback), 9 troubleshooting scenarios, a 5-minute timed demo walkthrough, and links to other docs.

The 5-minute demo walkthrough table doubles as a viva script — each 30-second row covers a distinct rubric area with a concrete URL and a one-line talking point. Plan to print this and follow it during the actual viva.

The 9 troubleshooting scenarios were written from real failures encountered across Days 1-5: backend OSError on missing artifact mount (Day 4), Grafana red triangles from datasource UID mismatch (Day 5), Airflow login rejected from regenerated passwords (Day 4), DVC silent skip when nothing changed (Day 4), resolved-alert notification suppression (Day 5). Each scenario includes the exact remediation that worked.

---

## 8. Bonus — Exception handling audit and fix

After M6.4 was committed, paused to ask: "Have I actually used proper exception handling?" Audited the two most-runtime-critical files (`src/api/feature_service.py` and `src/features/build_features.py`).

### What was already good

- Cold-start handling in FeatureService: < 5 prior matches → zero-form features + warning; no FIFA snapshot before match_date → column means + warning. Both paths log + degrade rather than refuse. This is the canonical graceful-degradation pattern.
- Validation errors in `build_features`: `derive_target` raises ValueError on null outcomes; `chronological_split` raises ValueError on overlapping or unknown seasons. Loud failures with helpful messages.
- The `/predict` endpoint already had the layered approach: Pydantic at boundary, business validation with HTTPException(400), outer try/except with logger.exception + 500 + Prometheus error counter.

### Real gap: file-existence checks

Both `feature_service.py.__init__` and `build_features.py.load_raw_data` called `sqlite3.connect()` without first verifying the database file exists. `sqlite3.connect()` silently creates an empty database on a missing path — meaning a missing data file would surface as a confusing "no such table: Team_Attributes" SQL error 50 lines later, instead of a clear "database file missing" error at the entry point.

### The fix

Added explicit `Path.exists()` checks at both entry points, raising `FileNotFoundError` with a remediation hint:

```python
if not Path(db_path).exists():
    raise FileNotFoundError(
        f"SQLite database not found at {db_path}. "
        f"Run `dvc pull` to fetch the raw data, or check config.yaml's data.raw_db path."
    )
```

Verified: backend rebuilt and restarted cleanly; the path *is* present so the new check stays silent in the happy path. If anyone runs the system on a fresh machine without `dvc pull`, they now get a clear error pointing them to the fix.

Committed as `d60d3ea`. Total cost: ~10 minutes including the audit, two file edits, and a backend rebuild.

### Decision: built-in exceptions, no custom hierarchy

The codebase uses `FileNotFoundError`, `ValueError`, `RuntimeError`, and FastAPI's `HTTPException` — no custom exception classes. At ~1,500 lines of Python, custom hierarchies add maintenance burden without proportional benefit. The HTTP boundary already provides the taxonomy callers need (status codes). Documented this design choice as decision 5.10 in the HLD.

This audit produced six concrete viva talking points that I'll add to the Q&A study guide as a new "Exception Handling" subsection.

---

## 9. Where I ended

By end of Day 6:

- ✅ Test plan documented (9 sections, 14 cases)
- ✅ Test suite implemented and passing (14/14 in 2.78 seconds)
- ✅ HLD complete with 3 Mermaid diagrams
- ✅ LLD complete with module-level interfaces and 2 sequence/architecture diagrams
- ✅ User manual complete with troubleshooting and demo script
- ✅ Exception handling audit + 2 files hardened with file-existence checks
- ✅ All design documents render correctly on GitHub
- ✅ All 11 containers still running, all 14 tests still passing
- ✅ This Day 6 daily log written
- ✅ Day 6 Q&A LaTeX section drafted (with new Exception Handling subsection)
- ✅ Develop ahead of main by ~10 commits, ready for v0.6.0 tag and merge

**Rubric coverage at end of Day 6:**

| Section | Status |
|---|---|
| Source Control & CI (DVC) | ✅ Day 3 |
| MLflow tracking | ✅ Day 3 |
| Software Packaging | ✅ Days 2-4 |
| Data Engineering (Airflow) | ✅ Day 4 |
| Exporter Instrumentation | ✅ Day 5 |
| **Software Engineering** (HLD/LLD/test plan/user manual) | **✅ Day 6** |
| Demonstration | Day 7 (recording + viva rehearsal) |
| Viva | Day 7 + actual viva |

All grader-visible deliverables are now in place. Day 7 is recording, polish, and optional ML experiments.

---

## 10. What I learned today

- **Documentation work compounds when daily logs are good.** Every design decision in the HLD was already written somewhere in Days 1-5 logs. Day 6 was assembly, not authorship. This is a strong argument for journaling design decisions in real-time during a build, even when it feels like overhead.
- **Mermaid diagrams require viewer-aware theming.** Three iterations on three diagrams to find a config that renders well on both GitHub light mode (default for graders) and VS Code dark theme (default for me). Lesson: stop optimizing for the editor preview; optimize for where the reader will see it.
- **Tests written after the code is fine, as long as the test plan is rigorous.** The 14 cases I implemented were all enumerated in the test plan in advance. Each took 5–10 minutes. Total time from "no tests" to "14 passing tests + a 9-section IEEE-style plan" was about 90 minutes. This is actually faster than TDD for a project of this scope, because the architecture was stable.
- **The contract test pattern is high-value, low-cost.** `assert FeatureService._expected_columns() == train.FEATURE_COLUMNS` is a single line that prevents an entire class of silent train/serve drift bugs. I'd add this pattern to any future ML project on day one.

---

## 11. What's left for Day 7

- (Optional) ML experiments — ELO features, days-since-last-match, XGBoost vs LightGBM, possibly TabNet or simple MLP. All as MLflow runs to demonstrate the experimentation infrastructure.
- Demo video recording — 5 minutes following the user manual's Section 10 walkthrough.
- Final viva rehearsal using the Q&A study guide.
- README cleanup if needed.

The technical bar is met. Day 7 is polish.