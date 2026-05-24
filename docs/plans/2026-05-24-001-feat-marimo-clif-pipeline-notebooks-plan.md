---
status: active
plan_type: feat
depth: standard
title: 'feat: Add Marimo notebooks for interactive CLIF pipeline exploration'
marimo-version: 0.23.8
---

# feat: Add Marimo notebooks for interactive CLIF pipeline exploration

**Goal:** Wire marimo into the project venv, close the gap in `features.py` test coverage, and
build four reactive marimo notebooks that let a developer interactively run the full Phase 1–3
pipeline (inspect → QC → cohort → features) against local CLIF-MIMIC parquet files on Mac.

---

## Problem Frame

The `icumodelstream` library is fully implemented and the CLI works. The next productive step
before LightGBM modeling is an interactive loop where the developer can explore the actual
CLIF-MIMIC tables, visually confirm the cohort waterfall, and spot-check feature distributions —
without re-running CLI commands and hunting through JSON reports.

Marimo provides reactive Python notebooks as plain `.py` files, which satisfy CLAUDE.md's
requirement that notebooks be version-controllable and not contain core logic. The analysis logic
stays in `src/icumodelstream/`; notebooks call it and display results.

A secondary gap: `features.py` has no tests. This plan closes that hole alongside the notebook
work because the notebooks will exercise `aggregate_numeric_table` directly, and untested
library code should not be called from interactive work before it is unit-tested.

---

## Requirements Trace

| Requirement | Source |
|---|---|
| Marimo usable inside project venv via `make notebook` | User request |
| Notebooks call `icumodelstream.*`, not duplicate logic | CLAUDE.md coding standards |
| `features.py` has unit test coverage | CLAUDE.md §10 ("make small commits with tests") |
| No PHI, no row-level patient output committed | CLAUDE.md data safety rules |
| Config loaded from `configs/local.yaml` — no hardcoded paths | CLAUDE.md §1 (no hardcoded paths) |
| Notebooks are `.py` files tracked in git | CLAUDE.md + Marimo convention |

---

## Scope

### In scope
- Add `marimo` to `pyproject.toml` dev extras and install it into the project venv
- Add `make notebook` Makefile target
- Add `tests/test_features.py` with unit coverage for `aggregate_numeric_table`
- Four marimo notebooks under `notebooks/`:
  - `01_inspect.py` — table discovery, inventory, schema validation
  - `02_qc.py` — missingness report
  - `03_cohort.py` — adult ICU cohort with row-count waterfall
  - `04_features.py` — vitals/labs baseline aggregate preview

### Deferred to follow-up work
- LightGBM baseline notebook (Phase 4 per roadmap)
- GPU training scripts (Phase 5)
- Export/save of notebook outputs to `reports/`
- CI integration for notebooks (marimo headless run)

### Out of scope
- Modifying any core library logic in `src/icumodelstream/`
- Adding a notebook for GPU/model training
- Publishing notebooks as HTML reports

---

## Context & Research

**Marimo version installed system-wide:** `/Library/Frameworks/Python.framework/Versions/3.13/bin/marimo`
(Python 3.13). Not yet present in the project venv (Python 3.11). Must be added to
`pyproject.toml` and reinstalled.

**Marimo `.py` notebook format:** Each cell is a function decorated with `@app.cell`. Cells
declare outputs by returning values. Reactive: changing an upstream cell re-runs dependents.
Config loading and data loading belong in early cells; display cells depend on them.

**Config integration:** `src/icumodelstream/config.py` has `load_config(path) -> AppConfig`.
Notebooks load `AppConfig` from `configs/local.yaml`, which already points to the correct data
root at `~/Desktop/OneDrive_1_5-24-2026/data`. No hardcoded paths needed.

**Marimo + polars:** `mo.ui.table(df.to_pandas())` or use the native polars DataFrame display
that marimo supports from marimo ≥ 0.9 via `mo.ui.dataframe(df)`.

**Test pattern in this repo:** `tmp_path` fixture, hand-built polars DataFrames written as
parquet, then `discover_tables(tmp_path)` to get table refs. See `tests/test_cohorts.py`.

---

## Key Technical Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Marimo added to which extra? | `dev` | Keeps a single install command (`pip install -e '.[dev,ml]'`); notebooks are development artifacts |
| Version pin | `marimo>=0.9` | 0.9+ has native polars DataFrame display |
| Notebook cell structure | Config cell → data cell → analysis cells → display cells | Enforces dependency order; changing config path re-runs everything downstream |
| PHI guard | `assert config.safety.allow_phi is False` in notebooks | Fail loudly if safety config is misconfigured before any data loads |
| `test_features.py` placement | `tests/test_features.py` | Matches existing test file naming |

---

## Open Questions

None blocking. All deferred items are sequencing decisions, not blockers.

---

## Output Structure

```
notebooks/
├── 01_inspect.py        # Table inventory + schema validation
├── 02_qc.py             # Missingness report
├── 03_cohort.py         # Adult ICU cohort with waterfall
└── 04_features.py       # Vitals/labs baseline feature preview

tests/
└── test_features.py     # NEW — unit tests for features.py
```

---

## Implementation Units

- U1. **Add marimo to pyproject.toml and Makefile**

  **Goal:** Make marimo installable via the project venv and launchable with `make notebook`.

  **Requirements:** Marimo usable inside project venv.

  **Dependencies:** None.

  **Files:**
  - `pyproject.toml` (modify — add `marimo>=0.9` to `[project.optional-dependencies][dev]`)
  - `Makefile` (modify — add `notebook` target)

  **Approach:** Add `marimo>=0.9` to the `dev` list in `pyproject.toml`. Add a `make notebook`
  target that takes a `NOTEBOOK` variable and runs `$(VENV)/bin/marimo edit $(NOTEBOOK)`.
  Example: `make notebook NOTEBOOK=notebooks/01_inspect.py`. Add `$(VENV) = .venv` at top of
  Makefile. The developer reinstalls with `pip install -e '.[dev,ml]'`.

  **Test scenarios:**
  - After `pip install -e '.[dev,ml]'`, `.venv/bin/marimo --version` exits 0
  - `make notebook NOTEBOOK=notebooks/01_inspect.py` launches the editor without error (manual
    verification only; not automated)

  **Verification:** `.venv/bin/marimo --version` succeeds.

---

- U2. **Add tests/test_features.py**

  **Goal:** Unit-test `aggregate_numeric_table` with toy parquet data.

  **Requirements:** `features.py` has unit test coverage.

  **Dependencies:** None (independent of U1).

  **Files:**
  - `tests/test_features.py` (new)

  **Approach:** Follow the `tmp_path` + hand-built polars DataFrame pattern from
  `tests/test_cohorts.py`. Create a toy `vitals.parquet` with `hospitalization_id` and `value`
  columns. Call `aggregate_numeric_table` and assert on mean/min/max/n.

  **Test scenarios:**
  - Happy path: two hospitalizations, known values → assert mean/min/max/n are correct for each
  - Cohort filter: pass a cohort DataFrame with only one hospitalization_id → assert only that
    hospitalization appears in output (left join behavior)
  - Missing value_col: table has `hospitalization_id` but no recognized value column → assert
    `ValueError` is raised with a message containing the table name
  - Missing hospitalization_id: table has `value` but no `hospitalization_id` → assert
    `ValueError` is raised

  **Verification:** `pytest tests/test_features.py -v` passes all four scenarios.

---

- U3. **Create notebooks/01_inspect.py (table inventory + schema validation)**

  **Goal:** Reactive marimo notebook that loads config, discovers tables, shows inventory and
  schema validation results.

  **Requirements:** No hardcoded paths; calls `icumodelstream.*`; safety guard.

  **Dependencies:** U1 (marimo installed).

  **Files:**
  - `notebooks/01_inspect.py` (new)

  **Approach:**
  - Cell 1: imports and `mo.md("# CLIF Table Inspector")`
  - Cell 2: `config = load_config("configs/local.yaml")` + safety assert
  - Cell 3: `tables = discover_tables(config.data.root, config.data.table_glob)` → display
    `table_inventory(tables)` as a marimo table
  - Cell 4: `results = validate_table_contracts(tables)` → display
    `validation_results_to_frame(results)` with a summary pass/fail count

  Paths in the notebook are relative to the repo root (`configs/local.yaml`). Developer runs
  marimo from the repo root.

  **Test scenarios:**
  - Test expectation: none — notebook is a display artifact; the underlying functions have
    unit tests in `tests/test_schema.py` and `tests/test_io.py`. Notebook correctness is
    verified by running it against the real data.

  **Verification:** `marimo run notebooks/01_inspect.py` (or `make notebook NOTEBOOK=...`)
  displays the table inventory and validation table without errors.

---

- U4. **Create notebooks/02_qc.py (missingness report)**

  **Goal:** Reactive notebook that shows row counts and per-column null counts for all
  discovered tables.

  **Requirements:** No hardcoded paths; calls `icumodelstream.qc`.

  **Dependencies:** U1.

  **Files:**
  - `notebooks/02_qc.py` (new)

  **Approach:**
  - Config cell (same pattern as U3)
  - Tables cell
  - QC report cell: `report = build_qc_report(tables)` → iterate `report["tables"]` and
    display each as `mo.ui.table()` with columns `column`, `n_null`, `pct_null`
  - Summary cell: total row counts across all tables as a single aggregate table

  **Test scenarios:**
  - Test expectation: none — underlying `qc.py` functions are tested in `tests/test_qc.py`.

  **Verification:** Notebook renders row counts and missingness for each CLIF table.

---

- U5. **Create notebooks/03_cohort.py (adult ICU cohort with waterfall)**

  **Goal:** Reactive notebook that runs the adult ICU cohort and shows a row-count waterfall
  (all hospitalizations → age filter → ICU location filter → final cohort).

  **Requirements:** No PHI; no cohort CSV written unless user triggers; calls
  `icumodelstream.cohorts`.

  **Dependencies:** U1.

  **Files:**
  - `notebooks/03_cohort.py` (new)

  **Approach:**
  - Config cell
  - Tables cell
  - Intermediate counts cell: count hospitalizations before each filter step by calling
    `scan_table(tables, "hospitalization").collect().height` for total, then applying each
    filter independently to compute waterfall counts
  - Cohort cell: `cohort = build_adult_icu_cohort(tables, CohortSpec(...))`
  - Waterfall display cell: `mo.md(...)` table with steps and counts
  - Preview cell: `mo.ui.table(cohort.head(20).to_pandas())` for spot-check — note explicitly
    that this is non-PHI because it shows only `patient_id`, `hospitalization_id`, and age

  `patient_id` and `hospitalization_id` from MIMIC-IV are de-identified integers per PhysioNet
  rules, but the safety config `allow_phi = false` is still asserted in the config cell.

  **Test scenarios:**
  - Test expectation: none — `build_adult_icu_cohort` is tested in `tests/test_cohorts.py`.

  **Verification:** Waterfall table shows non-zero counts at each step; final count matches
  `cohort.height`.

---

- U6. **Create notebooks/04_features.py (vitals/labs baseline feature preview)**

  **Goal:** Reactive notebook that computes per-hospitalization vitals and labs aggregates for
  the cohort and displays the joined feature table.

  **Requirements:** Calls `icumodelstream.features`; cohort-filtered output only.

  **Dependencies:** U1, U2 (test_features.py should pass before this notebook is built).

  **Files:**
  - `notebooks/04_features.py` (new)

  **Approach:**
  - Config cell
  - Tables cell
  - Cohort cell: `cohort = build_adult_icu_cohort(tables, ...)`
  - Vitals features cell: `vitals_features = aggregate_numeric_table(tables, "vitals", "vitals",
    cohort=cohort)` — wrapped in a try/except that shows a warning if "vitals" table is absent
  - Labs features cell: same pattern for "labs"
  - Join cell: join vitals and labs features on `hospitalization_id`
  - Display cell: `mo.ui.table(features_df.head(20).to_pandas())`

  **Test scenarios:**
  - Test expectation: none — `aggregate_numeric_table` is tested in `tests/test_features.py`
    (U2). Notebook exercises the real data path.

  **Verification:** Feature table renders with `*_mean`, `*_min`, `*_max`, `*_n` columns for
  each available table.

---

## System-Wide Impact

| Surface | Impact |
|---|---|
| `pyproject.toml` | Adds `marimo>=0.9` to `dev` extras |
| `Makefile` | Adds `VENV` variable and `notebook` target |
| `tests/` | Adds `test_features.py` |
| `notebooks/` | Adds 4 new `.py` marimo notebooks |
| `src/icumodelstream/` | **No changes** |

---

## Risks & Dependencies

| Risk | Mitigation |
|---|---|
| Marimo version incompatibility with Python 3.11 venv | Pin `marimo>=0.9`; verify with `marimo --version` in venv after install |
| CLIF-MIMIC parquet tables use different column names than expected | Notebooks rely on the tolerant `first_existing_column` logic already in `cohorts.py` and `features.py`; missing columns produce clear errors |
| Notebook writes patient-level rows to disk accidentally | Safety assert in every config cell; PHI guard is explicit |
| Real data root path wrong in `configs/local.yaml` | Config cell shows `config.data.root` so developer sees it immediately |

---

## Sources & References

- `src/icumodelstream/` — existing library
- `configs/local.yaml` — data root config
- `tests/test_cohorts.py` — fixture pattern to follow for U2
- [Marimo docs: cell structure](https://docs.marimo.io) — reactive cell model
- `docs/roadmap.md` — confirms this work spans Phases 1–3