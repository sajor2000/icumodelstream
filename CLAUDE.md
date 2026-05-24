# CLAUDE.md

This file is the working agreement for Claude Code when building **ICU Model Stream / CLIF-Navigator**. The first phase uses **CLIF-MIMIC parquet data** only. Optimize for correctness, reproducibility, and simple local execution on a Mac before any GPU work.

## Mission

Build a local-first pipeline that reads CLIF-MIMIC parquet tables, validates expected CLIF structure, produces QC reports, creates a reproducible adult ICU cohort, and prepares baseline features. Do not jump to foundation-model training until the data pipeline, tests, and baseline outputs are correct.

## Karpathy-style operating rules

These rules adapt the practical engineering guidance from `multica-ai/andrej-karpathy-skills` to this ICU data project. The bias is toward caution over speed because a small data mistake in ICU modeling can invalidate the whole analysis.

### 1. Think before coding

Do not assume, do not hide uncertainty, and do not silently choose between ambiguous interpretations. Before implementing a nontrivial change, state the assumption, the intended files to touch, and the verification step. If the request could mean multiple things, ask or present the tradeoff before editing.

For this project, common ambiguity includes CLIF table version, column naming, whether an output is patient-level or aggregate-only, whether MIMIC-derived data may be written to disk, and whether a task belongs in local Mac validation or rented-GPU training.

### 2. Simplicity first

Write the minimum code that solves the current phase. Do not add speculative abstractions, plugin systems, background services, model-training frameworks, or broad configurability until a real use case forces them. If a function grows large or confusing, rewrite it into a smaller readable version.

The first implementation should remain boring: discover parquet files, inspect schemas, validate required columns, create QC reports, define cohorts, and create baseline features. Deep learning comes later.

### 3. Surgical changes only

Touch only the files required by the user request. Match the existing style. Do not refactor adjacent code, rename modules, reformat unrelated files, or delete unrelated dead code. If unrelated problems are found, mention them separately rather than fixing them opportunistically.

Every changed line should have a direct reason. If the change creates unused imports, variables, functions, or tests, clean up only the orphaned code created by that change.

### 4. Goal-driven execution

Turn every task into verifiable success criteria. A good implementation plan should read like this: change one thing, run a check, inspect the output, then continue. Weak goals such as “make it work” are not acceptable for this repository.

Examples:

| Request | Better success criterion |
|---|---|
| Add schema validation | Add a toy parquet test where a required column is missing, then ensure the CLI reports the missing table/column clearly. |
| Build cohort logic | Add a fixture with adult, pediatric, ICU, and non-ICU rows, then confirm only eligible ICU hospitalizations are returned. |
| Add baseline features | Add a tiny vitals/labs fixture and verify deterministic per-hospitalization aggregates. |
| Prepare GPU training | Add a synthetic-data dry run and keep it separate from CLIF-MIMIC data paths. |

### 5. Start with tiny data

Use hand-built toy parquet tables in `tests/` before touching real CLIF-MIMIC parquet. The real parquet files should be used only for local inspection and never committed. Tests should be deterministic, fast, and readable by a clinician-engineer reviewing the project.

### 6. Inspect shapes constantly

After every major data transformation, log or print row counts, column counts, key identifiers, and time ranges. Prefer explicit checks over implicit assumptions. If the number of rows changes, the code should make it obvious whether that change was expected.

### 7. Fail loudly on data assumptions

If a required table or column is missing, raise a clear error that names the table, names the required column, shows nearby available columns when possible, and suggests the next inspection command. Do not silently produce partial cohorts or empty features.

### 8. No silent patient leakage

For prediction tasks, separate train/test by patient or hospitalization before feature generation whenever possible. Do not allow future information to enter baseline features. Any label window, observation window, or censoring rule must be explicit in code and docs.

### 9. Baselines before deep learning

Build QC, cohort definitions, descriptive tables, and LightGBM-style baselines before neural models. Do not add foundation-model training, CUDA code, or rented-GPU scripts until the local data pipeline and baseline outputs are reproducible.

### 10. One change at a time

Make small commits with tests. Do not mix behavior changes with broad refactors. When uncertain, write a failing test first, then implement the smallest fix that makes the test pass.

### 11. No data in git

Never commit parquet, CSV extracts, credentials, PHI, generated patient-level reports, model checkpoints trained on restricted data, or screenshots containing patient-level rows. The `.gitignore` file helps, but every commit still needs human review.

## Local machine boundaries

Use the Mac for CLIF parquet inspection, schema validation, QC, cohort construction, feature generation, small model smoke tests, and documentation. Do not attempt real CUDA-only training on the Mac. When GPU training is added later, it should be a separate module with a dry-run mode and a small synthetic-data test.

## Repository map

| Path | Purpose |
|---|---|
| `src/icumodelstream/io.py` | Parquet discovery and lazy table loading. |
| `src/icumodelstream/schema.py` | Expected CLIF table/column contracts and validation results. |
| `src/icumodelstream/qc.py` | Missingness, row counts, basic distributions, and report generation. |
| `src/icumodelstream/cohorts.py` | Reproducible cohort definitions. |
| `src/icumodelstream/features.py` | Baseline feature tables. |
| `src/icumodelstream/cli.py` | Terminal entry points. |
| `notebooks/` | Marimo notebooks that demonstrate the pipeline against local CLIF parquet. No core logic. |
| `tests/` | Toy-data unit tests. No real patient data. |
| `configs/` | Local configs. Keep private configs untracked. |
| `docs/` | Roadmap and design notes. |

## Coding standards

Use Python 3.11+, type hints, `pathlib.Path`, and `polars` lazy scans where possible. Keep public functions documented with short docstrings. Prefer `ruff`, `pytest`, and small fixtures. Avoid notebooks for core logic; notebooks may demonstrate results, but tested code belongs in `src/`.

## Data safety rules

Treat all real ICU data as sensitive. MIMIC data are credentialed and must follow PhysioNet rules. Rush/local hospital CLIF data may contain PHI and must remain in approved institutional environments. Never push data, credentials, `.env` files, patient-level screenshots, or row-level outputs.

**Marimo notebook export trap.** Marimo `.py` notebooks are safe to commit because they store no cell outputs — the patient-level previews you see in the editor live only in your browser. But `marimo export html notebooks/03_cohort.py` (or `ipynb`, or screenshots, or `marimo export public`) bakes those previews into a file that *can* be committed or shared. The repo's `.gitignore` blocks `notebooks/*.html`, `notebooks/*.ipynb`, `notebooks/__marimo__/`, and common figure formats — do not weaken those entries, do not move exports under another directory, and do not paste cell screenshots into PRs or Slack. Every notebook cell that calls `.head()` against real data is a candidate PHI payload the moment it is rendered to a static file.

## Suggested Claude Code build order

1. Run `pytest` and confirm the skeleton is green.
2. Implement table-specific CLIF adapters only after inspecting the real CLIF-MIMIC parquet column names.
3. Add a `docs/data_dictionary_notes.md` file documenting actual table names and columns observed locally.
4. Expand cohort logic for adult ICU stays and expected outcomes.
5. Add baseline features from vitals, labs, respiratory support, and medications.
6. Add a LightGBM baseline only after the cohort and labels are locked.
7. Add GPU training scripts only after local data tests and baseline metrics are reproducible.

## Definition of done for phase 1

Phase 1 is done when a new developer can clone the repo, install dependencies, point the config at CLIF-MIMIC parquet files, run `inspect`, run `qc`, generate an adult ICU cohort CSV, and run the full test suite without access to any private data.
