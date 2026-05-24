# CLAUDE.md

This file is the working agreement for Claude Code when building **ICU Model Stream / CLIF-Navigator**. The first phase uses **CLIF-MIMIC parquet data** only. Optimize for correctness, reproducibility, and simple local execution on a Mac before any GPU work.

## Mission

Build a local-first pipeline that reads CLIF-MIMIC parquet tables, validates expected CLIF structure, produces QC reports, creates a reproducible adult ICU cohort, and prepares baseline features. Do not jump to foundation-model training until the data pipeline, tests, and baseline outputs are correct.

## Karpathy-inspired engineering rules

The project should follow a practical, small-surface-area style often associated with strong research engineering: make the code obvious, keep state explicit, test the small pieces, and avoid clever abstractions until the simple version works.

1. **Keep it simple.** Prefer one readable file with clear functions over a premature framework. If a function needs a long explanation, rewrite it.
2. **Make every step runnable.** Each major pipeline step should have a CLI command and a small test.
3. **Start with tiny data.** Use hand-built toy tables in tests before touching real CLIF-MIMIC parquet.
4. **Inspect shapes constantly.** Print or log row counts, column counts, key IDs, and time ranges after every major transformation.
5. **Fail loudly on data assumptions.** If a required column is missing, raise a clear error that names the table and candidate fix.
6. **Prefer deterministic outputs.** Sort by stable identifiers before writing cohorts or features. Avoid randomness unless a seed is explicit.
7. **Do not hide complexity.** If CLIF tables vary by version, write explicit adapters and document them.
8. **No silent patient leakage.** For prediction tasks, separate train/test by patient or hospitalization before feature generation whenever possible.
9. **Baselines before deep learning.** Build TableOne/QC and LightGBM-style baselines before neural models.
10. **One change at a time.** Make small commits with tests. Do not refactor and change behavior in the same commit.
11. **No data in git.** Never commit parquet, CSV extracts, credentials, PHI, or generated model checkpoints.
12. **When uncertain, write a failing test first.** Capture the expected behavior, then implement the smallest fix.

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
| `tests/` | Toy-data unit tests. No real patient data. |
| `configs/` | Local configs. Keep private configs untracked. |
| `docs/` | Roadmap and design notes. |

## Coding standards

Use Python 3.11+, type hints, `pathlib.Path`, and `polars` lazy scans where possible. Keep public functions documented with short docstrings. Prefer `ruff`, `pytest`, and small fixtures. Avoid notebooks for core logic; notebooks may demonstrate results, but tested code belongs in `src/`.

## Data safety rules

Treat all real ICU data as sensitive. MIMIC data are credentialed and must follow PhysioNet rules. Rush/local hospital CLIF data may contain PHI and must remain in approved institutional environments. Never push data, credentials, `.env` files, patient-level screenshots, or row-level outputs.

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
