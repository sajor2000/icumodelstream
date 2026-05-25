# Plan — Upgrade `icumodelstream` From Basic Skeleton to CLIF-Navigator Foundation

The first pushed repository was intentionally safe but under-scoped. This plan moves the codebase toward the status artifact without claiming that empirical results have already been reproduced.

## Unit 1 — Evaluation primitives

Add pure-Python/numpy implementations for AUROC, Brier score, log loss, calibration tables, expected calibration error, subgroup metrics, and decision-curve analysis. These should work on saved prediction files, independent of the eventual LightGBM or LSTM implementation.

## Unit 2 — Leakage controls

Add deterministic split utilities and first-class label derivation. Every model later added to the repository should consume these utilities rather than inventing its own split or label logic.

## Unit 3 — CLI surface

Expose `evaluate-predictions` and `decision-curve` commands so that any model can write a prediction file and immediately produce TRIPOD+AI-relevant evaluation outputs.

## Unit 4 — Next modeling sprint

After the local CLIF-MIMIC parquet columns are inspected, implement these modules in order:

| Module | Purpose | Verification |
|---|---|---|
| `outcomes.py` | Lock in-hospital mortality target and censoring rules. | Toy hospitalization fixture. |
| `feature_windows.py` | Leakage-safe first-24h and first-48h feature windows. | Fixture with post-window rows that must be excluded. |
| `baseline.py` | LightGBM or sklearn-compatible baseline. | Synthetic and tiny parquet smoke tests only. |
| `sequence.py` | LSTM baseline after tabular baseline is stable. | CPU-only synthetic dry run. |
| `reports.py` | Markdown/JSON reports for metrics, calibration, subgroups, and DCA. | Snapshot-style tests with toy predictions. |

## Definition of done for the next sprint

The next sprint is complete when `icumodelstream baseline` can produce a prediction file from local CLIF-MIMIC parquet data, and `icumodelstream evaluate-predictions --subgroup-cols sex --subgroup-cols age_band` can generate a JSON report with overall metrics, calibration bins, and subgroup results.
