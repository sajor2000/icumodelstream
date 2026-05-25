# Gap Analysis — CLIF Navigator Status vs Current `icumodelstream` Repository

The attached status artifact describes a substantially more advanced CLIF-Navigator implementation than the repository that was pushed. The repository currently contains a useful **phase-0 skeleton**, but it does **not** contain the core modeling, evaluation, notebook, and TRIPOD+AI functionality described in the status document.

## Executive finding

The user critique is correct. The current repository is too basic relative to the intended target. It provides parquet discovery, simple schema validation, basic QC, adult ICU cohort construction, and a small test suite. The status document expects an implementation with leakage-safe features, LightGBM in-hospital mortality modeling, LSTM sequence modeling, subgroup evaluation, calibration reporting, decision-curve planning, marimo notebooks, and 117 passing tests.

## Side-by-side comparison

| Capability | Status artifact expectation | Current repository state | Gap severity |
|---|---|---|---|
| CLIF parquet inspection | Done | Basic `inspect` command exists | Low |
| QC reporting | Done | Basic JSON table summaries exist | Medium |
| Baseline features | Done, leakage-safe windowed features | Only one generic numeric aggregation helper | High |
| In-hospital mortality label | Done | Not implemented as a first-class module | High |
| Train/validation/test split | Done or implied | Not implemented | High |
| LightGBM baseline | AUROC/Brier reported, reproducible CLI | Not implemented | High |
| LSTM sequence baseline | Head-to-head comparison reported | Not implemented | High |
| Calibration reporting | Reported | Not implemented | High |
| TRIPOD+AI subgroup metrics | `--subgroup-cols` on baseline CLIs | Not implemented | High |
| Decision-curve analysis | Planned 6.b | Not implemented | Medium-high |
| Notebooks | marimo notebooks 01–04 | Only empty notebook folder | High |
| Tests | 117 passing | 5 passing tests | High |
| Manuscript-ready docs | Phase progress and plans | Basic roadmap only | High |

## Correct next revision

The immediate fix should not pretend that the full reported AUROC results exist. Instead, the repository should be upgraded into a credible **phase-1/phase-2 implementation scaffold** that Claude Code can continue from. The next commit should add concrete, tested modules for labels, deterministic patient-level splits, binary metrics, calibration, subgroup metrics, and decision-curve analysis. It should also add CLI hooks that can evaluate saved predictions even before the full LightGBM and LSTM training loops are implemented.

This moves the project from a generic skeleton toward the actual CLIF-Navigator arc while remaining honest: real model performance must be generated from the user's local CLIF-MIMIC parquet data, not fabricated in the repo.

## Recommended implementation order

| Step | Module/area | Why it comes next |
|---|---|---|
| 1 | `labels.py` | Define in-hospital mortality target explicitly. |
| 2 | `splits.py` | Prevent patient leakage before feature work grows. |
| 3 | `metrics.py` | Centralize AUROC, Brier, log loss, calibration. |
| 4 | `subgroups.py` | Enable TRIPOD+AI-style subgroup performance reporting. |
| 5 | `decision_curve.py` | Add clinical utility reporting before manuscript framing. |
| 6 | CLI commands | Make the above runnable on prediction files. |
| 7 | Tests | Raise confidence and establish Claude Code guardrails. |
| 8 | Feature/model modules | Add LightGBM, then sequence baseline after labels and splits are locked. |

## Important boundary

The status document mentions specific empirical results. Those should not be committed unless they are reproduced from the actual local CLIF-MIMIC parquet data and their scripts are included. The safer next commit is therefore an implementation scaffold plus tests, not a fake reproduction of the reported AUROC values.
