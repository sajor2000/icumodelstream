---
title: "feat: TRIPOD+AI subgroup performance analysis"
status: active
created: 2026-05-25
plan_type: feat
depth: lightweight
---

# feat: TRIPOD+AI subgroup performance analysis

**Goal:** Add per-subgroup AUROC / AUPRC / Brier / calibration metrics to both the
LightGBM (`baseline`) and LSTM (`sequence-baseline`) CLI commands so deployment
audits and publication tables can answer "does this model fail any protected
subgroup?" — the question TRIPOD+AI requires and current ICU mortality papers
routinely skip. The subgroup variables — sex, race, ethnicity, age band, ICU type —
are already in the CLIF tables (per `docs/data_dictionary_notes.md`); the
infrastructure to slice and report on them is the missing piece.

This is the first of three publication-track follow-ups. Decision-curve analysis
and CLIF multi-site external validation are separate plans.

---

## Problem Frame

The LightGBM baseline reports a single global AUROC=0.866 and a single global
calibration intercept/slope. That answers "does the model discriminate on the
test set?" — but not "does the model discriminate equally well for women vs men,
Black vs white patients, 80+ vs 40-65 patients, MICU vs SICU?" TRIPOD+AI and the
FDA AI/ML guidance both name subgroup performance as a deployment gate; in
practice, most published ICU mortality models don't report it, and that gap is
itself a publication opportunity.

The infrastructure to do this is mostly already built:

- `BaselineResult` and `SequenceResult` both carry `.y_true` and `.y_pred_proba`
  for the test set (arrays of length `n_test`).
- `models.compute_metrics()` and `models.calibration_table()` are now public
  (commit `9a73b0c`), so per-subgroup metrics reuse exactly the metrics the
  whole-cohort baseline reports.
- The CLIF tables already carry every subgroup variable: `patient.sex_category`,
  `patient.race_category`, `patient.ethnicity_category`,
  `hospitalization.age_at_admission`, `adt.location_category`.

What's missing:

1. A pure function `compute_subgroup_metrics(y_true, y_pred_proba, groups)` that
   returns a tidy per-group table. Should reuse `compute_metrics()` so a
   per-subgroup row looks identical to the existing whole-cohort row.
2. A helper to attach per-row subgroup labels to the test set — neither
   `BaselinePipelineResult` nor `SequencePipelineResult` currently exposes the
   test-set `hospitalization_ids`, so the post-train identity of each test row is
   not recoverable today.
3. CLI integration: a `--subgroup-cols` flag on both baseline commands that
   triggers the analysis and adds a `subgroups` block to the JSON payload + a
   `## Subgroup performance` section to the markdown summary.

Bias is toward **boring** per CLAUDE.md §2: reuse existing primitives, don't
introduce a new analysis framework, age bands are documented constants not a
configurable DSL.

---

## Requirements Trace

| Requirement | Source |
|---|---|
| TRIPOD+AI compliant per-subgroup AUROC / AUPRC / Brier / calibration | TRIPOD+AI checklist; FDA AI/ML guidance |
| Compatible with both `BaselineResult` and `SequenceResult` | User constraint: "reusable module" |
| Subgroup variables: sex, race, ethnicity, age band, ICU type | Already in CLIF (`docs/data_dictionary_notes.md`) |
| Single-class subgroups (e.g., all-survivor age band) handled gracefully | Existing `compute_metrics` returns NaN AUROC for single-class — preserve that |
| No PHI in outputs | CLAUDE.md data safety rules; existing JSON / MD writers already drop per-row data |
| Mac-runnable, no GPU required | Subgroup analysis is post-hoc numpy ops |
| Small commits with tests (CLAUDE.md §10) | 3 implementation units, each <80 LOC |
| Existing CLI patterns (typer flags, `_baseline_result_to_dict`, Rich output) | Mirror `baseline` / `sequence-baseline` shape |

---

## Scope

### In scope
- New module `src/icumodelstream/subgroups.py` with two pure functions:
  - `compute_subgroup_metrics(y_true, y_pred_proba, groups) -> pl.DataFrame`
  - `assign_age_band(ages, bins=DEFAULT_AGE_BINS) -> np.ndarray[str]`
- Extend `BaselinePipelineResult` and `SequencePipelineResult` to expose
  `test_hospitalization_ids: list[str]` so subgroup labels can be aligned.
- New helper in `pipeline.py`: `extract_subgroup_labels(tables, hospitalization_ids,
  subgroup_cols)` that joins the test hospitalization_ids to patient + hospitalization
  + adt and returns a per-row label DataFrame.
- CLI integration: `--subgroup-cols` flag on both `baseline` and `sequence-baseline`.
  When set, the JSON payload gains a `subgroups` array and the markdown summary
  gains a `## Subgroup performance` section.
- Tests with toy CLIF fixtures: unit tests for the pure functions and one
  end-to-end CLI smoke test verifying the subgroup section appears.

### Deferred to follow-up work
- Decision-curve analysis (separate plan; uses the same y_true / y_pred_proba)
- CLIF multi-site external validation (separate plan; uses the same pipeline
  against a different data root)
- Fairness mitigation (re-weighting, post-hoc calibration per subgroup) — analysis
  first, mitigation only if findings warrant it
- Temporal calibration drift analysis (needs multi-year CLIF data)
- Subgroup-stratified calibration plots (matplotlib image artifacts; current
  reliability-table format is sufficient for the first pass)

### Out of scope
- Modifying `BaselineResult` / `SequenceResult` schemas (the metrics they expose
  are reused as-is; only the pipeline result types get the new `test_hospitalization_ids`)
- Adding new outcome variables — works against existing `mortality` / `los_gt_7d`
- A standalone `subgroups` CLI subcommand that reads pre-saved metrics — the
  inline flag is simpler today; promote to a subcommand if a second consumer appears

---

## Context & Research

**Existing patterns to mirror:**

- `src/icumodelstream/models.py`: `compute_metrics(y_true, y_pred_proba)` returns
  a dict with `auroc / auprc / brier_score / prevalence / calibration_intercept /
  calibration_slope`. `calibration_table(y_true, y_pred_proba)` returns a
  10-bin polars DataFrame. Both handle single-class `y_true` by returning NaN
  for AUROC/AUPRC and computing what it can. Subgroup metrics REUSE these
  functions verbatim — one call per (variable, value) pair.
- `src/icumodelstream/pipeline.py` `_run_sequence_baseline` and
  `run_baseline_pipeline`: both build a test-set partition via
  `group_train_test_split`. The latter currently discards the test
  hospitalization_ids; the former exposes them via `SplitTensors.test_indices`
  (commit `644b172`).
- `src/icumodelstream/cli.py` `_baseline_result_to_dict`: drops per-row arrays
  from the JSON payload. Subgroup metrics are AGGREGATES per (variable, value)
  pair — they retain the same data-safety property (no patient-level rows).
- `tests/test_models.py`, `tests/test_pipeline.py`: `tmp_path` + hand-built
  polars DataFrames + `discover_tables` pattern. Subgroup tests follow the
  same shape.

**Observed CLIF subgroup vocabularies** (from `docs/data_dictionary_notes.md`):

- `patient.sex_category`: typically `{"Male", "Female", "Unknown"}`
- `patient.race_category`: e.g., `{"White", "Black", "Hispanic", "Asian", ...}`
- `patient.ethnicity_category`: `{"Hispanic", "Non-Hispanic", "Unknown"}`
- `hospitalization.age_at_admission`: Int64, range 18+ for the existing
  adult-ICU cohort
- `adt.location_category`: includes various ICU subtypes (MICU, SICU, NICU,
  Cardiac ICU, etc.) — the cohort filter already keeps only ICU stays

**Default age bands:** `<40`, `40-65`, `65-80`, `80+`. Conventional in ICU
mortality literature; clinically meaningful (organ-system reserve, frailty,
top-coded ages in MIMIC). Configurable via a constant tuple, not a runtime arg.

---

## Key Technical Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Subgroup metric reuse | `compute_metrics()` from `models.py` for every (variable, value) pair | Apples-to-apples with whole-cohort metrics; one place to fix if metrics evolve. |
| Output shape | Tidy long-format DataFrame: rows = (subgroup_var, subgroup_value); columns = n, prevalence, auroc, auprc, brier, calib_intercept, calib_slope | JSON-friendly. Easy to filter for publication tables. |
| Test-row identity | Add `test_hospitalization_ids` to both pipeline result types | The minimal extension; clean across both pipelines; downstream callers (subgroup analysis, future per-row export, future decision-curve analysis) all need it |
| Age band binning | `DEFAULT_AGE_BINS = ((0, 40), (40, 65), (65, 80), (80, 200))` with labels `["<40", "40-65", "65-80", "80+"]`; right-open intervals | Conventional ICU bins; documented as a constant so a reviewer can change it; right-open matches numpy semantics |
| Single-class subgroup handling | Inherit `compute_metrics`'s NaN-on-single-class behavior; emit a `warning` field in the JSON row | Doesn't crash; surfaces the issue; reuses existing contract |
| Missing / null subgroup values | Treat as a separate `"Unknown"` bucket per variable | Don't silently drop; the count and metrics for "Unknown" are themselves publishable findings |
| CLI flag shape | `--subgroup-cols sex,race_category,age_band,icu_type` (comma-separated) with sensible default | Mirrors `--feature-set`'s string-flag pattern; default opt-in keeps the existing CLI behavior unchanged |
| ICU type derivation | Use the FIRST `adt.location_category` value in the cohort window per hospitalization (the admitting ICU) | Some patients move between ICUs; admitting ICU is the documented convention in ICU outcome literature |
| Subgroup minimum n | No hard floor; the JSON row reports `n` and metrics; downstream readers (the publication table) decide what subgroups to suppress | Don't bury subgroups silently — visibility is the point |

---

## Open Questions

None blocking. Both `--subgroup-cols` defaults and the age-band tuple are configurable
constants; the user can pick different values for follow-up runs without redesign.

---

## Implementation Units

- U1. **`src/icumodelstream/subgroups.py` — pure subgroup metrics function**

  **Goal:** Pure-numpy / polars module with two public functions:
  `compute_subgroup_metrics()` and `assign_age_band()`. No CLIF dependency.

  **Requirements:** TRIPOD+AI subgroup metrics; reuse `compute_metrics()`.

  **Dependencies:** None (`models.compute_metrics` already public).

  **Files:**
  - `src/icumodelstream/subgroups.py` (new)
  - `tests/test_subgroups.py` (new)

  **Approach:**

  - `DEFAULT_AGE_BINS` and `DEFAULT_AGE_LABELS` as module-level constants.
  - `assign_age_band(ages: np.ndarray, bins=..., labels=...) -> np.ndarray[str]`:
    digitize ages into bins using `np.searchsorted` or `pd.cut`; map to labels.
    Out-of-range or null ages → `"Unknown"`.
  - `compute_subgroup_metrics(y_true, y_pred_proba, groups: dict[str, np.ndarray]) ->
    pl.DataFrame`:
    - `groups` is `{variable_name: array_of_subgroup_labels}` — each array has
      length == `len(y_true)`.
    - For each variable, iterate unique values; for each value, slice y_true and
      y_pred_proba to that subgroup; call `compute_metrics` from `models.py`.
    - Build a tidy DataFrame with columns: `subgroup_var, subgroup_value, n,
      prevalence, auroc, auprc, brier_score, calibration_intercept,
      calibration_slope, warning` (warning = `"single_class_y_true"` when AUROC
      came back NaN; null otherwise).
    - Sort by `subgroup_var`, then by `n` descending.

  **Patterns to follow:** `models.compute_metrics` for the per-subgroup metric
  call. `cohorts.first_existing_column` for the tolerant-candidate idiom (not
  needed here — subgroups module is column-agnostic).

  **Test scenarios:**
  - Happy path: 100 rows, 2 variables (sex with 2 values, age_band with 3
    values). Result has 5 rows total. Each row's `n` sums to the global count
    per variable. Each row's `auroc` matches a hand-computed value from sklearn.
  - Single-class subgroup: one age band has all `y_true == 0`. That row's AUROC
    is NaN, `warning == "single_class_y_true"`, but `n` and `prevalence` are
    populated.
  - Missing / null subgroup labels: `groups["sex"]` has 10 nulls in 100 rows.
    Output includes an `"Unknown"` bucket with n=10.
  - Empty groups dict: returns an empty DataFrame with the expected column
    schema (so downstream code doesn't crash on no-flag invocations).
  - `assign_age_band` happy path: ages `[25, 50, 70, 85]` → `["<40", "40-65",
    "65-80", "80+"]`.
  - `assign_age_band` with null and out-of-range: `[None, -5, 200]` →
    `["Unknown", "Unknown", "Unknown"]` (or `"80+"` for 200 if the top bin is
    open-ended — pick one and pin via test).

  **Verification:** `pytest tests/test_subgroups.py -v` passes; coverage
  includes all six scenarios.

---

- U2. **Pipeline support: expose test_hospitalization_ids + subgroup-label extractor**

  **Goal:** Make the test-row identity recoverable, and provide a helper that
  attaches subgroup labels to a list of test hospitalization_ids.

  **Requirements:** Test rows need identity for subgroup attachment.

  **Dependencies:** U1 (the column shape the helper produces feeds the U1 function).

  **Files:**
  - `src/icumodelstream/pipeline.py` (modify — add `test_hospitalization_ids`
    to both `BaselinePipelineResult` and `SequencePipelineResult`; add
    `extract_subgroup_labels()`)
  - `tests/test_pipeline.py` (modify — add test for `extract_subgroup_labels`
    + assert `test_hospitalization_ids` field populated correctly)

  **Approach:**

  - Add `test_hospitalization_ids: list[str]` to both `BaselinePipelineResult`
    and `SequencePipelineResult`. Both pipelines run `group_train_test_split`
    internally; capture the ids of the test rows at that point. (For sequence
    pipeline, `SplitTensors.test_indices` is already present — combine with
    `sequences.hospitalization_ids` to recover the ids. For LightGBM pipeline,
    the cohort + label join is already on hospitalization_id — the split's
    test rows can be extracted from the underlying DataFrame.)
  - New function `extract_subgroup_labels(tables, hospitalization_ids: list[str],
    subgroup_cols: list[str]) -> pl.DataFrame`:
    - Required columns (`hospitalization_id`) + one column per requested subgroup.
    - For each requested col, do the right join:
      - `sex` ← `patient.sex_category`
      - `race_category` ← `patient.race_category`
      - `ethnicity` ← `patient.ethnicity_category`
      - `age_band` ← computed from `hospitalization.age_at_admission` via
        `subgroups.assign_age_band`
      - `icu_type` ← first `adt.location_category` per hospitalization (sort by
        `adt.in_dttm`, take first)
    - Returns a DataFrame with one row per hospitalization_id in the input list,
      preserving order.
    - Null subgroup values stay null (the consumer turns them into "Unknown"
      via U1's logic).

  **Patterns to follow:** `extract_mortality_labels` in `labels.py` for the
  CLIF-table-join pattern. `cohorts.first_existing_column` if any subgroup
  column is missing (the conservative fail-loud path is to raise, but for
  publication-table use a missing column should produce a warning + an
  "Unknown" column rather than crash).

  **Test scenarios:**
  - Happy path: toy CLIF with patient + hospitalization + adt; request all 5
    subgroup cols; output has 5 columns + hospitalization_id; values match the
    toy fixture.
  - Order preservation: input hospitalization_ids = ["H003", "H001", "H002"] →
    output rows in that order.
  - Missing patient row: a hospitalization_id with no matching patient → that
    row's `sex` is null (Polars left-join semantics); test verifies behavior.
  - Missing adt row: a hospitalization with no ICU stay → `icu_type` is null.
  - Age band uses `assign_age_band` from U1: ages of 25, 50, 75, 85 produce
    the expected band labels.
  - `BaselinePipelineResult.test_hospitalization_ids` populated end-to-end:
    after `run_baseline_pipeline` on toy data, `result.test_hospitalization_ids`
    matches the hospitalization_ids of rows in `result.lightgbm.y_true` /
    `result.lightgbm.y_pred_proba`.

  **Verification:** `pytest tests/test_pipeline.py -v` passes.

---

- U3. **CLI integration: `--subgroup-cols` flag + JSON / MD subgroup section**

  **Goal:** Wire U1 + U2 into both `baseline` and `sequence-baseline` so a
  single CLI invocation produces whole-cohort metrics + subgroup metrics.

  **Requirements:** All in-scope items.

  **Dependencies:** U1, U2.

  **Files:**
  - `src/icumodelstream/cli.py` (modify — add `--subgroup-cols` to both
    commands; new private helper `_render_subgroup_markdown()`; payload
    builders gain a `subgroups` block)
  - `tests/test_cli.py` (modify — add one smoke test for sequence-baseline +
    subgroup analysis on the existing toy CLIF fixture)

  **Approach:**

  - Add `subgroup_cols: str = typer.Option("", help="Comma-separated subgroup
    column list. Empty (default) skips analysis. Supported: sex, race_category,
    ethnicity, age_band, icu_type.")` to both `baseline` and
    `sequence-baseline`.
  - When the flag is set, after `_run_sequence_baseline` / `run_baseline_pipeline`
    returns:
    1. Parse the comma-separated list; validate against the supported set
       (raise `typer.BadParameter` on unknown cols).
    2. Call `extract_subgroup_labels(tables, result.test_hospitalization_ids,
       subgroup_cols)` to get the per-test-row labels.
    3. Build a `groups` dict (mapping variable name → numpy array of labels);
       call `compute_subgroup_metrics(y_true, y_pred_proba, groups)`.
    4. Attach the resulting DataFrame to the JSON payload via
       `payload["subgroups"] = subgroup_df.to_dicts()`.
    5. Add a `## Subgroup performance` section to the markdown summary with a
       table per subgroup variable. Mirror the `_render_calibration_md`
       structure for consistency.
    6. Print a brief Rich summary to the terminal: one row per subgroup variable
       showing the min and max AUROC across that variable's values, with the
       value names so a clinician can see at a glance if any subgroup is
       underperforming.
  - When the flag is empty, all of the above is skipped — backwards-compatible
    with every existing baseline / sequence-baseline run.

  **Patterns to follow:** Existing `_build_metrics_payload`,
  `_build_markdown_summary`, `_build_sequence_metrics_payload`,
  `_build_sequence_markdown_summary`, `_print_baseline_terminal_summary`. The
  new code is a new section; do not refactor the existing builders.

  **Test scenarios:**
  - CLI smoke test: invoke `sequence-baseline --subgroup-cols sex,age_band ...`
    against the toy CLIF fixture (extend `_build_toy_clif_with_categories` to
    include `sex_category` in patient table — currently it doesn't).
    Assert exit code 0; assert `payload["subgroups"]` exists and has rows for
    each (variable, value) pair; assert the markdown contains
    `## Subgroup performance`.
  - Validation: `--subgroup-cols foo` (unknown column) → exit code 2 +
    BadParameter message naming the invalid column.
  - Empty `--subgroup-cols` (default): payload does NOT contain a `subgroups`
    key; markdown does NOT contain the section header. Backwards-compatible.
  - Single-class subgroup: smoke test deliberately constructs a fixture where
    one age band has all `mortality==0`. The JSON row for that subgroup has
    `"warning": "single_class_y_true"` and `null` for `auroc` / `auprc`.

  **Verification:** `pytest tests/test_cli.py -v -k subgroup` passes;
  `icumodelstream baseline --help` shows the new flag.

---

## System-Wide Impact

| Surface | Impact |
|---|---|
| `src/icumodelstream/` | 1 new module (`subgroups.py`), 1 modified (`pipeline.py` — additive only) |
| `tests/` | 1 new test file (`test_subgroups.py`), 2 modified (`test_pipeline.py`, `test_cli.py` — additive) |
| `src/icumodelstream/cli.py` | New `--subgroup-cols` flag on both `baseline` and `sequence-baseline`; new private helper for markdown section; payload builders attach optional `subgroups` block |
| `pyproject.toml` | No change — uses existing numpy / polars / sklearn |
| `Makefile` | No change |
| `reports/` and `models/` | No new output paths; subgroup block is part of the existing JSON / MD outputs |

Affected parties:

- **A future contributor doing decision-curve analysis or external validation** —
  both follow-up plans depend on `test_hospitalization_ids` being exposed; this
  plan lands that as a side effect, unblocking both.
- **The publication** — Subgroup tables become a one-line CLI invocation:
  `icumodelstream baseline --subgroup-cols sex,race_category,age_band,icu_type
  --data-root ...`. JSON output is publication-table-ready.

---

## Risks & Dependencies

| Risk | Severity | Mitigation |
|---|---|---|
| Small subgroups (e.g., a race category with n < 30) produce noisy metrics | Medium | No silent filtering — every subgroup is reported with its `n` so the publication table author can see what's reliable. The first publication should report all subgroups + flag low-n ones in a footnote. |
| Single-class subgroups (e.g., all-survivor 18-25 age band) make AUROC undefined | Medium | `compute_metrics` already returns NaN; subgroup row gets a `warning` field. The JSON consumer sees the warning and handles. Tests pin this. |
| `adt.location_category` ICU subtypes vary across CLIF sites | Low | Same problem the cohort builder already handles via tolerant matching — for the first publication, MIMIC-only is sufficient. External validation plan will surface cross-site name drift. |
| Missing subgroup data (null `race_category`, etc.) silently dropped | Medium | Explicit `"Unknown"` bucket per variable; tests pin the contract. |
| Polars vs pandas join semantics on null hospitalization_ids | Low | Use left joins everywhere — the cohort's hospitalization_ids drive the output, not the patient/adt tables. Polars left-join behavior with nulls is well-defined. |
| Future LSTM result type drift | Low | `compute_subgroup_metrics` consumes `y_true` and `y_pred_proba` arrays — agnostic to which model produced them. Duck-typing on these two fields is the same pattern `_baseline_result_to_dict` already relies on. |

---

## Documentation Notes

- `docs/data_dictionary_notes.md`: already documents the relevant CLIF
  columns; no doc change required for this plan.
- `docs/roadmap.md`: add a row for "Phase 6 — Publication-track validation"
  with this plan as the first sub-deliverable (subgroup performance).
  Decision-curve analysis and external validation are subsequent
  sub-deliverables in the same phase.
- `CLAUDE.md` Repository map: add row for `src/icumodelstream/subgroups.py`
  once U1 lands.
- This plan does NOT change the `--feature-set` or `--outcome` contracts on
  existing commands; the new flag is purely additive.

---

## Sources & References

- `docs/data_dictionary_notes.md` — CLIF column vocabulary
- `docs/plans/2026-05-24-002-feat-lightgbm-baseline-phase4-plan.md` — Phase 4
  baseline contract (the analysis target)
- `docs/plans/2026-05-25-001-feat-phase5-sequence-model-plan.md` — Phase 5 LSTM
  contract (the second analysis target)
- `docs/plans/2026-05-25-002-feat-sequence-baseline-cli-plan.md` — Phase 5 U4
  CLI integration pattern this plan mirrors
- `src/icumodelstream/models.py` — `compute_metrics`, `calibration_table`
  (reused as-is)
- `src/icumodelstream/labels.py` — `extract_mortality_labels` (the CLIF-join
  pattern U2 mirrors)
- `src/icumodelstream/cli.py` — `_build_*_metrics_payload`, `_build_*_markdown_summary`
  (the structure U3 extends)
- [TRIPOD+AI checklist](https://www.equator-network.org/reporting-guidelines/tripod-ai/) —
  the publication checklist this plan brings the baseline into compliance with
- [FDA AI/ML Software as a Medical Device guidance](https://www.fda.gov/medical-devices/software-medical-device-samd/artificial-intelligence-and-machine-learning-aiml-enabled-medical-devices) —
  algorithmic fairness as a deployment gate
- CLAUDE.md — §2 (simplicity first), §7 (fail loudly), §10 (small commits with tests)
