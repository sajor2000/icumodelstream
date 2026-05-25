---
title: "Decision-curve analysis (TRIPOD+AI clinical utility)"
type: feat
status: active
created: 2026-05-25
plan_id: 2026-05-25-004
parent: docs/phase6-progress-2026-05-25.md
related:
  - docs/plans/2026-05-25-003-feat-subgroup-performance-plan.md
---

# Decision-curve analysis (DCA) for the LightGBM + LSTM baselines

## Parent context

Phase 6 publication-track work. Subgroup performance is shipped (plan `…-003`).
DCA is the second of three TRIPOD+AI evaluation pieces that complete the
manuscript-grade evaluation triad:

| Triad piece | Status |
|---|---|
| Calibration (intercept, slope, Brier, reliability table) | ✅ (Phase 4 / 5) |
| Subgroup performance | ✅ (Phase 6.a) |
| **Decision-curve analysis (net benefit)** | **this plan** |

External validation (out-of-MIMIC cohort) is **out of scope** — it has a data-access
dependency that is not a coding task. It will be its own plan.

## Goal

Report **net benefit** at clinically actionable probability thresholds for both
the LightGBM and the LSTM baseline, compared against the two reference strategies
(`treat-all` and `treat-none`). Output goes into the existing JSON payload and
the existing Markdown summary, behind a `--dca-thresholds` flag that, like
`--subgroup-cols`, defaults to empty so old baseline runs are unchanged.

Net benefit at threshold `pt`:

```
NB(pt) = TP/n − (FP/n) * (pt / (1 − pt))
```

…where `TP` and `FP` are counted at the cut-point `y_pred_proba >= pt`. The two
reference curves are computed against the same denominator so the consumer can
plot all three side by side.

## Why this fits the existing shape

DCA does **not** need any new CLIF plumbing. Both `BaselineResult` and
`SequenceResult` already expose `y_true` and `y_pred_proba` on the test set.
The new module is pure numpy/polars (mirrors `src/icumodelstream/subgroups.py`)
and the CLI integration is byte-for-byte parallel to the subgroup-cols work that
just landed.

That's why this is **two units** instead of three — no pipeline plumbing is needed.

---

## Implementation units

### U1 — `src/icumodelstream/decision_curves.py` + tests

**Files:** `src/icumodelstream/decision_curves.py` (new), `tests/test_decision_curves.py` (new).
**Estimated LOC:** ~60 module + ~80 test.

Module public surface:

```python
DEFAULT_THRESHOLDS: tuple[float, ...] = (0.05, 0.10, 0.15, 0.20, 0.30, 0.50)

def compute_decision_curve(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
) -> pl.DataFrame:
    """Per-threshold net benefit + the two reference strategies.

    Returns a tidy long DataFrame with columns:
      threshold, n, prevalence, n_positive_pred, n_true_positive,
      n_false_positive, net_benefit, net_benefit_treat_all,
      net_benefit_treat_none.

    `net_benefit_treat_none` is always 0 by construction; included for
    plotting symmetry. `net_benefit_treat_all` = prevalence − (1−prevalence) * pt/(1−pt).
    """
```

Design decisions (encode in code comments where load-bearing):

- **Thresholds in (0, 1).** Reject `pt <= 0` and `pt >= 1` with a `ValueError`
  naming the offending value (CLAUDE.md §7).
- **Length mismatch raises**, same shape as `compute_subgroup_metrics`.
- **No plotting.** This module only computes the curve. Visualization is a notebook
  concern and lives in `notebooks/06_decision_curve.py` if/when added.
- **Reuse existing weighting.** Net benefit treats each test-set hospitalization
  equally — same denominator as `compute_metrics`. No survey weights.
- **NaN handling.** If a threshold is so high that no rows are positive
  (`n_positive_pred == 0`), `net_benefit` is exactly `0` (zero TP and zero FP).
  Do not return NaN; clinical readers expect a 0-line on the plot, not a gap.

#### U1 test scenarios (6 — match the subgroups.py test footprint)

1. **Happy path** — perfectly separable signal, thresholds `[0.1, 0.5, 0.9]`.
   Assert per-threshold `net_benefit > 0`, `net_benefit_treat_all` matches the
   analytic formula, `net_benefit_treat_none == 0`.
2. **Worthless model** — `y_pred_proba` is constant 0.5. Net benefit at `pt=0.5`
   equals `prevalence − (1−prevalence)` (i.e., treat-all at this threshold).
3. **Boundary thresholds rejected** — `pt=0.0` and `pt=1.0` raise `ValueError`.
4. **Length mismatch raises** — different lengths for `y_true` and `y_pred_proba`.
5. **Empty thresholds** → empty DataFrame with expected schema (mirror
   `_empty_subgroup_frame`).
6. **High threshold zero-cohort** — `pt=0.99` with no scores above it returns
   `net_benefit == 0` (not NaN) and `n_positive_pred == 0`.

### U2 — CLI integration on both baseline commands

**Files modified:** `src/icumodelstream/cli.py`, `tests/test_cli.py`.
**Estimated LOC:** ~50 CLI + ~60 test.

Mirror the subgroup-cols wiring exactly:

- New helper `_parse_dca_thresholds(raw: str) -> list[float]` — comma-split,
  cast to `float`, drop empties, validate each ∈ (0,1) with
  `typer.BadParameter` on failure (so unknown input fails loudly at exit 2,
  matching `_parse_subgroup_cols`).
- New helper `_compute_dca_block(y_true, y_pred_proba, thresholds) -> list[dict[str, Any]]`
  that calls `compute_decision_curve` and returns `[]` when `thresholds` is
  empty. (No CLIF coupling — does not take `tables`.)
- New helper `_render_dca_md(rows) -> str` — emits a `## Decision-curve analysis`
  section with one Markdown table:

  ```
  | Threshold | Net benefit | NB (treat all) | NB (treat none) | n+ pred | TP | FP |
  ```

  Returns `""` when `rows` is empty so callers can append unconditionally.
- Add `--dca-thresholds` (default `""`) `typer.Option` to **both** the `baseline`
  and `sequence_baseline` commands. Same docstring shape as `--subgroup-cols`:
  *"Comma-separated probability thresholds for decision-curve analysis. Empty
  (default) skips analysis. Example: 0.05,0.10,0.20,0.50"*.
- In both command bodies: compute DCA after model fit, attach
  `payload["decision_curve"] = dca_rows` only when non-empty, and append
  `_render_dca_md(dca_rows)` to the markdown summary.

#### U2 test scenarios (3 — parallel to the subgroup-cols tests)

1. **`test_baseline_dca_thresholds_happy_path`** — `--dca-thresholds 0.1,0.2,0.5`
   produces `payload["decision_curve"]` with 3 rows, each row has the seven
   numeric fields populated; summary contains `## Decision-curve analysis` and
   one row per threshold.
2. **`test_baseline_dca_thresholds_default_empty_omits_key`** — default flag
   leaves `payload["decision_curve"]` absent and no DCA section in the summary
   (byte-for-byte unchanged from a pre-DCA run).
3. **`test_baseline_dca_thresholds_invalid_value_fails_loudly`** —
   `--dca-thresholds 0.5,1.5` exits with code `2` (`typer.BadParameter`).

Optionally, **one** parallel sequence-baseline smoke test that asserts the
flag works on `sequence-baseline` too (lightweight — same fixture as the
existing sequence-baseline happy path test, just with the flag added).

---

## Files changed (final)

| File | Change |
|---|---|
| `src/icumodelstream/decision_curves.py` | **NEW** — `compute_decision_curve` + `DEFAULT_THRESHOLDS` |
| `tests/test_decision_curves.py` | **NEW** — 6 unit tests |
| `src/icumodelstream/cli.py` | Add `_parse_dca_thresholds`, `_compute_dca_block`, `_render_dca_md`; wire `--dca-thresholds` into `baseline` and `sequence_baseline` |
| `tests/test_cli.py` | 3 (+ optional 1) new CLI smoke tests |

No edits to: `pipeline.py`, `models.py`, `torch_train.py`, `subgroups.py`,
`features.py`, the toy CLIF fixture, or any notebook. DCA is computed from the
existing result objects.

---

## Verification

```bash
# After U1
.venv/bin/pytest tests/test_decision_curves.py -v   # 6 pass

# After U2
.venv/bin/pytest tests/test_cli.py -v               # 10 (or 11 with sequence test) pass
.venv/bin/pytest -q                                 # full suite green

# Smoke
icumodelstream baseline \
  --data-root $CLIF_ROOT \
  --metrics-out out/dca_metrics.json \
  --summary-out out/dca_summary.md \
  --dca-thresholds 0.05,0.10,0.20,0.50 \
  --subgroup-cols sex,age_band \
  --seed 42
# Expect payload to contain both `subgroups` and `decision_curve` blocks.
```

---

## Out of scope

- Plot rendering (matplotlib / altair). Notebook concern, separate work item.
- Confidence intervals around net benefit (bootstrap). Defer to a "publication
  polish" plan once external validation is available.
- Stratified DCA by subgroup. Possible follow-up, but not needed for the headline
  manuscript figure.
