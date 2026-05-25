---
title: "feat: `icumodelstream sequence-baseline` CLI command"
status: active
created: 2026-05-25
plan_type: feat
depth: lightweight
---

# feat: `icumodelstream sequence-baseline` CLI command

**Goal:** Add one Typer command that runs the already-built LSTM end-to-end against
real CLIF data, writes the same JSON/MD/model triple the `baseline` command does.
This is U4 of the parent Phase 5 plan
(`docs/plans/2026-05-25-001-feat-phase5-sequence-model-plan.md`) carved out at
execution scope. Mac-runnable; GPU training is a separate user-side step.

This plan is intentionally lightweight: 2 implementation units, each fits in a single
focused edit + a single test. The constraint is small-bite execution, not architectural
ambition.

---

## Problem Frame

`fit_sequence_model(sequences, labels, groups, ..., device, seed) -> (LSTMBaseline, SequenceResult)`
already exists in `src/icumodelstream/torch_train.py` and works end-to-end on synthetic
data. What's missing is the glue from real CLIF parquet on disk down to that function
call:

1. Build cohort + extract labels (already done — same as `baseline` command)
2. Build admission anchors (already done — `get_admission_anchors` in `pipeline.py`)
3. Build per-hospitalization sequence tensors (already done — `build_sequence_tensors`)
4. Hand the tensors to `fit_sequence_model`
5. Write metrics JSON, markdown summary, model `.pt` artifact, terminal summary

Steps 1-4 are roughly 15 lines of orchestration. Step 5 reuses the JSON/MD writers
from the `baseline` command via a slim adapter.

**Why two units, not one:** The orchestration logic (the 15 lines in the CLI body)
is testable in isolation if pulled into a private helper. Splitting CLI + helper
prevents one fat function that's hard to unit-test and easier to dispatch via subagent
without scope creep.

---

## Requirements Trace

| Requirement | Source |
|---|---|
| New `icumodelstream sequence-baseline` Typer command | Parent Phase 5 plan U4 |
| Same JSON / MD / model artifact contract as `baseline` | Parent plan: "same JSON / MD / model artifact contract" |
| `--device {auto,cuda,mps,cpu}` flag | Parent plan |
| `--max-epochs`, `--patience`, `--batch-size`, `--learning-rate` flags | Parent plan |
| `--outcome {mortality,los_gt_7d}` flag (reuses existing pipeline contract) | Parent plan + parity with `baseline` |
| Smoke test on toy CLIF fixture (no GPU required) | Parent plan U4 verification |
| `--seed 42` reproducibility plumbed end-to-end | CLAUDE.md §10 |
| Patient-aware split via `prepare_split_tensors` (already enforced inside `fit_sequence_model`) | CLAUDE.md §8 |
| Model artifact persisted via `torch.save(state_dict)` to `models/sequence_baseline.pt` | Parent plan key decisions |

---

## Scope

### In scope
- New private helper `_run_sequence_baseline(tables, ...)` in `cli.py` that orchestrates
  cohort → labels → anchors → sequences → `fit_sequence_model` and returns the trained
  model + `SequenceResult`. Pure function over tables + parameters.
- New `@app.command()` `sequence_baseline` in `cli.py` that wraps the helper, writes
  the three artifacts, prints a Rich summary.
- Adapter to reuse `_build_metrics_payload` from the existing `baseline` command —
  either by adding a `model_type: "lstm" | "lightgbm"` discriminator to the JSON, or by
  constructing a separate but structurally similar payload (decision recorded below).
- One CliRunner smoke test (`tests/test_cli.py`) using the existing toy CLIF fixture
  pattern. Asserts exit-0, asserts the three artifacts exist and parse correctly.

### Deferred to follow-up work
- A standalone `run_sequence_baseline_pipeline` in `pipeline.py` (parent plan considered
  this; the orchestration is small enough that inlining in `cli.py` is cleaner today —
  promote to pipeline.py only when a second caller appears)
- Reading the saved `.pt` back via a `predict` CLI command (separate plan)
- Marimo notebook driving `sequence-baseline` (parent plan deferred this)

### Out of scope
- Any change to `sequences.py`, `torch_train.py`, `torch_models.py`, `models.py`,
  `labels.py`, `cohorts.py`, `pipeline.py` (other than imports if needed in `cli.py`)
- GPU configuration / cloud provider runbook (that's U5)

---

## Context & Research

**Existing pattern to mirror:** `baseline` command in `src/icumodelstream/cli.py`.
Read it once before editing. Note the helpers already in `cli.py`:

- `_resolve_data_root(data_root)` — argument validation
- `_discover(data_root)` — converts FileNotFoundError / ValueError into typer.BadParameter
- `_git_head_sha()` — best-effort HEAD SHA for reproducibility marker
- `_nan_to_none(obj)` — strict-JSON NaN handling
- `_baseline_result_to_dict(result)` — serializes `BaselineResult` (drops `y_true` /
  `y_pred_proba` arrays, keeps metrics + calibration table)
- `_build_metrics_payload(data_root, result, code_version, generated_at)` — top-level
  JSON shape
- `_build_markdown_summary(...)`, `_print_baseline_terminal_summary(...)` — display

`SequenceResult` has fields `model_name`, `y_true`, `y_pred_proba`, `metrics`,
`calibration_table`, `epochs_trained`, `early_stopped_at_epoch`. The first five match
`BaselineResult` exactly, so `_baseline_result_to_dict` can be widened to accept either
(it just reads the metrics dict + calibration table; both have those).

**Key technical decision — JSON shape:**
- Keep `_baseline_result_to_dict` polymorphic by duck-typing (reads `.metrics`,
  `.calibration_table`). Add nothing.
- Build a NEW helper `_build_sequence_metrics_payload(data_root, model, result, ..., code_version, generated_at)`
  that has its own top-level shape: `{"config", "cohort_waterfall", "split", "model_type": "lstm", "lstm": <dict>, "training": {"epochs_trained", "early_stopped_at_epoch"}, "generated_at", "code_version"}`.
  Different enough from baseline that a separate payload builder is honest. Reuses
  `_baseline_result_to_dict` for the inner LSTM-result serialization.

**Device autodetect:** Already lives in `torch_train._autodetect_device`. CLI flag
`--device auto` passes `None` into `fit_sequence_model`, which calls the autodetect
internally.

**Model save path:** `torch.save(model.state_dict(), path)` is the spec'd serialization
(parent plan, U2 key decisions). Path defaults to `models/sequence_baseline.pt`.
`models/` is already in `.gitignore`.

---

## Key Technical Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Orchestration location | Private helper `_run_sequence_baseline` inside `cli.py` | Parent plan considered a `run_sequence_baseline_pipeline` in `pipeline.py`; that's the right move only when a 2nd caller appears. Today the only caller is the CLI command. Keep it inline + private. |
| JSON shape | Separate `_build_sequence_metrics_payload` (not unified with `_build_metrics_payload`) | LSTM result is shaped slightly differently (no `logistic` comparator, plus `epochs_trained`); a separate payload is cleaner than overloading the existing function with conditional branches. |
| Inner serialization | Reuse existing `_baseline_result_to_dict` — duck-typed on `.metrics` + `.calibration_table` | Both `BaselineResult` and `SequenceResult` expose those attributes. Polymorphism via field-name compatibility avoids touching the existing helper. |
| Model save format | `torch.save(state_dict)` (not full pickle) | Parent plan U2 contract. Loadable into a fresh `LSTMBaseline(input_dim=...)` later. |
| Config sidecar | Embedded in the JSON `config` block (not a separate file) | The JSON already serves as the reproducibility record; doubling it as a sidecar is noise. |
| Outcome dispatch | Reuse the same `--outcome {mortality, los_gt_7d}` plumbing the `baseline` command has, threading through to label extraction inside the helper | Parity with `baseline` so the comparison is apples-to-apples. |
| Device flag default | `"auto"` (translated to `None` for `fit_sequence_model`) | User who passes nothing gets autodetect (cuda > mps > cpu). Explicit override via `--device cpu` etc. |

---

## Open Questions

None. The Mac-runnable scope is fully constrained by the parent plan + existing `cli.py`
patterns. Actual GPU training and provider selection is U5, not this plan.

---

## Implementation Units

- U1. **Add `_run_sequence_baseline` helper + `sequence_baseline` CLI command**

  **Goal:** A new Typer command that orchestrates cohort → labels → sequences →
  `fit_sequence_model` → writes JSON + MD + `.pt`. Plus the supporting private helper
  (`_run_sequence_baseline`) and the new payload builder (`_build_sequence_metrics_payload`).

  **Requirements:** All "In scope" items above except the smoke test.

  **Dependencies:** None within this plan (parent plan U1-U3 already shipped).

  **Files:**
  - `src/icumodelstream/cli.py` (modify — append at end of file; do not touch existing
    `baseline` command or its helpers)

  **Approach:** Three sub-pieces inside `cli.py`, written top-to-bottom:

  1. **`_run_sequence_baseline(tables, cohort_spec, *, window_hours, outcome, los_threshold_hours, include_hospice, hidden_dim, n_layers, dropout, max_epochs, patience, learning_rate, weight_decay, batch_size, device, seed) -> tuple[LSTMBaseline, SequenceResult, CohortWaterfall, dict[str, Any]]`**

     The returned dict carries split metadata: `n_train`, `n_val`, `n_test`,
     `n_train_patients`, `n_val_patients`, `n_test_patients`, `train_prevalence`,
     `val_prevalence`, `test_prevalence`, `n_channels`, `channel_names`. (Computed by
     calling `prepare_split_tensors` once — same deterministic split that
     `fit_sequence_model` will run internally.)

     Body:
     - `build_cohort_with_waterfall(tables, cohort_spec)` → `cohort, waterfall`
     - Label dispatch on `outcome` (mortality / los_gt_7d), rename to `"outcome"` column
       (mirrors `run_baseline_pipeline` exactly — copy the 4-line dispatch block)
     - Inner-join cohort + labels
     - `anchors = get_admission_anchors(tables).join(...)`
     - `sequences = build_sequence_tensors(tables, cohort_with_labels, anchors, window_hours, RICH_VITAL_CATEGORIES, RICH_LAB_CATEGORIES, RICH_ASSESSMENT_CATEGORIES, RICH_RESPIRATORY_DEVICES)`
     - Build `labels_for_torch` / `groups_for_torch` DataFrames
     - `split_tensors = prepare_split_tensors(sequences, labels_for_torch, groups_for_torch, seed=seed)` (for prevalences)
     - Compute prevalences + split sizes from `split_tensors`
     - `lstm_model, lstm_result = fit_sequence_model(sequences, labels_for_torch, groups_for_torch, hidden_dim=..., n_layers=..., dropout=..., max_epochs=..., patience=..., learning_rate=..., weight_decay=..., batch_size=..., device=device, seed=seed)`
     - Return `(lstm_model, lstm_result, waterfall, split_meta)`

  2. **`_build_sequence_metrics_payload(*, data_root, model_state, result, waterfall, split_meta, config_snapshot, code_version, generated_at) -> dict`**

     Top-level JSON shape:
     ```
     {
       "config": {...},                 # includes outcome, device, hidden_dim, etc.
       "model_type": "lstm",
       "cohort_waterfall": {...asdict(waterfall)...},
       "n_channels": int,
       "channel_names": list[str],
       "split": {n_train, n_val, n_test, n_*_patients, *_prevalence},
       "model": _baseline_result_to_dict(result),   # metrics + calibration_table
       "training": {"epochs_trained": int, "early_stopped_at_epoch": int | None},
       "warnings": [],                                # reserved for future use; empty for now
       "generated_at": str,
       "code_version": str | None,
     }
     ```

  3. **`@app.command() def sequence_baseline(...)` Typer command** with flags:
     `--data-root` (required), `--metrics-out` (default `reports/sequence_metrics.json`),
     `--summary-out` (default `reports/sequence_summary.md`), `--model-out`
     (default `models/sequence_baseline.pt`), `--min-age` (18), `--require-icu-location`
     (True), `--window-hours` (24), `--seed` (42), `--outcome` (`mortality`),
     `--los-threshold-hours` (168.0), `--include-hospice` (False), `--hidden-dim` (128),
     `--n-layers` (2), `--dropout` (0.3), `--max-epochs` (20), `--patience` (3),
     `--batch-size` (256), `--learning-rate` (1e-3), `--weight-decay` (1e-4),
     `--device` (`auto` — translated to `None` before passing to `_run_sequence_baseline`).

     Body:
     - Resolve `data_root`, `tables = _discover(...)`
     - Build `cohort_spec`
     - Call `_run_sequence_baseline(...)` to get `model, result, waterfall, split_meta`
     - `model_out.parent.mkdir(parents=True, exist_ok=True)`; `torch.save(model.state_dict(), model_out)`
     - Build `config_snapshot` dict from CLI args
     - `payload = _build_sequence_metrics_payload(...)`
     - Write `metrics_out` via `json.dump(_nan_to_none(payload), ...)`
     - Build markdown summary string and write to `summary_out`. (Adapt
       `_build_markdown_summary` if it's polymorphic enough, otherwise write a small
       sequence-specific markdown builder — judgment call during implementation.)
     - Print a Rich summary to terminal: cohort size, prevalence, AUROC/AUPRC/Brier/CalibSlope, epochs_trained, where files landed.

  **Patterns to follow:** The existing `baseline` command in `cli.py`. The orchestration
  inside `_run_sequence_baseline` mirrors `run_baseline_pipeline` in `pipeline.py` but
  with sequences instead of aggregates.

  **Test scenarios:** None in this unit. The smoke test is U2 (kept separate so this
  unit can land first and U2 picks up only what U1 actually wired).

  **Verification:**
  - `.venv/bin/icumodelstream sequence-baseline --help` exits 0 and lists all flags
  - No tests broken: `.venv/bin/pytest tests/test_cli.py -v` (targeted file, not full suite —
    full suite has known LightGBM threading flakiness)

---

- U2. **Smoke test for `sequence-baseline` against the toy CLIF fixture**

  **Goal:** One CliRunner-based integration test that confirms the command runs
  end-to-end on synthetic CLIF data, exits 0, writes all three artifacts, and the JSON
  parses with the expected top-level keys.

  **Requirements:** "Smoke test on toy CLIF fixture (no GPU required)" requirement
  from the trace.

  **Dependencies:** U1 must be in place.

  **Files:**
  - `tests/test_cli.py` (modify — append one new test at end; do not touch existing
    `baseline` tests)

  **Approach:** Reuse the same `_build_toy_clif(tmp_path, ...)` fixture pattern that the
  existing CLI tests use (read `tests/test_cli.py` first to see the exact helper
  signature). Run:

  ```
  runner = CliRunner()
  result = runner.invoke(app, [
      "sequence-baseline",
      "--data-root", str(tmp_path),
      "--metrics-out", str(metrics_path),
      "--summary-out", str(summary_path),
      "--model-out", str(model_path),
      "--require-icu-location", "False",
      "--hidden-dim", "8",
      "--n-layers", "1",
      "--max-epochs", "2",
      "--batch-size", "4",
      "--device", "cpu",
      "--seed", "42",
  ])
  ```

  **Test scenarios:** All in one test, named
  `test_sequence_baseline_command_happy_path`:

  - **Happy path:** runs against ~20-hospitalization toy CLIF with both classes present.
    Assert `result.exit_code == 0`. Assert all three files exist (`metrics_path`,
    `summary_path`, `model_path`).
  - **JSON shape:** Parse `metrics_path`, assert top-level keys include
    `config`, `model_type` (== `"lstm"`), `cohort_waterfall`, `n_channels`,
    `channel_names`, `split`, `model`, `training`, `generated_at`. Assert
    `payload["model"]["metrics"]` has the 6 expected keys (auroc, auprc, brier_score,
    prevalence, calibration_intercept, calibration_slope).
  - **Model loadable:** `state_dict = torch.load(model_path)`. Assert it's a dict (or
    a `collections.OrderedDict`) with at least one key. Don't try to load it back into
    an `LSTMBaseline` — that requires knowing `input_dim`, which is fine to defer to
    a future `predict` command.
  - **Markdown sanity:** `summary_path.read_text()` contains `"## Cohort waterfall"` or
    similarly-named section header (sniff one substring; don't pin the full template).

  Use tiny model (`--hidden-dim 8 --n-layers 1`) and tiny `--max-epochs 2` to keep the
  test under 30 seconds on CPU.

  **Patterns to follow:** Existing `test_baseline_command_happy_path` in
  `tests/test_cli.py`.

  **Verification:** `.venv/bin/pytest tests/test_cli.py -v -k sequence_baseline`
  exits 0. Total test count: existing CLI test count + 1.

---

## System-Wide Impact

| Surface | Impact |
|---|---|
| `src/icumodelstream/cli.py` | +1 command + 2 private helpers (~80 lines total) |
| `tests/test_cli.py` | +1 test (~30 lines) |
| `pyproject.toml` | No change (torch already shipped via U6-prereq) |
| `Makefile` | No change today; optional follow-up to add `make sequence-baseline` target later |
| `reports/` and `models/` | New default output paths, both already gitignored |

Affected parties:
- **The user** — gets a headless `icumodelstream sequence-baseline ...` command that
  mirrors the existing `baseline` UX. Can run on Mac CPU/MPS or rented GPU with
  `--device cuda`.
- **A future contributor adding more sequence outcomes** — has a wired CLI to extend
  rather than starting from scratch.

---

## Risks & Dependencies

| Risk | Severity | Mitigation |
|---|---|---|
| `_baseline_result_to_dict` polymorphism breaks if `SequenceResult` shape drifts | Low | The duck-typing depends on `.metrics` (dict) and `.calibration_table` (polars DataFrame). Both are stable contracts. Add a type comment in `cli.py` noting the helper accepts either; if anyone widens it further, they'll see the note. |
| `torch.save` of a CPU-loaded model on a MPS/CUDA device produces device-specific tensors | Medium | Always move model to CPU before saving: `torch.save(model.cpu().state_dict(), path)`. Document this in the helper. |
| Markdown summary builder code duplicates the existing one in `cli.py` | Low | Acceptable for a small command. Refactor to a shared builder only when a third command appears. |
| Full pytest suite flaky in this environment (LightGBM threading) | Confirmed in session | Always verify via targeted file run (`pytest tests/test_cli.py -v`), not `pytest -q`. Documented in U1 + U2 verification. |
| Subagent dispatch caused a loop earlier in this session | Procedural | Each unit is small enough to implement directly (no subagent needed). If a subagent IS used, keep the prompt under 100 lines and run targeted-file tests only. |

---

## Documentation Notes

- After U1+U2 land, update `README.md` quick-start to list `icumodelstream sequence-baseline`
  alongside the other commands. Separate small commit; not part of this plan.
- Update `docs/roadmap.md` Phase 5 row from "in progress" to a state reflecting "CLI
  ready; awaiting GPU rental" once U2 is committed.
- The parent Phase 5 plan U5 (rented-GPU runbook) is the next plan to draft after this
  one lands.

---

## Sources & References

- `docs/plans/2026-05-25-001-feat-phase5-sequence-model-plan.md` — origin plan; U4 row
- `src/icumodelstream/cli.py` — `baseline` command is the template
- `src/icumodelstream/pipeline.py` — `run_baseline_pipeline` for the outcome dispatch
  block to mirror
- `src/icumodelstream/torch_train.py` — `fit_sequence_model`, `prepare_split_tensors`
- `src/icumodelstream/sequences.py` — `build_sequence_tensors`, `SequenceTensors`
- `src/icumodelstream/torch_models.py` — `LSTMBaseline`, `SequenceResult`
- `tests/test_cli.py` — existing `test_baseline_command_happy_path` for fixture pattern
- CLAUDE.md §8 (no patient leakage — enforced upstream), §9 (baselines first — done),
  §10 (small commits, reproducibility — this plan)
