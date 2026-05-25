---
title: "feat: Phase 5 sequence-model baseline on rented GPU"
status: active
created: 2026-05-25
plan_type: feat
depth: deep
---

# feat: Phase 5 sequence-model baseline on rented GPU

**Goal:** Beat the rich LightGBM baseline (AUROC 0.866 on in-hospital mortality, 0.760 on
LOS > 7d) with a learned time-series representation that doesn't require hand-crafted
per-category aggregates. The Mac builds the data extraction and training scaffold; actual
training runs on a rented CUDA GPU. CLAUDE.md §9 explicitly gates Phase 5 on Phase 4 being
reproducible — that gate is now satisfied (commit `c33488c`).

This is the first deep-learning work in the repository. Bias is still toward **boring**:
start with an LSTM (the smallest sensible sequence model) and only escalate to a transformer
or foundation model if the LSTM doesn't materially beat LightGBM. CLAUDE.md §2 (simplicity
first) and §9 (baselines before deep learning) apply throughout.

---

## Problem Frame

The LightGBM baseline at 0.866 AUROC is strong but uses 89 hand-crafted features and
collapses each hospitalization to a single 24-hour summary. The clinical reality is that
trajectories matter: a patient whose lactate goes from 4 → 8 in 12 hours is sicker than one
whose lactate goes from 4 → 4 over the same window, but both produce identical
`labs_lactate_mean = 4` and `labs_lactate_max = 8` features. A sequence model can read the
trajectory directly.

The work is split into three distinct concerns:
1. **Data shape** — Convert per-event CLIF rows into per-hospitalization tensors:
   `(n_hosps, n_timesteps, n_channels)`. Timesteps are hourly bins for the first 24h
   post-admission; channels are the same 24 numeric signals (7 vitals + 12 labs + 2 GCS +
   3 respiratory) we already aggregate, plus a mask channel per signal.
2. **Architecture** — Start with LSTM (1-2 layers, 64-128 hidden units). Escalate only if
   warranted. Foundation-model fine-tune (e.g., ClinicalBERT, Med-LLaMA, BioGPT) is
   explicitly deferred until LSTM and transformer-from-scratch are both characterized.
3. **Compute** — Rented CUDA GPU (RunPod / Lambda Labs / Modal recommended for spot pricing).
   The training script must run dry on Mac MPS / CPU with synthetic data so changes can be
   validated locally before paying for GPU time.

What we **do not** want:
- A foundation-model wrapper as the first move — too much surface area, too many failure
  modes, too expensive to debug iteratively.
- An end-to-end training script tied to one cloud provider — keep training portable.
- Manual feature engineering replicating the LightGBM features inside a neural model.

---

## Requirements Trace

| Requirement | Source |
|---|---|
| Beat LightGBM AUROC 0.866 on mortality (or document why we couldn't) | CLAUDE.md §9 (baselines before DL) — DL must justify its own existence |
| No patient leakage in train/val/test | CLAUDE.md §8 (already enforced by `group_train_test_split`) |
| Reproducible: torch seed + numpy seed + dataloader workers seeded | CLAUDE.md §10 |
| Dry-run mode on Mac (synthetic data, MPS or CPU, completes in < 5 min) | CLAUDE.md "Add a synthetic-data dry run and keep it separate from CLIF-MIMIC data paths" |
| Calibration check (Brier + reliability table) — same shape as Phase 4 | Phase 4 contract; lets us compare apples-to-apples |
| Model artifacts gitignored under `models/`, never commit weights | CLAUDE.md data safety + existing `.gitignore` |
| GPU choice is the user's, not hard-coded in the script | "Separate rented-GPU scripts" — the user picks where to rent |
| Training cost estimate provided before any GPU time is spent | This plan documents expected GPU hours |

---

## Scope

### In scope
- New module `src/icumodelstream/sequences.py` — convert CLIF tables into a
  `(n_hosps, n_timesteps, n_channels)` numpy tensor plus a parallel mask tensor.
- New module `src/icumodelstream/torch_models.py` — PyTorch LSTM (and stub transformer
  scaffold) with a single `fit_sequence_model` entry point that returns a `SequenceResult`
  shaped like `BaselineResult`.
- New module `src/icumodelstream/torch_train.py` — training loop with seed plumbing,
  early-stopping on validation AUROC, optional MPS / CUDA / CPU backend selection.
- New CLI command `icumodelstream sequence-baseline` mirroring `baseline` but using the
  torch model. Same JSON / markdown / model artifact contract.
- Synthetic-data dry-run smoke test (`tests/test_torch_train.py`) that fits one epoch on
  10 fake hospitalizations and asserts the loss decreased. Runs on every `pytest -q`.
- `docs/plans/2026-05-25-002-rented-gpu-runbook.md` (companion) documenting the rental
  decision: which provider, expected cost, kickoff command, and the smoke-test that confirms
  the GPU rental is alive before any training fires.
- `pyproject.toml` — add `torch>=2.3` to a new `[ml-dl]` optional extras group (separate
  from `ml` so a clinician-engineer who only wants LightGBM doesn't pull down 3 GB of CUDA).

### Deferred to follow-up work
- Transformer-from-scratch architecture (scaffolded in `torch_models.py` but the LSTM is
  the implemented baseline)
- Foundation-model fine-tune (ClinicalBERT, Med-LLaMA, BioGPT) — wait until LSTM AUROC
  vs LightGBM AUROC tells us whether more capacity is warranted
- Hyperparameter tuning (Optuna / Ray Tune) — characterize first, tune second
- Distributed training (DDP / FSDP) — single GPU is sufficient for 85K hospitalizations
- SHAP / attention visualization for interpretability
- Decision curves, subgroup performance, fairness analysis (publication-track)
- Time-varying features beyond the first 24h post-admission

### Out of scope
- Any change to `cohorts.py`, `labels.py`, `qc.py`, `io.py`, `schema.py`, `config.py`,
  `splits.py` beyond imports. The Phase 4 cohort + label + split contract is the input
  here.
- Modifying the LightGBM/logistic baselines.
- A Marimo notebook for sequence training — the dry-run smoke test + CLI are the only
  interactive surfaces. Marimo on a remote GPU is awkward.

---

## Context & Research

**Existing primitives the sequence model will consume:**
- `cohorts.build_cohort_with_waterfall` — already produces a 85K-row cohort.
- `labels.extract_mortality_labels` / `labels.extract_los_label` — already produce per-
  hospitalization binary labels.
- `splits.group_train_test_split` — already enforces no patient leakage.
- `models.BaselineResult` — the shape the sequence model returns (metrics + calibration
  table). Keep parity so the CLI's JSON / markdown writers work unchanged.

**Existing data dictionary observations** (from `docs/data_dictionary_notes.md`):
- Vitals `recorded_dttm`, labs `lab_collect_dttm` are the timestamps we'll bin by.
- 7 vital categories, 12 lab categories, 2 assessment categories, 5 respiratory device flags
  — same channel set as `RICH_*` constants in `pipeline.py`. Total ≈ 26 channels.
- All `*_dttm` are tz-aware UTC; `admission_dttm` is the anchor.

**Tensor shape:** `(n_hosps, 24, n_channels * 2)` where `n_channels * 2` accounts for the
value channel + a binary "was this signal observed in this hour" mask channel. Hourly bins
are coarse enough that polars can do the binning lazily in a single pass.

**Compute considerations:**
- 85,248 hospitalizations × 24 timesteps × ~52 channels × float32 = ~430 MB tensor.
  Fits easily in 24GB VRAM.
- A 2-layer LSTM with 128 hidden units = ~150K params. ~2-5 minutes per epoch on an A10
  (24GB, ~$0.80/hour spot on RunPod as of Q2 2026). 10 epochs ≈ 30 minutes ≈ $0.40 per
  training run.
- Mac dry-run on synthetic 10-hospitalization tensor: < 30 seconds on MPS, < 10 seconds
  on CPU.

**GPU rental options surveyed** (May 2026 pricing, approximate):
- RunPod spot A10: $0.20–0.40/hr. Easiest to start; web UI + Python SDK.
- Lambda Labs A100: $1.30/hr on-demand. Cleaner CLI, higher reliability.
- Modal labs serverless: pay-per-second, ~$1.10/hr A10 equivalent. Best for the dry-run
  → train → done pattern that we expect.
- A user's own desktop / institution GPU if available — zero marginal cost but variable
  availability.

The plan does **not** commit to a provider — the companion runbook does. This plan is
GPU-agnostic and the training script must accept a `--device {cuda,mps,cpu}` flag.

---

## Key Technical Decisions

| Decision | Choice | Rationale |
|---|---|---|
| First architecture | 2-layer LSTM, 128 hidden, dropout 0.3 | Smallest sensible sequence model; well-understood; cheap to train. Comparable to LightGBM in parameter count (~150K vs LightGBM's ~3K leaves) |
| Sequence length | 24 hourly bins (first 24h post-admission) | Matches `window_hours=24` in Phase 4. Apples-to-apples comparison against LightGBM. Future plan can extend the window. |
| Aggregation within an hour | Mean of values in that hour (or null if no values) | Polars-native, single pass. Median would require extra computation; mean is fine for a baseline. |
| Mask channel | 1 if at least one value observed in that hour, else 0 | Lets the model learn "missingness is signal" without imputing |
| Imputation for value channel | Forward-fill from previous observed hour, then 0 for any leading nulls | Standard ICU sequence-model preprocessing. Documented; the mask channel preserves the original observation pattern. |
| Outcome targets | Mortality (priority 1) and LOS > 7d (priority 2) | Same as Phase 4 — direct comparison |
| Loss | BCE with logits + class weights = (n_neg/n_pos) | Mortality is 11.5% prevalence; class-weighted loss matches what `is_unbalance` did for LightGBM but on a calibrated scale |
| Optimizer | AdamW, lr=1e-3, weight_decay=1e-4 | Defaults; tune only if first results disappoint |
| Early stopping | Patience 3 on validation AUROC, max 20 epochs | Holdout patience > 0 to avoid memorization; max epochs caps cost |
| Train/val/test | 70/15/15 by patient_id via `group_kfold` | Adds a validation set for early stopping vs Phase 4's 80/20 train/test |
| Device selection | `--device {cuda,mps,cpu}` CLI flag, auto-detect default | Lets the same script run locally (dry-run) and on rented GPU |
| Model save format | `torch.save(state_dict)` to `models/sequence_baseline.pt` + JSON config sidecar | Standard PyTorch; the JSON sidecar makes loading reproducible without inspecting the state dict |
| Random seed | `seed=42` plumbed through torch, numpy, dataloader workers | Reproducibility is non-negotiable |
| Mixed precision | Off for the LSTM baseline; on (`torch.amp`) for the transformer follow-up | LSTM is fast enough at full precision; transformer is where it matters |

---

## Open Questions

**Resolved during planning:**
- LSTM vs transformer first → LSTM. The transformer is scaffolded but not implemented in
  this plan's scope.
- Outcome priority → mortality first, LOS > 7d as a follow-up the same week.
- Single GPU sufficient? → Yes for 85K hospitalizations; revisit if cohort grows 10×.

**Deferred to implementation** (decide when seeing the data):
- Whether to drop hospitalizations with < 6 hours of vitals data (sparse coverage). The
  current Phase 4 cohort includes them all; an LSTM may benefit from a coverage floor.
- Whether per-channel normalization (StandardScaler equivalent) is needed before LSTM
  input. Probably yes for stable training; verify with a 1-epoch test run.

**Open for the user before any GPU spend:**
- Which GPU provider. Default recommendation: **RunPod A10 spot** for the first training
  run (cheapest), Lambda A100 for any production retraining. This is in the runbook, not
  this plan.

---

## Output Structure

```
src/icumodelstream/
├── sequences.py         NEW   build_sequence_tensors() converts CLIF -> (n, t, c) tensor + mask
├── torch_models.py      NEW   LSTMBaseline + TransformerBaseline (stub) + SequenceResult
├── torch_train.py       NEW   train_sequence_model() loop, early stopping, device dispatch
└── cli.py               MOD   Add `sequence-baseline` command

tests/
├── test_sequences.py    NEW   tensor shape, mask correctness, time-window leakage
└── test_torch_train.py  NEW   1-epoch synthetic-data dry run, loss-decreased smoke test

scripts/
└── train_sequence.py    NEW   thin wrapper for rented-GPU invocation
                              (mirrors `make baseline` but expects --device cuda)

docs/plans/
└── 2026-05-25-002-rented-gpu-runbook.md   COMPANION
                              (Provider choice, kickoff command, smoke test, cost estimate)

pyproject.toml           MOD   Add [ml-dl] extras: torch>=2.3
```

---

## High-Level Technical Design

*Directional guidance for review, not implementation specification.*

### Data shape

For each hospitalization in the cohort, produce a tensor of shape `(24, n_channels)`:

```
hour 0: [hr=80, sbp=120, dbp=70, ..., spo2=98, lab_lactate=null, ..., gcs=15, mask_hr=1, mask_sbp=1, ..., mask_lactate=0]
hour 1: [hr=85, sbp=122, ..., mask_hr=1, ...]
...
hour 23: [...]
```

Value channels: 7 vitals + 12 labs + 2 GCS scores = 21 numeric signals.
Indicator channels: 5 respiratory device flags (binary).
Mask channels: 21 (one per numeric channel; the indicators don't need a mask).
Total: 21 + 5 + 21 = 47 channels.

Stack all hospitalizations: `X_tensor: (85_248, 24, 47)`, `y_tensor: (85_248,)`,
`groups: (85_248,)`.

### Architecture sketch

```
LSTMBaseline(
    input_dim=47,
    hidden=128,
    n_layers=2,
    dropout=0.3,
    bidirectional=False,
)
    → LSTM
    → take hidden state from last timestep
    → Linear(128, 1)
    → sigmoid (in BCEWithLogitsLoss, no explicit sigmoid)
```

Output is a single logit per hospitalization. Training loss is class-weighted BCE.
Evaluation reuses `_compute_metrics` and `_calibration_table` from `models.py` to keep
the metrics dict identical to Phase 4.

### Compare against Phase 4

After training, run the same per-bin reliability check and Brier score the LightGBM model
went through. Acceptance criterion: **LightGBM AUROC < Sequence AUROC** on the same
holdout, OR a documented finding that the sequence model is no better (which would itself
be a useful empirical result and would inform whether to attempt the transformer/
foundation-model follow-up).

---

## Implementation Units

- U1. **`src/icumodelstream/sequences.py` — Build sequence tensors**

  **Goal:** Convert tables + cohort + anchors → `(X_tensor, y_tensor, groups, channel_names)`.

  **Requirements:** Tensor shape correct, no leakage past window, mask channels honest.

  **Dependencies:** None code-wise. Conceptually reads the same Phase 4 cohort/labels/anchors.

  **Files:**
  - `src/icumodelstream/sequences.py` (new)
  - `tests/test_sequences.py` (new)

  **Approach:** Use polars LazyFrame to bin each channel by hour relative to the anchor.
  For each channel (vital/lab/assessment), aggregate by `(hospitalization_id, hour_bin)`
  taking mean of value. Pivot to wide format: rows = hospitalizations, columns = (channel,
  hour). Reshape into a numpy `(n_hosps, 24, n_channels)` tensor. Build the mask tensor in
  parallel (1 where the source had at least one row in that hour for that channel).

  For respiratory device flags, the "value" is itself binary; mask channel is constant 1
  for hours that have any respiratory_support row. Decision: include the indicator as a
  channel; skip the mask (the indicator itself already encodes presence).

  Forward-fill imputation lives in this module so the tensor is model-ready when it returns.
  The mask channel is computed BEFORE forward-fill so it reflects original observations.

  **Test scenarios:**
  - Happy path: toy cohort with 2 hospitalizations, hand-built values at known hours →
    tensor shape `(2, 24, n_channels)`, specific cell values match.
  - Window boundary: a row at exactly `anchor + 24h` is excluded.
  - Empty hospitalization: a cohort member with zero rows in window → tensor has all-null
    values for that hospitalization (forward-fill produces nulls, then 0), mask is all 0.
  - Channel ordering is deterministic: same input → same channel order across runs.
  - Mask correctness: a hospitalization that has heart_rate measurements only at hours
    1, 5, 10 → mask_hr is `[0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, ...]`.
  - Forward-fill leaves no leading nulls: even if first measurement is at hour 10, hours
    0-9 get the same value as hour 10 (or 0 if forward-fill can't seed). Document the
    choice in the docstring.

  **Verification:** `pytest tests/test_sequences.py -v` passes.

---

- U2. **`src/icumodelstream/torch_models.py` — LSTM + result dataclass**

  **Goal:** `LSTMBaseline` nn.Module + `SequenceResult` dataclass shaped like `BaselineResult`.

  **Requirements:** Reproducible, calibration-aware output.

  **Dependencies:** U1 produces the tensor shape this module consumes.

  **Files:**
  - `src/icumodelstream/torch_models.py` (new)
  - `tests/test_torch_models.py` (new, optional — model forward shape is easy to test
    without GPU)

  **Approach:** Define `LSTMBaseline(nn.Module)` with the architecture from the Key Technical
  Decisions table. Forward pass takes `(batch, timesteps, channels)`, returns
  `(batch,)` logits. No sigmoid in the model (loss applies it).

  Stub `TransformerBaseline(nn.Module)` — same input/output contract, raises
  `NotImplementedError` in `__init__` with a message pointing at the deferred-work note in
  this plan. Lets a follow-up implement it without redesigning the CLI surface.

  `SequenceResult` is a frozen dataclass with the same fields as `BaselineResult` plus
  `model_name="lstm" | "transformer"`, `epochs_trained`, `early_stopped_at_epoch`.

  **Test scenarios:**
  - Forward shape: input `(8, 24, 47)` → output `(8,)`.
  - Reproducibility: two `LSTMBaseline` instances with the same seed and same input produce
    identical outputs.
  - Stub raises: `TransformerBaseline()` raises NotImplementedError immediately.

  **Verification:** `pytest tests/test_torch_models.py -v` passes on CPU.

---

- U3. **`src/icumodelstream/torch_train.py` — Training loop + device dispatch**

  **Goal:** `fit_sequence_model(X, y, groups, device, seed) → (model, SequenceResult)`.

  **Requirements:** Reproducible, runs on CUDA/MPS/CPU, early stopping on validation AUROC.

  **Dependencies:** U1, U2.

  **Files:**
  - `src/icumodelstream/torch_train.py` (new)
  - `tests/test_torch_train.py` (new — synthetic data, 1 epoch, CPU)

  **Approach:** Use `group_kfold` from `splits.py` to produce train/val/test (70/15/15).
  Wrap tensors in `TensorDataset` + `DataLoader` (workers seeded with `worker_init_fn`).
  Train loop: forward → BCEWithLogitsLoss (with `pos_weight = (n_neg/n_pos)`) → backward →
  AdamW step. After each epoch, evaluate validation AUROC; track best; stop if no
  improvement for 3 epochs.

  Test-set evaluation uses the same `_compute_metrics` + `_calibration_table` from
  `models.py` (refactor those to top-level functions if currently private — they are
  already used by the LightGBM/logistic baselines).

  Device dispatch: `device = device_str if device_str else _autodetect_device()`. Autodetect
  picks `cuda` > `mps` > `cpu`. Log which device was chosen at the start of training.

  **Test scenarios:**
  - Synthetic dry-run: random `(50, 24, 47)` tensor + random labels → trains 1 epoch on
    CPU, loss is finite, model state dict can be saved/loaded.
  - Loss decreases: train 3 epochs with `device="cpu"` on a tiny synthetic problem where
    one channel is highly predictive of the label → loss epoch 0 > loss epoch 2.
  - Early stopping: validation AUROC plateau → training halts before max epochs.
  - Reproducibility: same seed + same data → same final test AUROC (within 1e-4 tolerance
    on CPU; CUDA non-determinism makes exact match harder, document it).
  - Save / load round trip: write state_dict to tmp_path, load into a fresh model, predict
    on same X → identical outputs.

  **Verification:** `pytest tests/test_torch_train.py -v` completes in < 30 seconds on Mac CPU.

---

- U4. **`src/icumodelstream/cli.py` — Add `sequence-baseline` command**

  **Goal:** New CLI command that runs cohort → labels → sequence tensors → train → eval and
  writes the same JSON / MD / model artifacts as `baseline`.

  **Requirements:** Mirror the existing `baseline` CLI signature; same JSON schema.

  **Dependencies:** U1, U2, U3.

  **Files:**
  - `src/icumodelstream/cli.py` (modify)
  - `tests/test_cli.py` (modify — add a smoke test that runs `sequence-baseline` on
    synthetic CLIF data, asserts JSON keys, no GPU required)

  **Approach:** New `@app.command()` named `sequence-baseline` with these flags inherited
  from `baseline`: `data_root`, `metrics_out`, `summary_out`, `model_out`, `min_age`,
  `require_icu_location`, `window_hours`, `seed`, `outcome`, `los_threshold_hours`. New
  flags: `--device {auto, cuda, mps, cpu}`, `--max-epochs INT` (default 20),
  `--patience INT` (default 3), `--batch-size INT` (default 256), `--learning-rate FLOAT`
  (default 1e-3).

  Reuse `_build_metrics_payload` and the markdown writer, just substituting model_name and
  adding the new training-config keys. The user can then `diff baseline_metrics.json
  sequence_metrics.json` to compare apples-to-apples.

  **Test scenarios:**
  - `icumodelstream sequence-baseline --help` exits 0 and lists the new flags.
  - Smoke test with `--device cpu` on the existing toy CLIF fixture → exits 0, writes
    JSON, JSON has the expected top-level keys.

  **Verification:** `pytest tests/test_cli.py -v` passes, including the new smoke test.

---

- U5. **`scripts/train_sequence.py` + `docs/plans/2026-05-25-002-rented-gpu-runbook.md`**

  **Goal:** A thin wrapper script + companion runbook that makes the rented-GPU step
  copy-paste from the runbook.

  **Requirements:** The user must be able to follow the runbook on a fresh RunPod / Lambda /
  Modal pod and have training kicked off in < 10 minutes.

  **Dependencies:** U1-U4.

  **Files:**
  - `scripts/train_sequence.py` (new — a CLI wrapper that calls
    `icumodelstream.cli.sequence_baseline` programmatically, useful for cron / SSH)
  - `docs/plans/2026-05-25-002-rented-gpu-runbook.md` (new companion doc)

  **Runbook structure:**
  - GPU provider comparison (RunPod / Lambda / Modal) with current pricing
  - Recommended choice + why
  - Pod setup commands (clone repo, `uv sync`, mount data, etc.)
  - The single training command to run
  - Cost estimate per training run
  - Smoke-test: a 1-epoch dry-run on the GPU to confirm CUDA is alive before the real
    training fires
  - What to download when training completes (just the .pt + JSON, never the data)
  - Teardown commands so the rental doesn't leak

  **Test scenarios:** Test expectation: none — this is documentation + a thin wrapper.
  Validation is "a different person can follow the runbook end-to-end."

  **Verification:** The runbook contains every command a user needs; the wrapper script
  works locally via `python scripts/train_sequence.py --device cpu --dry-run`.

---

- U6. **`pyproject.toml` — Add `[ml-dl]` extras**

  **Goal:** Pull torch only when the user actually wants the sequence baseline.

  **Files:**
  - `pyproject.toml` (modify)
  - `Makefile` (modify — new `install-dl` target that installs the dl extras)
  - `README.md` (modify — document the dl install step)

  **Approach:** Add a `ml-dl` group to `[project.optional-dependencies]`:
  ```toml
  ml-dl = [
      "torch>=2.3",
  ]
  ```
  Keep it separate from `ml` so clinician-engineers who only run the LightGBM baseline
  don't pull 3GB of CUDA wheels. Document the install path in README ("for the sequence
  baseline, run `make install-dl` after `make install`").

  Pin `torch>=2.3` — versions before 2.3 had MPS issues on Apple Silicon.

  **Test scenarios:** Test expectation: none — config change.

  **Verification:** `pip install -e '.[ml-dl]'` resolves torch successfully and
  `python -c "import torch; print(torch.__version__)"` works.

---

## System-Wide Impact

| Surface | Impact |
|---|---|
| `src/icumodelstream/` | 3 new modules (`sequences.py`, `torch_models.py`, `torch_train.py`); `cli.py` gains one command |
| `tests/` | 2-3 new test files; existing tests untouched |
| `scripts/` | 1 new file (`train_sequence.py`); existing scripts untouched |
| `docs/plans/` | 1 new companion runbook |
| `pyproject.toml` | New `ml-dl` extras group |
| `Makefile` | New `install-dl` target |
| `README.md` | Documents the optional dl install step + the `sequence-baseline` command |
| `.gitignore` | No change — `models/` and `reports/` are already covered |

Affected parties:
- **The user** — needs to pick a GPU provider and pay for time. Runbook makes the choice
  concrete and cheap to verify.
- **A future contributor** — sees a clear path: LSTM is implemented; transformer is
  scaffolded; foundation model is documented as deferred. No fishing-around required.

---

## Risks & Dependencies

| Risk | Severity | Mitigation |
|---|---|---|
| LSTM doesn't beat LightGBM | **Medium** | Plan explicitly accepts that as a result — documents the finding, defers transformer work until clearly warranted |
| Training cost blows up | **Medium** | Cost estimate in runbook (< $1 per training run on RunPod A10 spot); enforce `--max-epochs` cap; dry-run smoke test on GPU before real training |
| MPS / CPU dry-run diverges from CUDA results | **Low** | Dry-run is for code-correctness only; CUDA non-determinism is documented; reproducibility is checked on CUDA only |
| Sequence tensor exceeds memory | **Low** | ~430 MB total; fits easily in 24GB VRAM. Worst case use `DataLoader(num_workers=...)` and stream batches |
| Training script doesn't transfer to rented GPU | **Medium** | Runbook step-by-step + smoke test catch this; the wrapper script is intentionally provider-agnostic |
| PHI leakage in PyTorch model weights | **Low** | LSTM weights cannot recover raw patient data; still gitignore `models/*.pt` (already done) and never upload them to public model registries |
| `torch>=2.3` dep clashes with the existing `numpy>=1.26` | **Low** | Verified: torch 2.3+ supports numpy 1.26. Pin both via `uv.lock` |
| User has no GPU access | **Critical for execution, not for the plan** | This plan can land entirely; actual training requires user action. Runbook spells out the spot rental option (cheapest path) |

---

## Alternative Approaches Considered

**Foundation-model fine-tune as the first move.** Med-LLaMA, BioGPT, ClinicalBERT — all
plausible. Rejected as the FIRST approach because (a) the LSTM has a much lower failure
surface (we know how it should behave; we don't know how a foundation model will react to
our channel structure), (b) cost per training run is 10-100× higher, (c) interpretation
work after the fact is harder. Right answer: LSTM first; if LSTM doesn't beat LightGBM,
attempt transformer-from-scratch; if THAT doesn't either, then escalate to foundation
model with confidence that the simpler models genuinely couldn't.

**Transformer-from-scratch as the first move.** Plausible alternative to LSTM — modern
ICU papers often use transformers. Rejected because the dataset is small enough
(85K hospitalizations × 24 timesteps) that attention's relative-position advantages may
not show up. LSTM is the tighter baseline. The transformer scaffold lives in
`torch_models.py` so a follow-up can implement it without a new plan.

**Variable-length sequences instead of fixed 24-hour windows.** Plausible — some
hospitalizations have hours of data; some have days. Rejected because (a) the LightGBM
comparison is at 24h, (b) the dataset includes hospitalizations as short as 6 hours and as
long as 30 days; modeling variable length adds complexity (padding masks, attention masks)
that's not justified yet. Future plan can extend.

**Train on Mac MPS instead of rented CUDA.** Plausible — M4 Pro has MPS, MPS works for
basic torch ops. Rejected because (a) MPS still has gaps in operators that LSTMs need; (b)
training time on M4 vs A10 is ~10× slower; (c) CLAUDE.md explicitly says "rented-GPU
scripts" for Phase 5. MPS is fine for dry runs only.

---

## Documentation Notes

- `docs/data_dictionary_notes.md` is the authoritative source for channel names; do not
  hardcode them in `sequences.py` — import the `RICH_*` constants from `pipeline.py`.
- `docs/roadmap.md` Phase 5 row should be marked "in progress" when U1 lands and "done"
  when U4 (CLI command) lands. Track the empirical result in the same row.
- The companion runbook (`2026-05-25-002-rented-gpu-runbook.md`) goes in the same
  directory as plans because it IS a plan-shaped artifact — a step-by-step kickoff
  document that another contributor could follow.
- Each new module gets a docstring citing CLAUDE.md §9 (baselines before deep learning)
  and the comparison to LightGBM 0.866.

---

## Sources & References

- `docs/roadmap.md` — Phase 5 row + gate
- `CLAUDE.md` — §2 (simplicity), §8 (no patient leakage), §9 (baselines before DL), §10
  (reproducibility), and the "Local machine boundaries" paragraph stating GPU work is
  rented
- `docs/plans/2026-05-24-002-feat-lightgbm-baseline-phase4-plan.md` — immediate
  predecessor; the LightGBM 0.866 baseline is the bar
- `docs/data_dictionary_notes.md` — channel vocabulary
- `src/icumodelstream/pipeline.py` — `RICH_*` constants are the channel source of truth
- `src/icumodelstream/models.py` — `_compute_metrics` and `_calibration_table` are
  reused so the sequence model's metrics dict matches Phase 4 exactly
- `src/icumodelstream/splits.py` — `group_kfold` enforces the train/val/test split contract
- [PyTorch docs on `nn.LSTM`](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html)
- [RunPod pricing page](https://runpod.io/console/gpu-secure-cloud) — referenced in runbook
- [Lambda Labs on-demand](https://lambdalabs.com/service/gpu-cloud) — referenced in runbook
