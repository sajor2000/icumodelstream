"""Training loop for the Phase 5 LSTM sequence baseline.

Sprint 3a: data-prep helper that converts SequenceTensors + labels into
patient-aware 70/15/15 torch tensors.
Sprint 3b (this file, additions below): ``train_lstm`` + ``TrainingTrace``.
Sprint 3c will add the public ``fit_sequence_model`` and metrics/calibration.

CLAUDE.md rule 8 (no silent patient leakage): the split is by
hospitalization-level patient_id; group_train_test_split guarantees a patient
appears in only one fold.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from icumodelstream.models import calibration_table, compute_metrics
from icumodelstream.sequences import SequenceTensors
from icumodelstream.splits import group_train_test_split
from icumodelstream.torch_models import LSTMBaseline, SequenceResult


@dataclass(frozen=True)
class SplitTensors:
    """Pre-split torch tensors ready for training.

    Each X is shape (n_split, timesteps, channels), float32.
    Each y is shape (n_split,), float32 in {0.0, 1.0}.
    """

    X_train: torch.Tensor
    y_train: torch.Tensor
    X_val: torch.Tensor
    y_val: torch.Tensor
    X_test: torch.Tensor
    y_test: torch.Tensor
    n_train_patients: int
    n_val_patients: int
    n_test_patients: int


# Single-column outcome candidates checked when ``labels`` lacks an "outcome"
# column. Order matches the label builders in icumodelstream.labels.
_LABEL_FALLBACKS = ("mortality", "long_los")


def _pick_label_column(labels: pl.DataFrame) -> str:
    """Return the name of the label column to use.

    Priority: exact "outcome" column wins; otherwise look for a single
    non-id column matching one of the known label names. Fail loudly if
    neither is present (CLAUDE.md rule 7).
    """
    columns = labels.columns
    if "outcome" in columns:
        return "outcome"
    for candidate in _LABEL_FALLBACKS:
        if candidate in columns:
            return candidate
    raise ValueError(
        "labels DataFrame must contain an 'outcome' column or one of "
        f"{_LABEL_FALLBACKS}; got columns={columns}."
    )


def prepare_split_tensors(
    sequences: SequenceTensors,
    labels: pl.DataFrame,
    groups: pl.DataFrame,
    seed: int = 42,
    test_size: float = 0.15,
    val_size: float = 0.15,
) -> SplitTensors:
    """Align labels + groups to SequenceTensors, then patient-aware-split into 70/15/15.

    Parameters
    ----------
    sequences:
        Output of icumodelstream.sequences.build_sequence_tensors. The order of
        sequences.hospitalization_ids determines the alignment.
    labels:
        DataFrame with hospitalization_id and an integer 0/1 outcome column
        named "outcome" (or "mortality" or "long_los"; whichever single non-id
        column is present).
    groups:
        DataFrame with hospitalization_id and patient_id. Used as the grouping
        key for the patient-aware split.
    seed:
        Random seed for both splits (val carved out of train using ``seed + 1``
        so the val/train split is independent of the test/rest split).
    test_size, val_size:
        Fractions of the FULL dataset that go to test and val respectively.
        Train gets 1 - test_size - val_size.

    Returns
    -------
    SplitTensors with X/y as torch tensors on CPU.

    Raises
    ------
    ValueError
        On length mismatch between sequences and labels/groups, or unknown label column.
    """
    n = sequences.X.shape[0]
    if n == 0:
        raise ValueError("sequences.X is empty; cannot build split tensors.")

    label_col = _pick_label_column(labels)

    # Align labels + groups to the sequence order via inner-join on hospitalization_id.
    # The join is the load-bearing alignment: a positional zip would silently misalign
    # if labels/groups arrive in a different order than sequences.hospitalization_ids.
    seq_order = pl.DataFrame(
        {
            "hospitalization_id": list(sequences.hospitalization_ids),
            "_seq_idx": list(range(n)),
        }
    )
    aligned = (
        seq_order.join(
            labels.select(["hospitalization_id", label_col]),
            on="hospitalization_id",
            how="inner",
        )
        .join(
            groups.select(["hospitalization_id", "patient_id"]),
            on="hospitalization_id",
            how="inner",
        )
        .sort("_seq_idx")
    )

    if aligned.height != n:
        raise ValueError(
            "Alignment failure: sequences has "
            f"{n} hospitalizations but inner-join with labels+groups produced "
            f"{aligned.height} rows. Every sequences.hospitalization_ids entry "
            "must appear exactly once in both labels and groups."
        )

    # Pull aligned arrays. _seq_idx preserves the original sequence row order so X
    # rows line up with y and patient_ids by simple positional indexing.
    seq_idx = aligned["_seq_idx"].to_numpy()
    y_np = aligned[label_col].to_numpy().astype(np.int8)
    patient_ids = aligned["patient_id"].to_numpy()

    X_np = sequences.X[seq_idx]  # reorder X to match aligned row order

    # ------------------------------------------------------------------
    # Two-stage patient-aware split.
    #
    # Stage 1 carves the test fold from the full pool.
    # Stage 2 carves the val fold from what remains, with the val fraction
    # rescaled so it lands at the requested GLOBAL fraction:
    #   val_frac_within_remaining = val_size / (1 - test_size)
    # Seeds: test split uses ``seed``, val split uses ``seed + 1`` so the two
    # stages are independent (avoids hidden coupling between the two folds).
    # ------------------------------------------------------------------
    indices = np.arange(n, dtype=np.int64)
    idx_df = pl.DataFrame({"idx": indices.tolist()})
    y_series = pl.Series("y", y_np.tolist())
    groups_series = pl.Series("patient_id", patient_ids.tolist())

    rest_idx_df, test_idx_df, _, _ = group_train_test_split(
        idx_df, y_series, groups_series, test_size=test_size, seed=seed
    )
    test_idx = np.asarray(test_idx_df["idx"].to_list(), dtype=np.int64)
    rest_idx = np.asarray(rest_idx_df["idx"].to_list(), dtype=np.int64)

    val_frac = val_size / (1.0 - test_size)
    rest_y = pl.Series("y", y_np[rest_idx].tolist())
    rest_groups = pl.Series("patient_id", patient_ids[rest_idx].tolist())
    rest_idx_for_split = pl.DataFrame({"idx": rest_idx.tolist()})

    train_idx_df, val_idx_df, _, _ = group_train_test_split(
        rest_idx_for_split,
        rest_y,
        rest_groups,
        test_size=val_frac,
        seed=seed + 1,
    )
    train_idx = np.asarray(train_idx_df["idx"].to_list(), dtype=np.int64)
    val_idx = np.asarray(val_idx_df["idx"].to_list(), dtype=np.int64)

    # ------------------------------------------------------------------
    # Slice arrays and convert to torch tensors. X is already float32 per the
    # SequenceTensors contract; .float() is a cheap no-op safety net so callers
    # passing in float64 arrays still get the right dtype. y is float32 to
    # match BCEWithLogitsLoss expectations (Sprint 3b).
    # ------------------------------------------------------------------
    def _to_tensor_x(idx: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.ascontiguousarray(X_np[idx])).float()

    def _to_tensor_y(idx: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(y_np[idx].astype(np.float32))

    X_train = _to_tensor_x(train_idx)
    X_val = _to_tensor_x(val_idx)
    X_test = _to_tensor_x(test_idx)
    y_train = _to_tensor_y(train_idx)
    y_val = _to_tensor_y(val_idx)
    y_test = _to_tensor_y(test_idx)

    n_train_patients = len(set(patient_ids[train_idx].tolist()))
    n_val_patients = len(set(patient_ids[val_idx].tolist()))
    n_test_patients = len(set(patient_ids[test_idx].tolist()))

    return SplitTensors(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        n_train_patients=n_train_patients,
        n_val_patients=n_val_patients,
        n_test_patients=n_test_patients,
    )


# ---------------------------------------------------------------------------
# Sprint 3b: training loop with early stopping.
#
# CLAUDE.md rule 4 (goal-driven execution): each piece of train_lstm has a
# concrete reason -- pos_weight handles class imbalance without inflating the
# predicted probability scale (the trap we hit earlier with LightGBM's
# is_unbalance=True), AdamW + weight_decay is the safe default for sequence
# models, and early stopping on val AUROC keeps the best checkpoint so
# inference uses it instead of the final (potentially overfit) epoch.
# CLAUDE.md rule 7 (fail loudly): degenerate single-class val sets are
# handled explicitly with a warning rather than letting sklearn raise mid-run.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrainingTrace:
    """What happened during training; for downstream logging + the SequenceResult.

    Attributes
    ----------
    epochs_trained:
        Number of completed epochs (1-based count). If early stopping fires
        after epoch ``k``, this equals ``k`` (not ``max_epochs``).
    early_stopped_at_epoch:
        1-based epoch index where the last improvement happened, or ``None``
        if the loop ran out to ``max_epochs`` without triggering the patience
        counter. Used by Sprint 3c to populate ``SequenceResult.early_stopped_at_epoch``.
    best_val_auroc:
        Validation AUROC at the best-AUROC epoch (also the AUROC of the
        weights ``model`` carries on return).
    final_train_loss:
        Mean training loss over the final completed epoch.
    final_val_loss:
        Mean validation loss over the final completed epoch.
    """

    epochs_trained: int
    early_stopped_at_epoch: int | None
    best_val_auroc: float
    final_train_loss: float
    final_val_loss: float


def _safe_roc_auc(y_true: np.ndarray, y_score: np.ndarray, verbose: bool) -> float:
    """Compute AUROC, degrading to 0.5 if the validation set has a single class.

    sklearn raises ``ValueError`` when ``y_true`` is constant; catching that
    here keeps the training loop robust on tiny val folds and surfaces the
    degeneracy with a one-time warning (gated on ``verbose`` so tests stay
    quiet but real runs still see the signal via stdout).
    """
    if len(np.unique(y_true)) < 2:
        if verbose:
            print(
                "WARNING: validation set has a single class; using val_auroc=0.5 "
                "as a placeholder. Best-checkpoint selection will degrade to "
                "tracking the first epoch."
            )
        return 0.5
    return float(roc_auc_score(y_true, y_score))


def train_lstm(
    model: nn.Module,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    max_epochs: int = 20,
    patience: int = 3,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 256,
    device: str = "cpu",
    seed: int = 42,
    verbose: bool = False,
) -> TrainingTrace:
    """Train an LSTMBaseline with BCE-with-logits + pos_weight + AdamW + early stopping.

    The function MUTATES ``model``: at the end it loads the state dict from
    the epoch with the highest validation AUROC so downstream inference uses
    the best checkpoint, not the last. Returns a :class:`TrainingTrace` with
    enough information for Sprint 3c to populate :class:`SequenceResult`.

    Reproducibility: ``torch.manual_seed(seed)`` and ``numpy.random.seed(seed)``
    are set up front; the DataLoader gets a seeded ``torch.Generator`` and a
    ``worker_init_fn`` that seeds each worker as ``seed + worker_id``. On CPU
    this is bit-exact across runs; on CUDA there is residual non-determinism
    from cuDNN that is out of scope for this baseline.

    Parameters
    ----------
    model:
        ``nn.Module`` (typically :class:`icumodelstream.torch_models.LSTMBaseline`)
        whose ``forward`` returns ``(batch,)`` raw logits.
    X_train, y_train, X_val, y_val:
        Tensors from :func:`prepare_split_tensors`. ``X`` is
        ``(n, timesteps, channels)`` float32; ``y`` is ``(n,)`` float32 0/1.
    max_epochs:
        Hard cap on training epochs.
    patience:
        Early-stopping patience measured in epochs without val-AUROC improvement.
    learning_rate, weight_decay:
        Passed to ``AdamW``. ``weight_decay=1e-4`` is the conservative
        baseline; tune later if validation curves call for it.
    batch_size:
        Mini-batch size for both train and val loaders.
    device:
        ``"cpu"``, ``"mps"``, or ``"cuda"``. Tensors and the model are moved
        to this device for the duration of training.
    seed:
        Seed for torch + numpy + DataLoader shuffles.
    verbose:
        If True, print degenerate-val warnings. Default False so tests stay clean.

    Returns
    -------
    TrainingTrace
        Trace of the run; ``model`` itself is mutated in place.

    Notes
    -----
    Single-class validation sets (e.g. tiny folds with no positives) are
    handled by returning ``val_auroc=0.5``; in that case best-checkpoint
    selection effectively falls back to the first epoch and the caller
    should treat ``best_val_auroc`` as unreliable.
    """
    # Seeded RNG -- set BEFORE any tensor creation so DataLoader generators
    # and any internal randomness in model init (none here, but defensive)
    # share the same starting state across runs.
    torch.manual_seed(seed)
    np.random.seed(seed)

    torch_device = torch.device(device)
    model.to(torch_device)
    X_train_d = X_train.to(torch_device)
    y_train_d = y_train.to(torch_device)
    X_val_d = X_val.to(torch_device)
    y_val_d = y_val.to(torch_device)

    # Class-imbalance handling via pos_weight on BCEWithLogitsLoss. This
    # reweights the gradient contribution of positive examples WITHOUT
    # rescaling the predicted probabilities -- crucial for keeping calibration
    # intact downstream. If y_train has no positives or no negatives the
    # division blows up; clamp denominator to 1 and let the loss degenerate
    # gracefully (the training set should never be single-class in practice,
    # but a defensive guard here is cheap).
    n_pos = float((y_train_d == 1).sum().item())
    n_neg = float((y_train_d == 0).sum().item())
    pos_weight_value = n_neg / max(n_pos, 1.0)
    pos_weight = torch.tensor(pos_weight_value, dtype=torch.float32, device=torch_device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    # Seeded DataLoader: a torch.Generator for shuffle order + a worker_init_fn
    # so that any numpy randomness inside workers is also deterministic.
    loader_generator = torch.Generator()
    loader_generator.manual_seed(seed)

    def _worker_init_fn(worker_id: int) -> None:
        np.random.seed(seed + worker_id)

    train_dataset = TensorDataset(X_train_d, y_train_d)
    val_dataset = TensorDataset(X_val_d, y_val_d)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=loader_generator,
        worker_init_fn=_worker_init_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    best_val_auroc = -float("inf")
    best_state_dict: dict[str, torch.Tensor] | None = None
    best_epoch: int | None = None
    epochs_since_improvement = 0

    epochs_trained = 0
    final_train_loss = float("nan")
    final_val_loss = float("nan")
    early_stopped_at_epoch: int | None = None

    for epoch in range(1, max_epochs + 1):
        # ------------------------------------------------------------------
        # Train phase. Use sum of (loss * batch_size) / total_count so the
        # reported mean loss is independent of the final partial-batch size.
        # ------------------------------------------------------------------
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            bs = y_batch.shape[0]
            train_loss_sum += float(loss.item()) * bs
            train_count += bs
        train_loss = train_loss_sum / max(train_count, 1)

        # ------------------------------------------------------------------
        # Val phase. Concatenate per-batch logits so the AUROC sees the full
        # val set in one shot, and compute the loss on the same concatenated
        # tensor so train/val losses are comparable.
        # ------------------------------------------------------------------
        model.eval()
        all_logits: list[torch.Tensor] = []
        with torch.no_grad():
            for X_batch, _ in val_loader:
                all_logits.append(model(X_batch))
        val_logits = torch.cat(all_logits, dim=0)
        val_loss = float(criterion(val_logits, y_val_d).item())
        val_probs = torch.sigmoid(val_logits).cpu().numpy()
        val_auroc = _safe_roc_auc(y_val.cpu().numpy(), val_probs, verbose=verbose)

        epochs_trained = epoch
        final_train_loss = train_loss
        final_val_loss = val_loss

        # ------------------------------------------------------------------
        # Early stopping on val AUROC. Stash a CPU-resident copy of the best
        # state dict so we don't keep extra GPU/MPS allocations around for
        # checkpoints we may never restore.
        # ------------------------------------------------------------------
        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            best_state_dict = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            best_epoch = epoch
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1
            if epochs_since_improvement >= patience:
                early_stopped_at_epoch = best_epoch
                break

    # Restore the best checkpoint so the caller's ``model`` is the
    # best-AUROC version, not whatever the last epoch produced.
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    # If we never improved past the sentinel (e.g. single-class val + every
    # epoch tied at 0.5), record the achieved AUROC honestly rather than -inf.
    if best_val_auroc == -float("inf"):
        best_val_auroc = 0.5

    return TrainingTrace(
        epochs_trained=epochs_trained,
        early_stopped_at_epoch=early_stopped_at_epoch,
        best_val_auroc=float(best_val_auroc),
        final_train_loss=float(final_train_loss),
        final_val_loss=float(final_val_loss),
    )


# ---------------------------------------------------------------------------
# Sprint 3c: public entry point that wires Sprints 3a + 3b together.
#
# CLAUDE.md rule 2 (simplicity first): no new training logic lives here -- this
# function only sequences the existing pieces (prepare_split_tensors ->
# LSTMBaseline -> train_lstm -> predict -> compute_metrics/calibration_table)
# and packages the result. The next unit (U4) wraps this in a CLI command.
# CLAUDE.md rule 8 (no silent patient leakage): test-set predictions and
# metrics are computed AFTER the patient-aware split inside
# prepare_split_tensors, so the test fold is unseen by both train and val.
# ---------------------------------------------------------------------------


def _autodetect_device() -> str:
    """Return ``'cuda'`` if available, else ``'mps'`` if available, else ``'cpu'``.

    Pure detection: no side effects, no environment-variable overrides. The
    public ``fit_sequence_model`` calls this only when its ``device`` argument
    is ``None``; passing an explicit string (e.g. ``"cpu"`` to force CPU on a
    CUDA host) skips this entirely.
    """
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def fit_sequence_model(
    sequences: SequenceTensors,
    labels: pl.DataFrame,
    groups: pl.DataFrame,
    *,
    hidden_dim: int = 128,
    n_layers: int = 2,
    dropout: float = 0.3,
    max_epochs: int = 20,
    patience: int = 3,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 256,
    device: str | None = None,
    seed: int = 42,
) -> tuple[LSTMBaseline, SequenceResult]:
    """End-to-end: SequenceTensors + labels -> trained LSTM + SequenceResult.

    Wires the three Sprint-3 pieces together:

    1. :func:`prepare_split_tensors` produces a patient-aware 70/15/15 split
       (Sprint 3a). This is where the no-leakage guarantee lives.
    2. :class:`icumodelstream.torch_models.LSTMBaseline` is constructed with
       ``input_dim`` inferred from ``sequences.X.shape[2]``.
    3. :func:`train_lstm` fits the model with BCE+pos_weight + AdamW + early
       stopping (Sprint 3b) and mutates ``model`` so it carries the best-AUROC
       weights on return.
    4. The fitted model scores the test fold; sigmoid + numpy conversion give
       probabilities.
    5. :func:`icumodelstream.models.compute_metrics` and
       :func:`icumodelstream.models.calibration_table` produce the same metric
       dict and decile-binned calibration table as the flat baselines, so
       downstream reporting code does not need to branch on model family.

    Parameters
    ----------
    sequences:
        Per-hospitalization tensors from
        :func:`icumodelstream.sequences.build_sequence_tensors`.
    labels:
        DataFrame with ``hospitalization_id`` + an integer 0/1 outcome column
        (``outcome``, ``mortality``, or ``long_los``).
    groups:
        DataFrame with ``hospitalization_id`` + ``patient_id`` used as the
        grouping key for the patient-aware split.
    hidden_dim, n_layers, dropout:
        :class:`LSTMBaseline` architecture hyperparameters.
    max_epochs, patience, learning_rate, weight_decay, batch_size:
        :func:`train_lstm` hyperparameters.
    device:
        ``"cpu"``, ``"mps"``, ``"cuda"``, or ``None``. When ``None`` the
        function picks the best available device via :func:`_autodetect_device`;
        pass an explicit string to override (e.g. ``"cpu"`` for reproducible
        CPU runs on a CUDA host).
    seed:
        Forwarded to both ``prepare_split_tensors`` and ``train_lstm`` so the
        full pipeline is reproducible on CPU.

    Returns
    -------
    tuple[LSTMBaseline, SequenceResult]
        The fitted model (best-AUROC weights loaded) and a frozen result
        dataclass with the same shape as :class:`icumodelstream.models.BaselineResult`
        plus ``epochs_trained`` + ``early_stopped_at_epoch``.
    """
    tensors = prepare_split_tensors(sequences, labels, groups, seed=seed)
    resolved_device = device if device is not None else _autodetect_device()

    # Seed BEFORE constructing the model so LSTM weight init is deterministic.
    # train_lstm re-seeds again internally for the optimizer + DataLoader,
    # but by then the weights already exist; without this seed the model's
    # initial parameters would carry forward whatever global RNG state the
    # caller happened to leave behind, breaking end-to-end reproducibility.
    torch.manual_seed(seed)
    np.random.seed(seed)

    input_dim = sequences.X.shape[2]
    model = LSTMBaseline(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        dropout=dropout,
    )

    trace = train_lstm(
        model,
        tensors.X_train,
        tensors.y_train,
        tensors.X_val,
        tensors.y_val,
        max_epochs=max_epochs,
        patience=patience,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        batch_size=batch_size,
        device=resolved_device,
        seed=seed,
    )

    # Test-set inference. The model is on ``resolved_device`` after
    # ``train_lstm``; move X_test there too and pull the resulting probabilities
    # back to CPU/numpy for the metrics layer (sklearn doesn't accept torch
    # tensors). model.eval() disables dropout so test scores are deterministic.
    torch_device = torch.device(resolved_device)
    model.eval()
    with torch.no_grad():
        X_test_dev = tensors.X_test.to(torch_device)
        test_logits = model(X_test_dev)
        y_pred_proba = torch.sigmoid(test_logits).detach().cpu().numpy()

    y_true = tensors.y_test.detach().cpu().numpy().astype(int)
    metrics = compute_metrics(y_true, y_pred_proba)
    calib = calibration_table(y_true, y_pred_proba)

    result = SequenceResult(
        model_name="lstm",
        y_true=y_true,
        y_pred_proba=y_pred_proba,
        metrics=metrics,
        calibration_table=calib,
        epochs_trained=trace.epochs_trained,
        early_stopped_at_epoch=trace.early_stopped_at_epoch,
    )
    return model, result
