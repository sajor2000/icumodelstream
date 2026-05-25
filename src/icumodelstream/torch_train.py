"""Training loop for the Phase 5 LSTM sequence baseline.

Sprint 3a (this file): data-prep helper that converts SequenceTensors
+ labels into patient-aware 70/15/15 torch tensors.
Sprints 3b/3c add the training loop and the public fit_sequence_model.

CLAUDE.md rule 8 (no silent patient leakage): the split is by
hospitalization-level patient_id; group_train_test_split guarantees a patient
appears in only one fold.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
import torch

from icumodelstream.sequences import SequenceTensors
from icumodelstream.splits import group_train_test_split


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
