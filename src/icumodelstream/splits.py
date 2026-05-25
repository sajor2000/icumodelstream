"""Group-aware train/test splits that prevent patient leakage.

Implements CLAUDE.md section 8 ("No silent patient leakage"). The contract these
helpers enforce is:

    After splitting, the intersection of groups[train_idx] and groups[test_idx]
    MUST be empty.

For ICU prediction tasks, the group key is typically ``patient_id`` (a single
patient can have multiple hospitalizations and many rows of vitals/labs). If
those rows leak across train and test, evaluation metrics are optimistically
biased. Always pass the patient identifier as ``groups``.

Both functions wrap scikit-learn:

* ``group_train_test_split`` -> ``sklearn.model_selection.GroupShuffleSplit``
  (honors ``seed`` via ``random_state``).
* ``group_kfold`` -> ``sklearn.model_selection.GroupKFold`` which is
  deterministic by design and does NOT take a seed.
"""

from __future__ import annotations

import hashlib

import numpy as np
import polars as pl
from sklearn.model_selection import GroupKFold, GroupShuffleSplit


def _validate_lengths(X: pl.DataFrame, y: pl.Series, groups: pl.Series) -> None:
    if len(X) == 0:
        raise ValueError("Cannot split empty data: len(X) == 0")
    if len(X) != len(y):
        raise ValueError(
            f"Length mismatch: len(X)={len(X)} but len(y)={len(y)}. "
            f"X and y must have the same number of rows."
        )
    if len(X) != len(groups):
        raise ValueError(
            f"Length mismatch: len(X)={len(X)} but len(groups)={len(groups)}. "
            f"X and groups must have the same number of rows."
        )


def group_train_test_split(
    X: pl.DataFrame,
    y: pl.Series,
    groups: pl.Series,
    test_size: float = 0.2,
    seed: int = 42,
    stratify: bool = False,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.Series, pl.Series]:
    """Split (X, y) into train/test with no group shared across both sides.

    Wraps :class:`sklearn.model_selection.GroupShuffleSplit`. The ``seed``
    argument is forwarded as ``random_state``, so the same seed reproduces the
    same split.

    Parameters
    ----------
    X:
        Feature DataFrame, one row per sample.
    y:
        Target Series aligned to X.
    groups:
        Group key Series aligned to X (typically ``patient_id``). Any polars
        dtype is accepted; values are only compared for equality. In the
        CLIF-MIMIC dataset ``patient_id`` is ``String``.
    test_size:
        Approximate fraction of groups (not rows) assigned to the test set.
    seed:
        Random seed forwarded to ``GroupShuffleSplit(random_state=seed)``.

    Returns
    -------
    tuple
        ``(X_train, X_test, y_train, y_test)``.

    Raises
    ------
    ValueError
        If ``X``, ``y``, and ``groups`` do not share the same length, or if
        ``X`` is empty (CLAUDE.md rule 7: fail loudly on data assumptions).
    """
    _validate_lengths(X, y, groups)

    groups_np = groups.to_numpy()

    # Single-group edge case: GroupShuffleSplit refuses (one group cannot be split
    # in two without leakage). The no-leakage contract is still satisfied if we
    # put every row on one side. Convention: send everything to train, leave test
    # empty. The caller gets a clear, contract-honoring result and can decide
    # whether a one-group dataset is meaningful for evaluation.
    if len(np.unique(groups_np)) == 1:
        empty_idx = np.array([], dtype=np.int64)
        all_idx = np.arange(len(X), dtype=np.int64)
        return X[all_idx], X[empty_idx], y[all_idx], y[empty_idx]

    if stratify:
        # Stratified-group split: pick the split (out of a small number of candidates)
        # whose test fold's positive prevalence is closest to the global prevalence
        # AND has at least one positive. Necessary at low prevalence where a single
        # GroupShuffleSplit can produce an all-negative test fold (finding #3).
        y_np = y.to_numpy()
        global_prev = float(y_np.mean())
        best = None
        best_score = float("inf")
        for candidate_seed in range(seed, seed + 32):
            splitter = GroupShuffleSplit(
                n_splits=1, test_size=test_size, random_state=candidate_seed
            )
            tr, te = next(splitter.split(np.zeros(len(X)), groups=groups_np))
            if len(te) == 0 or y_np[te].sum() == 0:
                continue
            te_prev = float(y_np[te].mean())
            score = abs(te_prev - global_prev)
            if score < best_score:
                best = (tr, te)
                best_score = score
        if best is None:
            raise ValueError(
                "stratify=True: could not find a group split with any positives in the "
                "test fold after 32 seed attempts. Cohort is too small or prevalence "
                "too low; reduce test_size or fall back to stratify=False."
            )
        train_idx, test_idx = best
    else:
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        # GroupShuffleSplit only inspects len(X) and groups; a placeholder X is fine.
        train_idx, test_idx = next(splitter.split(np.zeros(len(X)), groups=groups_np))

    X_train = X[train_idx]
    X_test = X[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]
    return X_train, X_test, y_train, y_test


def group_kfold(
    X: pl.DataFrame,
    y: pl.Series,
    groups: pl.Series,
    n_splits: int = 5,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate group-aware K-fold splits.

    Wraps :class:`sklearn.model_selection.GroupKFold`. GroupKFold is
    deterministic by construction and does NOT accept a random seed; calling
    it twice with the same inputs always returns the same folds.

    Parameters
    ----------
    X:
        Feature DataFrame, one row per sample.
    y:
        Target Series aligned to X.
    groups:
        Group key Series aligned to X (typically ``patient_id``).
    n_splits:
        Number of folds. Must be <= number of unique groups.

    Returns
    -------
    list of tuple
        List of ``(train_idx, test_idx)`` numpy arrays, one per fold.

    Raises
    ------
    ValueError
        If ``X``/``y``/``groups`` lengths mismatch, if ``X`` is empty, or if
        the number of unique groups is smaller than ``n_splits``.
    """
    _validate_lengths(X, y, groups)

    groups_np = groups.to_numpy()
    n_unique_groups = len(np.unique(groups_np))
    if n_unique_groups < n_splits:
        raise ValueError(
            f"Cannot create {n_splits} folds from {n_unique_groups} unique groups. "
            f"Reduce n_splits or provide more groups."
        )

    splitter = GroupKFold(n_splits=n_splits)
    return [
        (train_idx, test_idx)
        for train_idx, test_idx in splitter.split(np.zeros(len(X)), groups=groups_np)
    ]


def stable_fraction(value: object, seed: str = "icumodelstream-v1") -> float:
    """Map an identifier to a deterministic fraction in [0, 1)."""
    digest = hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()
    return int(digest[:16], 16) / float(16**16)


def assign_split(
    value: object, train: float = 0.70, validation: float = 0.15, seed: str = "icumodelstream-v1"
) -> str:
    """Assign a stable train/validation/test split from a patient or hospitalization ID."""
    if train <= 0 or validation <= 0 or train + validation >= 1:
        raise ValueError("Expected train > 0, validation > 0, and train + validation < 1.")
    frac = stable_fraction(value, seed=seed)
    if frac < train:
        return "train"
    if frac < train + validation:
        return "validation"
    return "test"


def add_stable_split(
    frame: pl.DataFrame,
    id_col: str,
    split_col: str = "split",
    train: float = 0.70,
    validation: float = 0.15,
    seed: str = "icumodelstream-v1",
) -> pl.DataFrame:
    """Add a deterministic split column without using row order."""
    if id_col not in frame.columns:
        raise ValueError(f"Missing ID column for split assignment: {id_col}")
    return frame.with_columns(
        pl.col(id_col)
        .map_elements(
            lambda value: assign_split(value, train=train, validation=validation, seed=seed),
            return_dtype=pl.Utf8,
        )
        .alias(split_col)
    )
