"""Tests for group-aware splits (CLAUDE.md section 8: no silent patient leakage)."""

from __future__ import annotations

import polars as pl
import pytest

from icumodelstream.splits import group_kfold, group_train_test_split


def _make_xy(n_rows: int) -> tuple[pl.DataFrame, pl.Series]:
    """Build a minimal X/y pair of length n_rows for split tests."""
    X = pl.DataFrame({"feat1": [1.0] * n_rows, "feat2": [2.0] * n_rows})
    y = pl.Series("label", [0] * n_rows)
    return X, y


def test_group_train_test_split_no_patient_leakage() -> None:
    """Load-bearing contract: zero patients shared between train and test."""
    # 50 unique patients, 2 rows each = 100 rows
    patient_ids = [f"P{i:03d}" for i in range(50) for _ in range(2)]
    # Embed patient_id as a column so we can recover the split's groups directly
    # from the returned DataFrames -- no need to re-run sklearn.
    X = pl.DataFrame({"feat1": [1.0] * 100, "feat2": [2.0] * 100, "patient_id": patient_ids})
    y = pl.Series("label", [0] * 100)
    groups = pl.Series("patient_id", patient_ids)

    X_train, X_test, y_train, y_test = group_train_test_split(X, y, groups, test_size=0.2, seed=42)

    train_groups = set(X_train["patient_id"].to_list())
    test_groups = set(X_test["patient_id"].to_list())

    assert train_groups & test_groups == set(), "patient leaked across train/test"
    # GroupShuffleSplit rounds whole groups, so the test set has ~10 patients (~20 rows).
    assert 8 <= len(test_groups) <= 12
    assert 16 <= len(X_test) <= 24
    assert len(X_train) + len(X_test) == 100
    assert len(y_train) + len(y_test) == 100


def test_group_train_test_split_single_patient_no_overlap() -> None:
    """10 rows all from one patient: rows go entirely to train OR entirely to test."""
    X, y = _make_xy(10)
    groups = pl.Series("patient_id", ["P1"] * 10)

    X_train, X_test, y_train, y_test = group_train_test_split(X, y, groups, test_size=0.2, seed=0)

    # With one group, the wrapper puts every row on exactly one side -- never both,
    # so the no-leakage contract is preserved.
    assert (len(X_train) == 10 and len(X_test) == 0) or (len(X_train) == 0 and len(X_test) == 10)
    assert len(y_train) == len(X_train)
    assert len(y_test) == len(X_test)
    # Intersection is trivially empty because one side is empty.
    assert min(len(X_train), len(X_test)) == 0


def test_group_train_test_split_reproducible_and_seed_sensitive() -> None:
    """Same seed -> identical split. Different seed -> different split (high prob)."""
    # 10 patients, 1 row each, test_size=0.5 => 5-vs-5 contrast is visible.
    patient_ids = [f"P{i}" for i in range(10)]
    # Embed patient_id as a feature column so we can recover assignments per call.
    X = pl.DataFrame({"feat1": [1.0] * 10, "feat2": [2.0] * 10, "patient_id": patient_ids})
    y = pl.Series("label", [0] * 10)
    groups = pl.Series("patient_id", patient_ids)

    X_train_a, X_test_a, y_train_a, y_test_a = group_train_test_split(
        X, y, groups, test_size=0.5, seed=42
    )
    X_train_b, X_test_b, y_train_b, y_test_b = group_train_test_split(
        X, y, groups, test_size=0.5, seed=42
    )

    # Same seed: byte-identical results
    assert X_train_a.equals(X_train_b)
    assert X_test_a.equals(X_test_b)
    assert y_train_a.equals(y_train_b)
    assert y_test_a.equals(y_test_b)

    # Different seed: different test groups (5-vs-5 makes the contrast obvious)
    _, X_test_c, _, _ = group_train_test_split(X, y, groups, test_size=0.5, seed=7)
    assert set(X_test_a["patient_id"].to_list()) != set(X_test_c["patient_id"].to_list())


def test_group_train_test_split_raises_on_length_mismatch() -> None:
    X, y = _make_xy(10)
    bad_groups = pl.Series("patient_id", ["P1"] * 9)  # off by one

    with pytest.raises(ValueError) as exc:
        group_train_test_split(X, y, bad_groups)

    msg = str(exc.value)
    assert "10" in msg and "9" in msg


def test_group_train_test_split_raises_on_empty() -> None:
    X = pl.DataFrame({"feat1": [], "feat2": []})
    y = pl.Series("label", [], dtype=pl.Int64)
    groups = pl.Series("patient_id", [], dtype=pl.Utf8)

    with pytest.raises(ValueError, match="empty"):
        group_train_test_split(X, y, groups)


def test_group_kfold_5fold_disjoint() -> None:
    """Each of 50 patients appears in exactly one test fold over 5 folds."""
    patient_ids = [f"P{i:03d}" for i in range(50)]
    X, y = _make_xy(50)
    groups = pl.Series("patient_id", patient_ids)
    groups_np = groups.to_numpy()

    folds = group_kfold(X, y, groups, n_splits=5)

    assert len(folds) == 5

    all_test_idx: list[int] = []
    for train_idx, test_idx in folds:
        # No group shared within a fold
        train_groups = set(groups_np[train_idx])
        test_groups = set(groups_np[test_idx])
        assert train_groups & test_groups == set()
        all_test_idx.extend(test_idx.tolist())

    # Union of all test indices == every row exactly once
    assert sorted(all_test_idx) == list(range(50))


def test_group_kfold_raises_when_groups_lt_splits() -> None:
    """3 unique patients cannot fill 5 folds."""
    X, y = _make_xy(3)
    groups = pl.Series("patient_id", ["P1", "P2", "P3"])

    with pytest.raises(ValueError, match="folds"):
        group_kfold(X, y, groups, n_splits=5)
