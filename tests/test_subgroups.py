"""Tests for src/icumodelstream/subgroups.py (U1 of subgroup-performance plan)."""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest

from icumodelstream.subgroups import (
    DEFAULT_AGE_LABELS,
    UNKNOWN_LABEL,
    assign_age_band,
    compute_subgroup_metrics,
)


def test_assign_age_band_happy_path() -> None:
    ages = np.array([25, 50, 70, 85])
    bands = assign_age_band(ages)
    assert list(bands) == ["<40", "40-65", "65-80", "80+"]


def test_assign_age_band_handles_null_and_out_of_range() -> None:
    # ages: None, "not a number", -5, 200, 39 (just under 40)
    ages = np.array([None, "foo", -5, 200, 39], dtype=object)
    bands = assign_age_band(ages)
    # None and "foo" -> Unknown; -5 below bins -> Unknown; 200 falls in (80, 200)
    # interval which is right-open, so 200 itself is NOT in 80+. 39 is in <40.
    assert bands[0] == UNKNOWN_LABEL
    assert bands[1] == UNKNOWN_LABEL
    assert bands[2] == UNKNOWN_LABEL
    assert bands[3] == UNKNOWN_LABEL  # 200 is right-open boundary, excluded
    assert bands[4] == "<40"


def test_assign_age_band_right_open_intervals() -> None:
    """Verify boundary semantics: 40 -> "40-65" (not "<40")."""
    ages = np.array([40, 65, 80])
    bands = assign_age_band(ages)
    assert list(bands) == ["40-65", "65-80", "80+"]


def test_compute_subgroup_metrics_happy_path() -> None:
    """Two variables (sex: 2 vals; age_band: 3 vals) over 100 rows -> 5 rows."""
    rng = np.random.default_rng(0)
    n = 100
    # Make y predictable from a signal so AUROC > 0.5.
    signal = rng.standard_normal(n)
    y_true = (signal > 0).astype(np.int64)
    y_pred_proba = 1.0 / (1.0 + np.exp(-signal))  # perfectly correlated -> AUROC ~ 1

    sex_labels = np.array(["M" if i % 2 == 0 else "F" for i in range(n)], dtype=object)
    age_band_labels = np.array(
        ["<40" if i < 33 else "40-65" if i < 66 else "65-80" for i in range(n)],
        dtype=object,
    )

    out = compute_subgroup_metrics(
        y_true, y_pred_proba, {"sex": sex_labels, "age_band": age_band_labels}
    )

    # 2 sex values + 3 age_band values = 5 rows
    assert out.height == 5
    # Per-variable sum of n equals the global cohort size.
    sex_total = out.filter(pl.col("subgroup_var") == "sex")["n"].sum()
    age_total = out.filter(pl.col("subgroup_var") == "age_band")["n"].sum()
    assert sex_total == n
    assert age_total == n
    # On a perfectly separable signal, all subgroups have AUROC very close to 1.
    for row in out.iter_rows(named=True):
        assert row["auroc"] > 0.95, f"unexpectedly low AUROC for {row}"


def test_compute_subgroup_metrics_single_class_subgroup() -> None:
    """A subgroup where every y_true is 0 yields NaN AUROC and a warning row."""
    n = 50
    y_true = np.array([0] * 25 + [1] * 25, dtype=np.int64)
    y_pred_proba = np.linspace(0.05, 0.95, n)
    # Age band labels: first 25 (all y=0) tagged "under_40"; rest "over_40".
    age_band = np.array(["under_40"] * 25 + ["over_40"] * 25, dtype=object)

    out = compute_subgroup_metrics(y_true, y_pred_proba, {"age_band": age_band})

    under_40 = out.filter(pl.col("subgroup_value") == "under_40").row(0, named=True)
    over_40 = out.filter(pl.col("subgroup_value") == "over_40").row(0, named=True)

    # under_40 is single-class -> AUROC NaN + warning
    assert math.isnan(under_40["auroc"])
    assert math.isnan(under_40["auprc"])
    assert under_40["warning"] == "single_class_y_true"
    assert under_40["n"] == 25
    assert under_40["prevalence"] == 0.0
    # over_40 has both classes (all y=1 here is also single-class — let's check)
    # Actually y_true[25:] is all 1, so over_40 is also single-class. Adjust:
    assert math.isnan(over_40["auroc"])
    assert over_40["warning"] == "single_class_y_true"


def test_compute_subgroup_metrics_null_labels_become_unknown() -> None:
    """Explicit null sex labels are bucketed into an "Unknown" group, not dropped."""
    n = 100
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=n).astype(np.int64)
    y_pred_proba = rng.uniform(0, 1, size=n)
    # First 10 rows have null sex; remaining 90 split M/F.
    sex_labels = np.array(
        [None if i < 10 else ("M" if i % 2 == 0 else "F") for i in range(n)],
        dtype=object,
    )

    out = compute_subgroup_metrics(y_true, y_pred_proba, {"sex": sex_labels})

    values = set(out["subgroup_value"].to_list())
    assert UNKNOWN_LABEL in values
    unknown_row = out.filter(pl.col("subgroup_value") == UNKNOWN_LABEL).row(0, named=True)
    assert unknown_row["n"] == 10
    # Per-variable n sums to the full cohort (no silent drop).
    assert out.filter(pl.col("subgroup_var") == "sex")["n"].sum() == n


def test_compute_subgroup_metrics_empty_groups_dict() -> None:
    """No subgroup columns -> empty DataFrame with the expected schema."""
    y_true = np.array([0, 1, 0, 1], dtype=np.int64)
    y_pred_proba = np.array([0.1, 0.9, 0.2, 0.8])
    out = compute_subgroup_metrics(y_true, y_pred_proba, {})
    assert out.height == 0
    expected_cols = {
        "subgroup_var", "subgroup_value", "n", "prevalence", "auroc",
        "auprc", "brier_score", "calibration_intercept",
        "calibration_slope", "warning",
    }
    assert set(out.columns) == expected_cols


def test_compute_subgroup_metrics_length_mismatch_raises() -> None:
    y_true = np.array([0, 1, 0, 1])
    y_pred_proba = np.array([0.1, 0.9, 0.2, 0.8])
    bad_groups = {"sex": np.array(["M", "F", "M"])}  # length 3 vs 4
    with pytest.raises(ValueError, match="sex"):
        compute_subgroup_metrics(y_true, y_pred_proba, bad_groups)
