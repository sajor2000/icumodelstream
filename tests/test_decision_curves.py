"""Tests for src/icumodelstream/decision_curves.py (U1 of DCA plan)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from icumodelstream.decision_curves import compute_decision_curve


def test_compute_decision_curve_happy_path_separable() -> None:
    """Perfectly separable signal yields positive net benefit at every threshold,
    and treat-all matches the analytic formula."""
    n = 200
    rng = np.random.default_rng(0)
    signal = rng.standard_normal(n)
    y_true = (signal > 0).astype(np.int64)
    # Strong sigmoid (×6) makes negatives ≈ 0 and positives ≈ 1, so the
    # separable case actually exercises high thresholds like 0.9 instead of
    # collapsing to zero positive predictions there.
    y_pred = 1.0 / (1.0 + np.exp(-6.0 * signal))

    out = compute_decision_curve(y_true, y_pred, thresholds=[0.1, 0.5, 0.9])
    assert out.height == 3

    prevalence = float(y_true.mean())
    for row in out.iter_rows(named=True):
        pt = row["threshold"]
        # Reference: treat-none is exactly zero by construction.
        assert row["net_benefit_treat_none"] == 0.0
        # Reference: treat-all matches prevalence - (1-p) * pt/(1-pt).
        expected_treat_all = prevalence - (1.0 - prevalence) * (pt / (1.0 - pt))
        assert math.isclose(row["net_benefit_treat_all"], expected_treat_all, rel_tol=1e-9)
        # Separable model: net benefit at the model curve beats zero (treat-none).
        assert row["net_benefit"] > 0.0, f"separable model NB ≤ 0 at pt={pt}"
        # And n is preserved.
        assert row["n"] == n


def test_compute_decision_curve_worthless_model_matches_treat_all_at_threshold() -> None:
    """A constant 0.5 predictor at pt=0.5 should classify every row as positive,
    yielding net_benefit == prevalence - (1-prevalence), which equals
    net_benefit_treat_all at that threshold."""
    n = 100
    y_true = np.array([1] * 40 + [0] * 60, dtype=np.int64)  # prevalence 0.40
    y_pred = np.full(n, 0.5)

    out = compute_decision_curve(y_true, y_pred, thresholds=[0.5])
    row = out.row(0, named=True)

    prevalence = 0.40
    # At pt=0.5, weight = 1.0, so NB = prevalence - (1-prevalence).
    expected_nb = prevalence - (1.0 - prevalence)
    assert math.isclose(row["net_benefit"], expected_nb, rel_tol=1e-9)
    # And the treat-all curve agrees at this threshold.
    assert math.isclose(row["net_benefit_treat_all"], expected_nb, rel_tol=1e-9)
    # Every row was predicted positive.
    assert row["n_positive_pred"] == n
    assert row["n_true_positive"] == 40
    assert row["n_false_positive"] == 60


@pytest.mark.parametrize("bad_pt", [0.0, 1.0, -0.1, 1.5])
def test_compute_decision_curve_rejects_boundary_thresholds(bad_pt: float) -> None:
    y_true = np.array([0, 1, 0, 1], dtype=np.int64)
    y_pred = np.array([0.1, 0.9, 0.2, 0.8])
    with pytest.raises(ValueError, match=str(bad_pt)):
        compute_decision_curve(y_true, y_pred, thresholds=[bad_pt])


def test_compute_decision_curve_length_mismatch_raises() -> None:
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0.1, 0.9, 0.2])  # length 3 vs 4
    with pytest.raises(ValueError, match="length"):
        compute_decision_curve(y_true, y_pred, thresholds=[0.5])


def test_compute_decision_curve_empty_thresholds_returns_empty_schema() -> None:
    y_true = np.array([0, 1, 0, 1], dtype=np.int64)
    y_pred = np.array([0.1, 0.9, 0.2, 0.8])
    out = compute_decision_curve(y_true, y_pred, thresholds=[])
    assert out.height == 0
    expected_cols = {
        "threshold",
        "n",
        "prevalence",
        "n_positive_pred",
        "n_true_positive",
        "n_false_positive",
        "net_benefit",
        "net_benefit_treat_all",
        "net_benefit_treat_none",
    }
    assert set(out.columns) == expected_cols


def test_compute_decision_curve_high_threshold_zero_cohort_returns_zero_not_nan() -> None:
    """Threshold above every score → no positive predictions → net_benefit = 0,
    not NaN (clinical readers expect a 0-line on the plot, not a gap)."""
    n = 50
    y_true = np.array([0] * 25 + [1] * 25, dtype=np.int64)
    y_pred = np.linspace(0.05, 0.90, n)  # max 0.90 < 0.99

    out = compute_decision_curve(y_true, y_pred, thresholds=[0.99])
    row = out.row(0, named=True)

    assert row["n_positive_pred"] == 0
    assert row["n_true_positive"] == 0
    assert row["n_false_positive"] == 0
    assert row["net_benefit"] == 0.0
    assert not math.isnan(row["net_benefit"])
