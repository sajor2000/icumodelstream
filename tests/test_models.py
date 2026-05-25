"""Tests for LightGBM + logistic baselines and calibration (Phase 4 U5).

Pins:
* CLAUDE.md rule 10: same seed -> identical metrics (reproducibility).
* CLAUDE.md rule 7: fail loudly on empty inputs and single-class targets.
* Phase 4 plan: every baseline reports discrimination AND calibration.
* "No imputation for LightGBM" contract: nulls pass straight through.
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import polars as pl
import pytest

from icumodelstream.models import (
    BaselineResult,
    fit_lightgbm_baseline,
    fit_logistic_baseline,
    load_model,
    save_model,
)

EXPECTED_METRIC_KEYS = {
    "auroc",
    "auprc",
    "brier_score",
    "prevalence",
    "calibration_intercept",
    "calibration_slope",
}
EXPECTED_CAL_COLS = {"bin", "mean_pred", "mean_actual", "count"}


def _make_xy(
    n: int = 200, n_features: int = 4, prevalence: float = 0.3, seed: int = 0
) -> tuple[pl.DataFrame, pl.Series]:
    """Deterministic toy dataset: f0 is mildly predictive of the label."""
    rng = np.random.default_rng(seed)
    cols = {f"f{i}": rng.normal(size=n) for i in range(n_features)}
    X = pl.DataFrame(cols)
    logits = X["f0"].to_numpy() * 1.5
    p = 1.0 / (1.0 + np.exp(-logits))
    p = p * prevalence / p.mean()
    p = np.clip(p, 0.0, 1.0)
    y = pl.Series("label", (rng.uniform(size=n) < p).astype(int))
    return X, y


def _split(
    X: pl.DataFrame, y: pl.Series, train_frac: float = 0.7
) -> tuple[pl.DataFrame, pl.DataFrame, pl.Series, pl.Series]:
    """Simple deterministic positional split for fixtures (no leakage logic needed)."""
    n_train = int(len(X) * train_frac)
    return X[:n_train], X[n_train:], y[:n_train], y[n_train:]


def test_lightgbm_reproducible_with_same_seed() -> None:
    """Same data + same seed -> identical metrics (CLAUDE.md rule 10)."""
    X, y = _make_xy()
    X_train, X_test, y_train, y_test = _split(X, y)

    _, r1 = fit_lightgbm_baseline(X_train, y_train, X_test, y_test, seed=42)
    _, r2 = fit_lightgbm_baseline(X_train, y_train, X_test, y_test, seed=42)

    assert r1.metrics["auroc"] == pytest.approx(r2.metrics["auroc"], abs=1e-9)
    assert r1.metrics["brier_score"] == pytest.approx(r2.metrics["brier_score"], abs=1e-9)
    assert r1.metrics["prevalence"] == r2.metrics["prevalence"]
    np.testing.assert_allclose(r1.y_pred_proba, r2.y_pred_proba, atol=1e-12)


def test_logistic_reproducible_with_same_seed() -> None:
    """Logistic pipeline is also seed-stable end to end."""
    X, y = _make_xy()
    X_train, X_test, y_train, y_test = _split(X, y)

    _, r1 = fit_logistic_baseline(X_train, y_train, X_test, y_test, seed=42)
    _, r2 = fit_logistic_baseline(X_train, y_train, X_test, y_test, seed=42)

    assert r1.metrics["auroc"] == pytest.approx(r2.metrics["auroc"], abs=1e-9)
    assert r1.metrics["brier_score"] == pytest.approx(r2.metrics["brier_score"], abs=1e-9)
    assert r1.metrics["prevalence"] == r2.metrics["prevalence"]
    np.testing.assert_allclose(r1.y_pred_proba, r2.y_pred_proba, atol=1e-12)


def test_baseline_result_has_expected_metric_and_calibration_shape() -> None:
    """Both baselines emit 6 metrics + a 4-column calibration table."""
    X, y = _make_xy(n=400)
    X_train, X_test, y_train, y_test = _split(X, y)

    for fit in (fit_lightgbm_baseline, fit_logistic_baseline):
        _, result = fit(X_train, y_train, X_test, y_test, seed=0)

        assert isinstance(result, BaselineResult)
        assert set(result.metrics.keys()) == EXPECTED_METRIC_KEYS
        assert set(result.calibration_table.columns) == EXPECTED_CAL_COLS
        # 10 quantile bins requested; ties may collapse, but with 400 distinct
        # predictions we should never collapse below 5 bins.
        assert 5 <= result.calibration_table.height <= 10
        # y_pred_proba lives in [0, 1] and matches the test length.
        assert result.y_pred_proba.shape == (len(y_test),)
        assert float(result.y_pred_proba.min()) >= 0.0
        assert float(result.y_pred_proba.max()) <= 1.0


def test_perfect_separation_gives_auroc_one() -> None:
    """When a feature equals the label, both models should rank perfectly."""
    n = 80
    # 40 negatives, then 40 positives; the only feature mirrors the label.
    y_arr = np.concatenate([np.zeros(n // 2, dtype=int), np.ones(n // 2, dtype=int)])
    X = pl.DataFrame({"signal": y_arr.astype(float)})
    y = pl.Series("label", y_arr)

    # Interleave so both train and test contain both classes (no leakage of
    # information beyond the signal column itself, which is the point).
    order = np.argsort(np.tile([0, 1], n // 2))
    X = X[order]
    y = y[order]
    X_train, X_test, y_train, y_test = _split(X, y, train_frac=0.6)

    _, lgb_result = fit_lightgbm_baseline(X_train, y_train, X_test, y_test, seed=0)
    _, lr_result = fit_logistic_baseline(X_train, y_train, X_test, y_test, seed=0)

    assert lgb_result.metrics["auroc"] == pytest.approx(1.0, abs=1e-9)
    assert lr_result.metrics["auroc"] == pytest.approx(1.0, abs=1e-9)


def test_lightgbm_save_load_round_trip(tmp_path) -> None:
    """Persisted booster reproduces predict_proba within float tolerance."""
    X, y = _make_xy()
    X_train, X_test, y_train, y_test = _split(X, y)

    model, result = fit_lightgbm_baseline(X_train, y_train, X_test, y_test, seed=42)
    path = tmp_path / "lgb_booster.txt"
    save_model(model, path)

    loaded = load_model(path)
    # Booster.predict returns positive-class probabilities directly.
    reloaded_proba = loaded.predict(X_test.to_pandas())

    np.testing.assert_allclose(reloaded_proba, result.y_pred_proba, atol=1e-9)
    assert isinstance(loaded, lgb.Booster)


def test_empty_input_raises() -> None:
    """Fail loudly on empty X_train (CLAUDE.md rule 7)."""
    X_train = pl.DataFrame({"f0": pl.Series([], dtype=pl.Float64)})
    y_train = pl.Series("label", [], dtype=pl.Int64)
    X_test = pl.DataFrame({"f0": [0.1, 0.2]})
    y_test = pl.Series("label", [0, 1])

    with pytest.raises(ValueError, match="X_train"):
        fit_lightgbm_baseline(X_train, y_train, X_test, y_test)


def test_all_zero_outcome_raises() -> None:
    """All-zero y_train must be rejected before LightGBM degenerates silently."""
    X, _ = _make_xy(n=50)
    y_train = pl.Series("label", [0] * 35)
    y_test = pl.Series("label", [0, 1] * 7 + [1])
    X_train, X_test = X[:35], X[35:]

    with pytest.raises(ValueError, match="one class"):
        fit_lightgbm_baseline(X_train, y_train, X_test, y_test)
    with pytest.raises(ValueError, match="one class"):
        fit_logistic_baseline(X_train, y_train, X_test, y_test)


def test_single_class_y_test_yields_nan_metrics_not_crash() -> None:
    """When y_test is single-class, AUROC/AUPRC are undefined — return NaN gracefully
    rather than crashing (finding #2). The pipeline can then surface a clear error to
    the operator instead of an unhandled sklearn ValueError mid-training.
    """
    X, _ = _make_xy(n=100)
    y_train = pl.Series("label", [0, 1] * 35)
    y_test = pl.Series("label", [0] * 30)  # all-negative test fold
    X_train, X_test = X[:70], X[70:]

    _, result = fit_lightgbm_baseline(X_train, y_train, X_test, y_test, seed=0)

    assert np.isnan(result.metrics["auroc"])
    assert np.isnan(result.metrics["auprc"])
    assert np.isnan(result.metrics["calibration_intercept"])


def test_calibration_table_uses_fixed_decile_bins() -> None:
    """Bin edges must be model-independent (fixed deciles 0.0..1.0) so calibration
    tables from LightGBM and logistic are visually comparable on the same test set.
    """
    X, y = _make_xy(n=300, prevalence=0.3)
    X_train, X_test, y_train, y_test = _split(X, y)

    _, lgbm_res = fit_lightgbm_baseline(X_train, y_train, X_test, y_test, seed=0)
    _, log_res = fit_logistic_baseline(X_train, y_train, X_test, y_test, seed=0)

    # Both tables must reference the same bin index set; with fixed deciles we expect
    # bins in {0..9}, and both models share the index space.
    lgbm_bins = set(lgbm_res.calibration_table["bin"].to_list())
    log_bins = set(log_res.calibration_table["bin"].to_list())
    assert lgbm_bins.issubset(set(range(10)))
    assert log_bins.issubset(set(range(10)))


def test_lightgbm_handles_nulls_without_imputation() -> None:
    """Pin the 'no imputation for LightGBM' contract: half the values are null."""
    X, y = _make_xy(n=200)
    rng = np.random.default_rng(123)
    # Randomly null out ~50% of each feature column.
    X_with_nulls = X.with_columns(
        [
            pl.when(pl.Series(rng.uniform(size=len(X)) < 0.5))
            .then(None)
            .otherwise(pl.col(col))
            .alias(col)
            for col in X.columns
        ]
    )
    X_train, X_test, y_train, y_test = _split(X_with_nulls, y)

    # Sanity check: the fixture really does contain nulls on both sides.
    assert X_train.null_count().sum_horizontal().item() > 0
    assert X_test.null_count().sum_horizontal().item() > 0

    _, result = fit_lightgbm_baseline(X_train, y_train, X_test, y_test, seed=0)

    # All metrics finite (no NaN propagation from null features).
    assert np.isfinite(result.metrics["auroc"])
    assert np.isfinite(result.metrics["brier_score"])
    assert result.y_pred_proba.shape == (len(y_test),)
    assert float(result.y_pred_proba.min()) >= 0.0
    assert float(result.y_pred_proba.max()) <= 1.0
