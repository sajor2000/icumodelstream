"""LightGBM and logistic-regression baselines with discrimination + calibration.

Phase 4 commits to reporting calibration alongside discrimination for every
baseline: a model that ranks well (high AUROC) but is miscalibrated cannot be
used to communicate ICU mortality risk to clinicians. This module returns both
in :class:`BaselineResult`.

CLAUDE.md rule 9 (baselines before deep learning): two boring, well-understood
models live here -- LightGBM (handles nulls natively, captures non-linearities)
and a scikit-learn logistic pipeline (linear, requires imputation, gives an
interpretable coefficient baseline). Foundation-model work is out of scope.

CLAUDE.md rule 10 (one change at a time, reproducible via seed): every public
function takes ``seed`` and forwards it to the underlying ``random_state`` so
the same data + same seed reproduce identical predictions and metrics.

CLAUDE.md rule 7 (fail loudly on data assumptions): we reject empty inputs,
mismatched lengths, non-binary labels, and all-zero targets up front with
named ValueErrors rather than letting LightGBM raise an opaque core error or
silently train a degenerate model.

Persistence is intentionally minimal: ``save_model`` / ``load_model`` cover the
LightGBM booster only via the native text format. The logistic pipeline is
cheap to refit and pickling sklearn objects across versions is brittle, so we
do not provide a save path for it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

_PROBA_CLIP_EPS = 1e-7
_N_CALIBRATION_BINS = 10


@dataclass(frozen=True)
class BaselineResult:
    """Container for a fitted baseline's test-set predictions and metrics.

    Attributes
    ----------
    model_name:
        Either ``"lightgbm"`` or ``"logistic"``. Lets downstream code branch
        without inspecting the model object.
    y_true:
        Ground-truth labels on the test set, shape ``(n_test,)``, int 0/1.
    y_pred_proba:
        Predicted probabilities of the positive class, shape ``(n_test,)``,
        floats in ``[0, 1]``.
    metrics:
        Dict with six keys: ``auroc``, ``auprc``, ``brier_score``,
        ``prevalence``, ``calibration_intercept``, ``calibration_slope``.
    calibration_table:
        10-row polars DataFrame (or fewer rows if quantile ties collapse bins)
        with columns ``bin``, ``mean_pred``, ``mean_actual``, ``count``.
    """

    model_name: str
    y_true: np.ndarray
    y_pred_proba: np.ndarray
    metrics: dict[str, float]
    calibration_table: pl.DataFrame


def _validate_xy(
    X_train: pl.DataFrame,
    y_train: pl.Series,
    X_test: pl.DataFrame,
    y_test: pl.Series,
) -> None:
    """Reject empty/mismatched/non-binary inputs up front (CLAUDE.md rule 7)."""
    if len(X_train) == 0:
        raise ValueError("Cannot fit baseline: len(X_train) == 0")
    if len(X_test) == 0:
        raise ValueError("Cannot evaluate baseline: len(X_test) == 0")
    if len(X_train) != len(y_train):
        raise ValueError(
            f"Length mismatch: len(X_train)={len(X_train)} but "
            f"len(y_train)={len(y_train)}. They must match."
        )
    if len(X_test) != len(y_test):
        raise ValueError(
            f"Length mismatch: len(X_test)={len(X_test)} but "
            f"len(y_test)={len(y_test)}. They must match."
        )

    y_train_np = y_train.to_numpy()
    unique_train = set(np.unique(y_train_np).tolist())
    if not unique_train.issubset({0, 1}):
        raise ValueError(
            f"y_train must be binary 0/1. Observed unique values: "
            f"{sorted(unique_train)}."
        )
    if y_train_np.sum() == 0 or y_train_np.sum() == len(y_train_np):
        # Both LightGBM and logistic regression need both classes to fit a
        # meaningful classifier. Raise rather than letting the underlying
        # library produce a constant predictor with confusing metrics.
        raise ValueError(
            "y_train contains only one class (all 0s or all 1s); baseline "
            "models require at least one example of each class."
        )


def _calibration_intercept_slope(
    y_true: np.ndarray, y_pred_proba: np.ndarray
) -> tuple[float, float]:
    """Fit logit(p) -> y; intercept and slope are the calibration parameters.

    A perfectly calibrated model yields intercept=0, slope=1. Intercept != 0
    indicates over/under-prediction; slope != 1 indicates over/under-confidence.
    We use a very weak regularizer (C=1e10) so the fit is essentially
    unregularized -- the standard reporting convention.
    """
    clipped = np.clip(y_pred_proba, _PROBA_CLIP_EPS, 1.0 - _PROBA_CLIP_EPS)
    logits = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
    # If y_true has only one class on the test set we cannot fit a logistic
    # model; return NaNs rather than crashing the whole evaluation.
    if len(np.unique(y_true)) < 2:
        return float("nan"), float("nan")
    lr = LogisticRegression(C=1e10, solver="lbfgs", max_iter=1000)
    lr.fit(logits, y_true)
    return float(lr.intercept_[0]), float(lr.coef_[0, 0])


def _calibration_table(
    y_true: np.ndarray, y_pred_proba: np.ndarray
) -> pl.DataFrame:
    """Quantile-bin predictions into up to 10 bins; report mean pred vs actual."""
    # qcut with duplicates='drop' collapses tied quantile edges so heavily
    # tied predictions (common at low prevalence) do not crash this helper.
    bins = pd.qcut(
        y_pred_proba, q=_N_CALIBRATION_BINS, labels=False, duplicates="drop"
    )
    df = pd.DataFrame(
        {"bin": bins, "y_true": y_true, "y_pred": y_pred_proba}
    )
    grouped = (
        df.groupby("bin", sort=True)
        .agg(mean_pred=("y_pred", "mean"), mean_actual=("y_true", "mean"), count=("y_true", "size"))
        .reset_index()
    )
    return pl.from_pandas(grouped).with_columns(
        pl.col("bin").cast(pl.Int64),
        pl.col("mean_pred").cast(pl.Float64),
        pl.col("mean_actual").cast(pl.Float64),
        pl.col("count").cast(pl.Int64),
    )


def _compute_metrics(
    y_true: np.ndarray, y_pred_proba: np.ndarray
) -> dict[str, float]:
    """Discrimination (AUROC, AUPRC), Brier, prevalence, and calibration."""
    intercept, slope = _calibration_intercept_slope(y_true, y_pred_proba)
    return {
        "auroc": float(roc_auc_score(y_true, y_pred_proba)),
        "auprc": float(average_precision_score(y_true, y_pred_proba)),
        "brier_score": float(brier_score_loss(y_true, y_pred_proba)),
        "prevalence": float(y_true.mean()),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
    }


def fit_lightgbm_baseline(
    X_train: pl.DataFrame,
    y_train: pl.Series,
    X_test: pl.DataFrame,
    y_test: pl.Series,
    seed: int = 42,
    is_unbalance: bool = False,
) -> tuple[Any, BaselineResult]:
    """Fit a LightGBM classifier and return (model, evaluation result).

    LightGBM handles nulls natively, so feature DataFrames may contain missing
    values without preprocessing. Polars frames are converted to pandas before
    fitting because the LightGBM sklearn wrapper has best-tested support there
    (preserves feature names, accepts NaN, no copy on float64 columns).

    Hyperparameters are intentionally modest defaults from the plan; tuning
    happens in a later phase, not in this baseline module.

    is_unbalance defaults to False. Setting True up-weights the minority class
    during training, which inflates predicted probabilities (a -1.94 logit shift
    was observed on real CLIF-MIMIC adult ICU data, May 2026, lifting Brier from
    ~0.10 to 0.19 with only a small AUROC gain). Use is_unbalance=True only when
    AUROC ranking is the only metric that matters and downstream consumers do
    not interpret the predicted probabilities as actual risk. For calibrated
    probabilities, leave at False and apply Platt scaling or isotonic regression
    post-hoc if further calibration is needed.
    """
    _validate_xy(X_train, y_train, X_test, y_test)

    X_train_pd = X_train.to_pandas()
    X_test_pd = X_test.to_pandas()
    y_train_np = y_train.to_numpy().astype(int)
    y_test_np = y_test.to_numpy().astype(int)

    model = lgb.LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=31,
        random_state=seed,
        is_unbalance=is_unbalance,
        verbose=-1,
    )
    model.fit(X_train_pd, y_train_np)
    y_pred_proba = model.predict_proba(X_test_pd)[:, 1]

    result = BaselineResult(
        model_name="lightgbm",
        y_true=y_test_np,
        y_pred_proba=y_pred_proba,
        metrics=_compute_metrics(y_test_np, y_pred_proba),
        calibration_table=_calibration_table(y_test_np, y_pred_proba),
    )
    return model, result


def fit_logistic_baseline(
    X_train: pl.DataFrame,
    y_train: pl.Series,
    X_test: pl.DataFrame,
    y_test: pl.Series,
    seed: int = 42,
) -> tuple[Any, BaselineResult]:
    """Fit a median-impute -> standard-scale -> logistic pipeline.

    Logistic regression does not tolerate NaN, so we median-impute first.
    Standardization keeps the L2 default on similar scales across features.
    """
    _validate_xy(X_train, y_train, X_test, y_test)

    X_train_np = X_train.to_numpy().astype(float)
    X_test_np = X_test.to_numpy().astype(float)
    y_train_np = y_train.to_numpy().astype(int)
    y_test_np = y_test.to_numpy().astype(int)

    pipeline = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("lr", LogisticRegression(random_state=seed, max_iter=1000)),
        ]
    )
    pipeline.fit(X_train_np, y_train_np)
    y_pred_proba = pipeline.predict_proba(X_test_np)[:, 1]

    result = BaselineResult(
        model_name="logistic",
        y_true=y_test_np,
        y_pred_proba=y_pred_proba,
        metrics=_compute_metrics(y_test_np, y_pred_proba),
        calibration_table=_calibration_table(y_test_np, y_pred_proba),
    )
    return pipeline, result


def save_model(model: Any, path: Path) -> None:
    """Persist a fitted LightGBM model to ``path`` in native text format.

    Only the LightGBM ``LGBMClassifier`` (or its underlying ``Booster``) is
    supported. We avoid pickling because LightGBM's native text format is
    forward-compatible across versions and language bindings, whereas pickled
    sklearn pipelines silently break across minor releases.
    """
    if hasattr(model, "booster_"):
        booster = model.booster_
    elif isinstance(model, lgb.Booster):
        booster = model
    else:
        raise TypeError(
            f"save_model only supports LightGBM models; got {type(model).__name__}."
        )
    booster.save_model(str(path))


def load_model(path: Path) -> Any:
    """Load a LightGBM model previously written by :func:`save_model`.

    Returns the native :class:`lightgbm.Booster`. Use ``booster.predict(X)``
    which returns positive-class probabilities directly (no ``[:, 1]`` slice).
    """
    return lgb.Booster(model_file=str(path))
