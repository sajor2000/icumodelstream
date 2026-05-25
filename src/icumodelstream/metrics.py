from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl


@dataclass(frozen=True)
class BinaryMetrics:
    """Core binary prediction metrics used for ICU mortality baselines."""

    n: int
    prevalence: float
    auroc: float | None
    brier: float
    log_loss: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "prevalence": self.prevalence,
            "auroc": self.auroc,
            "brier": self.brier,
            "log_loss": self.log_loss,
        }


def _as_clean_arrays(
    y_true: list[float] | np.ndarray, y_score: list[float] | np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_score, dtype=float)
    if y.shape != p.shape:
        raise ValueError(
            f"y_true and y_score must have the same shape; got {y.shape} and {p.shape}"
        )
    mask = np.isfinite(y) & np.isfinite(p)
    y = y[mask]
    p = np.clip(p[mask], 1e-15, 1 - 1e-15)
    if y.size == 0:
        raise ValueError("No finite labels and scores were provided.")
    unique = set(np.unique(y).tolist())
    if not unique.issubset({0.0, 1.0}):
        raise ValueError(f"Binary labels must be 0/1 after filtering; observed {sorted(unique)}")
    return y.astype(int), p


def auroc_rank(y_true: list[float] | np.ndarray, y_score: list[float] | np.ndarray) -> float | None:
    """Compute AUROC from ranks with average handling for tied scores.

    Returns None when only one class is present, which is common in small subgroups.
    """
    y, p = _as_clean_arrays(y_true, y_score)
    n_pos = int(y.sum())
    n_neg = int(y.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return None

    order = np.argsort(p, kind="mergesort")
    sorted_scores = p[order]
    ranks = np.empty_like(p, dtype=float)
    i = 0
    while i < len(sorted_scores):
        j = i + 1
        while j < len(sorted_scores) and sorted_scores[j] == sorted_scores[i]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        ranks[order[i:j]] = avg_rank
        i = j

    pos_rank_sum = float(ranks[y == 1].sum())
    return (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def binary_classification_metrics(
    y_true: list[float] | np.ndarray,
    y_score: list[float] | np.ndarray,
) -> BinaryMetrics:
    """Compute discrimination, calibration-sensitive, and overall error metrics."""
    y, p = _as_clean_arrays(y_true, y_score)
    brier = float(np.mean((p - y) ** 2))
    log_loss = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    return BinaryMetrics(
        n=int(y.size),
        prevalence=float(np.mean(y)),
        auroc=auroc_rank(y, p),
        brier=brier,
        log_loss=log_loss,
    )


def calibration_table(
    frame: pl.DataFrame,
    label_col: str,
    score_col: str,
    n_bins: int = 10,
) -> pl.DataFrame:
    """Return equal-frequency calibration bins for model audit reports."""
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2.")
    needed = {label_col, score_col}
    missing = needed - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns for calibration table: {sorted(missing)}")

    clean = (
        frame.select([pl.col(label_col).cast(pl.Float64), pl.col(score_col).cast(pl.Float64)])
        .drop_nulls()
        .filter(pl.col(score_col).is_between(0.0, 1.0, closed="both"))
        .sort(score_col)
        .with_row_index("row_id")
    )
    if clean.height == 0:
        raise ValueError("No valid rows available for calibration table.")
    bins = min(n_bins, clean.height)
    return (
        clean.with_columns(((pl.col("row_id") * bins) // pl.len()).alias("bin"))
        .group_by("bin")
        .agg(
            pl.len().alias("n"),
            pl.col(score_col).mean().alias("mean_predicted_risk"),
            pl.col(label_col).mean().alias("observed_event_rate"),
            pl.col(score_col).min().alias("min_predicted_risk"),
            pl.col(score_col).max().alias("max_predicted_risk"),
        )
        .sort("bin")
    )


def expected_calibration_error(calibration: pl.DataFrame) -> float:
    """Compute weighted absolute calibration error from a calibration table."""
    required = {"n", "mean_predicted_risk", "observed_event_rate"}
    missing = required - set(calibration.columns)
    if missing:
        raise ValueError(f"Missing calibration columns: {sorted(missing)}")
    total = calibration["n"].sum()
    if total == 0:
        return math.nan
    error = (
        (calibration["mean_predicted_risk"] - calibration["observed_event_rate"]).abs()
        * calibration["n"]
    ).sum()
    return float(error / total)
