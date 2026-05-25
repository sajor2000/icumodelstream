from __future__ import annotations

import numpy as np
import polars as pl


def decision_curve(
    frame: pl.DataFrame,
    label_col: str,
    score_col: str,
    thresholds: list[float] | None = None,
) -> pl.DataFrame:
    """Compute decision-curve net benefit for model, treat-all, and treat-none strategies."""
    if thresholds is None:
        thresholds = [round(x, 3) for x in np.arange(0.05, 0.51, 0.05)]
    if any(t <= 0 or t >= 1 for t in thresholds):
        raise ValueError("Decision-curve thresholds must be strictly between 0 and 1.")
    missing = {label_col, score_col} - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns for decision curve: {sorted(missing)}")

    clean = frame.select(
        pl.col(label_col).cast(pl.Int64).alias("label"),
        pl.col(score_col).cast(pl.Float64).alias("score"),
    ).drop_nulls()
    if clean.height == 0:
        raise ValueError("No valid rows available for decision curve.")

    y = clean["label"].to_numpy()
    p = clean["score"].to_numpy()
    if not set(np.unique(y).tolist()).issubset({0, 1}):
        raise ValueError("Decision-curve labels must be binary 0/1.")
    n = len(y)
    prevalence = float(np.mean(y))
    rows = []
    for threshold in thresholds:
        predicted_positive = p >= threshold
        tp = int(np.sum((predicted_positive == 1) & (y == 1)))
        fp = int(np.sum((predicted_positive == 1) & (y == 0)))
        harm_weight = threshold / (1 - threshold)
        model_nb = (tp / n) - (fp / n) * harm_weight
        treat_all_nb = prevalence - (1 - prevalence) * harm_weight
        rows.append(
            {
                "threshold": float(threshold),
                "n": n,
                "prevalence": prevalence,
                "true_positives": tp,
                "false_positives": fp,
                "net_benefit_model": float(model_nb),
                "net_benefit_treat_all": float(treat_all_nb),
                "net_benefit_treat_none": 0.0,
            }
        )
    return pl.DataFrame(rows)
