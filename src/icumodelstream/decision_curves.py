"""Decision-curve analysis (net benefit) for TRIPOD+AI clinical utility.

Pure-numpy / polars module. No CLIF coupling -- inputs are parallel arrays of
binary labels and predicted probabilities; outputs are a tidy long DataFrame
of per-threshold net benefit alongside the two reference strategies (treat-all
and treat-none).

Mirrors :mod:`icumodelstream.subgroups` in shape so the CLI wiring stays
byte-for-byte parallel to the subgroup-cols pattern.

Net benefit at probability threshold ``pt``::

    NB(pt) = TP/n - (FP/n) * (pt / (1 - pt))

Reference strategies under the same denominator::

    NB_treat_all(pt)  = prevalence - (1 - prevalence) * (pt / (1 - pt))
    NB_treat_none(pt) = 0  (by construction)

CLAUDE.md rule 7 (fail loudly on data assumptions): boundary thresholds
``pt <= 0`` and ``pt >= 1`` are rejected with a named ``ValueError``, and a
length mismatch between ``y_true`` and ``y_pred_proba`` raises -- same shape
as :func:`icumodelstream.subgroups.compute_subgroup_metrics`.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import polars as pl

# Clinically actionable thresholds spanning low-acuity triage (0.05) through
# high-conviction decisions (0.50). The CLI default is empty, so this constant
# is exposed for programmatic callers (e.g. notebooks) that want a sensible
# starting sweep without picking thresholds themselves.
DEFAULT_THRESHOLDS: tuple[float, ...] = (0.05, 0.10, 0.15, 0.20, 0.30, 0.50)


def compute_decision_curve(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
) -> pl.DataFrame:
    """Per-threshold net benefit plus the two reference-strategy curves.

    Parameters
    ----------
    y_true:
        Shape ``(n,)``, int 0/1.
    y_pred_proba:
        Shape ``(n,)``, float in ``[0, 1]``.
    thresholds:
        Probability cut-points in the open interval ``(0, 1)``. Each value
        ``pt`` produces one output row.

    Returns
    -------
    Tidy long DataFrame with columns
    ``threshold, n, prevalence, n_positive_pred, n_true_positive,
    n_false_positive, net_benefit, net_benefit_treat_all,
    net_benefit_treat_none``. Empty ``thresholds`` returns an empty frame with
    the expected schema (mirrors :func:`subgroups._empty_subgroup_frame`).

    Raises
    ------
    ValueError
        If ``y_pred_proba`` length mismatches ``y_true``, or if any threshold
        is outside ``(0, 1)``.
    """
    n = len(y_true)
    if len(y_pred_proba) != n:
        raise ValueError(
            f"y_pred_proba length {len(y_pred_proba)} != y_true length {n}."
        )

    if len(thresholds) == 0:
        return _empty_decision_curve_frame()

    for pt in thresholds:
        if not (0.0 < float(pt) < 1.0):
            raise ValueError(
                f"threshold {pt!r} must be in the open interval (0, 1)."
            )

    y_true_arr = np.asarray(y_true)
    y_proba_arr = np.asarray(y_pred_proba)
    prevalence = float(y_true_arr.mean()) if n > 0 else 0.0

    rows: list[dict[str, object]] = []
    for pt in thresholds:
        pt_f = float(pt)
        positive_pred = y_proba_arr >= pt_f
        n_pos_pred = int(positive_pred.sum())
        # Counted at the cut-point pt: TP and FP among the row-positive predictions.
        # When n_pos_pred == 0 both TP and FP are zero, giving net_benefit == 0
        # rather than NaN -- clinical readers expect a 0-line on the plot, not a gap.
        tp = int(((y_true_arr == 1) & positive_pred).sum())
        fp = int(((y_true_arr == 0) & positive_pred).sum())
        weight = pt_f / (1.0 - pt_f)
        net_benefit = (tp / n) - (fp / n) * weight if n > 0 else 0.0
        nb_treat_all = prevalence - (1.0 - prevalence) * weight
        rows.append(
            {
                "threshold": pt_f,
                "n": n,
                "prevalence": prevalence,
                "n_positive_pred": n_pos_pred,
                "n_true_positive": tp,
                "n_false_positive": fp,
                "net_benefit": float(net_benefit),
                "net_benefit_treat_all": float(nb_treat_all),
                "net_benefit_treat_none": 0.0,
            }
        )

    return pl.DataFrame(rows).sort("threshold")


def _empty_decision_curve_frame() -> pl.DataFrame:
    """Empty frame with the expected schema. Lets callers downstream of empty
    ``thresholds`` treat the result uniformly with a populated curve."""
    return pl.DataFrame(
        schema={
            "threshold": pl.Float64,
            "n": pl.Int64,
            "prevalence": pl.Float64,
            "n_positive_pred": pl.Int64,
            "n_true_positive": pl.Int64,
            "n_false_positive": pl.Int64,
            "net_benefit": pl.Float64,
            "net_benefit_treat_all": pl.Float64,
            "net_benefit_treat_none": pl.Float64,
        }
    )
