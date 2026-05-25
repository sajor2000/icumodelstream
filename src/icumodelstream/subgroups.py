"""TRIPOD+AI subgroup performance metrics.

Pure-numpy / polars module. No CLIF coupling — the input is parallel arrays
of predictions, labels, and subgroup labels; the output is a tidy DataFrame.
The CLIF-side label extraction lives in :mod:`icumodelstream.pipeline`.

CLAUDE.md rule 2 (simplicity first): per-subgroup metrics reuse the existing
:func:`icumodelstream.models.compute_metrics` whole-cohort function so every
row in the output is apples-to-apples with the whole-cohort row in the
existing baseline JSON.

CLAUDE.md rule 7 (fail loudly on data assumptions): length-mismatched inputs
raise; single-class subgroups are NOT silently dropped — they get a `warning`
field instead so the JSON consumer can see the gap.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from icumodelstream.models import compute_metrics

# Conventional ICU age bands. Right-open intervals matching numpy semantics:
# 40 lands in "40-65" (not "<40"). Open-ended top bin includes top-coded
# MIMIC ages (89+).
DEFAULT_AGE_BINS: tuple[tuple[float, float], ...] = (
    (0, 40),
    (40, 65),
    (65, 80),
    (80, 200),
)
DEFAULT_AGE_LABELS: tuple[str, ...] = ("<40", "40-65", "65-80", "80+")
UNKNOWN_LABEL = "Unknown"


def assign_age_band(
    ages: np.ndarray,
    bins: tuple[tuple[float, float], ...] = DEFAULT_AGE_BINS,
    labels: tuple[str, ...] = DEFAULT_AGE_LABELS,
) -> np.ndarray:
    """Map an array of ages to band-label strings.

    Null or out-of-range ages map to ``"Unknown"``. Bins are right-open
    intervals: ``[lo, hi)``.
    """
    if len(bins) != len(labels):
        raise ValueError(
            f"bins and labels must have same length; got {len(bins)} vs {len(labels)}."
        )
    out = np.full(len(ages), UNKNOWN_LABEL, dtype=object)
    for (lo, hi), label in zip(bins, labels):
        mask = np.zeros(len(ages), dtype=bool)
        for i, age in enumerate(ages):
            if age is None:
                continue
            try:
                age_f = float(age)
            except (TypeError, ValueError):
                continue
            if lo <= age_f < hi:
                mask[i] = True
        out[mask] = label
    return out


def compute_subgroup_metrics(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    groups: dict[str, np.ndarray],
) -> pl.DataFrame:
    """Per-subgroup AUROC / AUPRC / Brier / calibration for TRIPOD+AI reporting.

    Parameters
    ----------
    y_true:
        Shape ``(n,)``, int 0/1.
    y_pred_proba:
        Shape ``(n,)``, float in ``[0, 1]``.
    groups:
        Mapping ``{variable_name: subgroup_labels}``. Each label array has
        length ``n``. Null or empty-string labels are bucketed into
        ``"Unknown"`` rather than silently dropped.

    Returns
    -------
    Tidy long-format DataFrame with columns:
    ``subgroup_var, subgroup_value, n, prevalence, auroc, auprc,
    brier_score, calibration_intercept, calibration_slope, warning``.
    Empty DataFrame (with the expected schema) when ``groups`` is empty.

    Raises
    ------
    ValueError
        If any group array length mismatches ``y_true``.
    """
    n = len(y_true)
    if len(y_pred_proba) != n:
        raise ValueError(
            f"y_pred_proba length {len(y_pred_proba)} != y_true length {n}."
        )
    for var, labels_arr in groups.items():
        if len(labels_arr) != n:
            raise ValueError(
                f"group {var!r} length {len(labels_arr)} != y_true length {n}."
            )

    if not groups:
        return _empty_subgroup_frame()

    rows: list[dict[str, object]] = []
    for var, labels_arr in groups.items():
        # Normalize: nulls and empty-strings become "Unknown".
        normalized = np.array(
            [UNKNOWN_LABEL if (v is None or (isinstance(v, str) and not v)) else v
             for v in labels_arr],
            dtype=object,
        )
        unique_values = sorted(set(normalized.tolist()))
        for value in unique_values:
            mask = normalized == value
            if not mask.any():
                continue
            sub_y = y_true[mask]
            sub_p = y_pred_proba[mask]
            metrics = compute_metrics(sub_y, sub_p)
            single_class = len(np.unique(sub_y)) < 2
            rows.append(
                {
                    "subgroup_var": var,
                    "subgroup_value": str(value),
                    "n": int(mask.sum()),
                    "prevalence": metrics["prevalence"],
                    "auroc": metrics["auroc"],
                    "auprc": metrics["auprc"],
                    "brier_score": metrics["brier_score"],
                    "calibration_intercept": metrics["calibration_intercept"],
                    "calibration_slope": metrics["calibration_slope"],
                    "warning": "single_class_y_true" if single_class else None,
                }
            )

    df = pl.DataFrame(rows)
    # Sort by variable, then by n descending so the biggest subgroup shows first.
    return df.sort(["subgroup_var", "n"], descending=[False, True])


def _empty_subgroup_frame() -> pl.DataFrame:
    """Empty frame with the expected schema. Lets callers downstream of an
    empty `groups` dict treat the result uniformly."""
    return pl.DataFrame(
        schema={
            "subgroup_var": pl.Utf8,
            "subgroup_value": pl.Utf8,
            "n": pl.Int64,
            "prevalence": pl.Float64,
            "auroc": pl.Float64,
            "auprc": pl.Float64,
            "brier_score": pl.Float64,
            "calibration_intercept": pl.Float64,
            "calibration_slope": pl.Float64,
            "warning": pl.Utf8,
        }
    )
