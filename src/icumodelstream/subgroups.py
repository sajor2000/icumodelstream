from __future__ import annotations

from typing import Any

import polars as pl

from icumodelstream.metrics import binary_classification_metrics


def subgroup_performance(
    frame: pl.DataFrame,
    label_col: str,
    score_col: str,
    subgroup_cols: list[str],
) -> list[dict[str, Any]]:
    """Compute binary metrics within requested subgroups.

    Null subgroup labels are converted to "Unknown". Single-class subgroups are retained with
    AUROC set to null and a warning field, rather than being silently dropped.
    """
    missing = {label_col, score_col, *subgroup_cols} - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns for subgroup performance: {sorted(missing)}")
    if not subgroup_cols:
        return []

    results: list[dict[str, Any]] = []
    for subgroup_col in subgroup_cols:
        normalized = frame.with_columns(
            pl.when(pl.col(subgroup_col).is_null())
            .then(pl.lit("Unknown"))
            .otherwise(pl.col(subgroup_col).cast(pl.Utf8))
            .alias("__subgroup_value")
        )
        for row in normalized.partition_by("__subgroup_value", as_dict=True).items():
            key, subset = row
            value = key[0] if isinstance(key, tuple) else key
            labels = subset[label_col].to_numpy()
            scores = subset[score_col].to_numpy()
            metrics = binary_classification_metrics(labels, scores).to_dict()
            output: dict[str, Any] = {
                "subgroup_col": subgroup_col,
                "subgroup_value": str(value),
                **metrics,
            }
            if metrics["auroc"] is None:
                output["warning"] = "AUROC undefined because subgroup has one observed class."
            results.append(output)
    return sorted(results, key=lambda item: (item["subgroup_col"], item["subgroup_value"]))
