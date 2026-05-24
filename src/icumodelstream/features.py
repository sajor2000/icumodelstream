from __future__ import annotations

import polars as pl

from icumodelstream.io import TableRef, scan_table

VALUE_CANDIDATES = ("value", "numeric_value", "measurement_value", "lab_value")
NAME_CANDIDATES = ("name", "variable", "lab_name", "vital_name", "category")


def _first_existing(columns: set[str], candidates: tuple[str, ...]) -> str | None:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


def aggregate_numeric_table(
    tables: dict[str, TableRef], table_name: str, prefix: str, cohort: pl.DataFrame | None = None
) -> pl.DataFrame:
    """Create simple per-hospitalization numeric aggregates from a CLIF table.

    This is a baseline feature helper, not a final modeling representation.
    """
    lf = scan_table(tables, table_name)
    columns = set(lf.collect_schema().names())
    value_col = _first_existing(columns, VALUE_CANDIDATES)
    if value_col is None or "hospitalization_id" not in columns:
        raise ValueError(f"{table_name} must contain hospitalization_id and a numeric value column")

    out = (
        lf.with_columns(pl.col(value_col).cast(pl.Float64, strict=False).alias("_value"))
        .group_by("hospitalization_id")
        .agg(
            pl.col("_value").mean().alias(f"{prefix}_mean"),
            pl.col("_value").min().alias(f"{prefix}_min"),
            pl.col("_value").max().alias(f"{prefix}_max"),
            pl.col("_value").count().alias(f"{prefix}_n"),
        )
        .collect()
    )
    if cohort is not None:
        out = cohort.select("hospitalization_id").join(out, on="hospitalization_id", how="left")
    return out.sort("hospitalization_id")
