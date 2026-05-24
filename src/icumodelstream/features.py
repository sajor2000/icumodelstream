from __future__ import annotations

import polars as pl

from icumodelstream.io import TableRef, scan_table

VALUE_CANDIDATES = (
    "vital_value",        # CLIF 2.1 vitals table
    "lab_value_numeric",  # CLIF 2.1 labs table
    "value",
    "numeric_value",
    "measurement_value",
    "lab_value",
)
NAME_CANDIDATES = ("name", "variable", "lab_name", "vital_name", "category")


def _first_existing(columns: set[str], candidates: tuple[str, ...]) -> str | None:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


def aggregate_numeric_table(
    tables: dict[str, TableRef],
    table_name: str,
    prefix: str,
    cohort: pl.DataFrame | None = None,
    max_unparseable_fraction: float = 0.5,
) -> pl.DataFrame:
    """Create simple per-hospitalization numeric aggregates from a CLIF table.

    Raises ValueError if more than ``max_unparseable_fraction`` of originally-non-null values
    fail to cast to Float64 (CLAUDE.md rule 7: fail loudly on data assumptions).
    """
    lf = scan_table(tables, table_name)
    schema = lf.collect_schema()
    columns = set(schema.names())
    value_col = _first_existing(columns, VALUE_CANDIDATES)
    if value_col is None or "hospitalization_id" not in columns:
        raise ValueError(f"{table_name} must contain hospitalization_id and a numeric value column")

    counts = lf.select(
        pl.len().alias("n_total"),
        pl.col(value_col).is_not_null().sum().alias("n_non_null"),
        pl.col(value_col).cast(pl.Float64, strict=False).is_not_null().sum().alias("n_numeric"),
    ).collect().row(0, named=True)
    n_non_null = counts["n_non_null"]
    n_parse_failures = n_non_null - counts["n_numeric"]
    if n_non_null > 0:
        unparseable_fraction = n_parse_failures / n_non_null
        if unparseable_fraction > max_unparseable_fraction:
            raise ValueError(
                f"{table_name}.{value_col}: {n_parse_failures} of {n_non_null} non-null values "
                f"are unparseable as Float64 ({unparseable_fraction:.0%} > "
                f"{max_unparseable_fraction:.0%} threshold). Source dtype: {schema[value_col]}. "
                f"Check VALUE_CANDIDATES ordering or pre-clean the column."
            )

    out = (
        lf.with_columns(pl.col(value_col).cast(pl.Float64, strict=False).alias("_value"))
        .group_by("hospitalization_id")
        .agg(
            pl.col("_value").mean().alias(f"{prefix}_mean"),
            pl.col("_value").min().alias(f"{prefix}_min"),
            pl.col("_value").max().alias(f"{prefix}_max"),
            pl.len().alias(f"{prefix}_n"),
        )
        .collect()
    )
    if cohort is not None:
        cohort_dtype = cohort.schema["hospitalization_id"]
        out_dtype = out.schema["hospitalization_id"]
        if cohort_dtype != out_dtype:
            raise ValueError(
                f"hospitalization_id dtype mismatch joining cohort to {table_name}: "
                f"cohort={cohort_dtype}, {table_name}={out_dtype}. "
                f"Cast IDs to a common type before calling aggregate_numeric_table."
            )
        out = cohort.select("hospitalization_id").join(out, on="hospitalization_id", how="left")
        # Hospitalizations with zero measurements get null _n from the left join.
        # Coerce to 0 so callers can distinguish "no data" (n=0) from "couldn't compute".
        # mean/min/max stay null -- they're undefined without data.
        out = out.with_columns(pl.col(f"{prefix}_n").fill_null(0))
    return out.sort("hospitalization_id")
