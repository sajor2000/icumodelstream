"""Convert CLIF tables into per-hospitalization sequence tensors.

Unit U1 of Phase 5. Reads the same CLIF tables that ``pipeline.py`` aggregates
into a flat feature matrix, but instead produces a 3-D tensor
``(n_hospitalizations, window_hours, n_channels)`` suitable for sequence
models (LSTM, Transformer). The numeric channels come with a parallel mask
tensor that tells the model which entries are real observations versus
forward-filled or zero-filled.

CLAUDE.md rule 2 (simplicity first): builds the tensor with the existing
polars + numpy stack; no torch dependency here. CLAUDE.md rule 7 (fail loudly
on data assumptions): timezone, dtype, and empty-cohort failures raise
ValueError with the same contract as :func:`aggregate_numeric_table_per_category`.

Forward-fill semantics: numeric channels are forward-filled along the time
axis so the model sees the most-recent value at every hour. Leading nulls
(hours before the FIRST observation for a given hospitalization-channel
pair) are filled with ``0.0``. This is a deliberate choice over
backward-filling: backward-fill would leak future information into earlier
hours. Callers that need a different imputation should consume the mask
channel and re-impute downstream.

Leakage contract: rows at exactly ``anchor + window_hours`` are EXCLUDED
(half-open interval ``[anchor, anchor + window_hours)``) — same contract as
:func:`aggregate_numeric_table_windowed`. This is the load-bearing safeguard
for "do not see the future".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl

from icumodelstream.features import (
    DATETIME_CANDIDATES,
    VALUE_CANDIDATES,
    _first_existing,
    _safe_category_name,
)
from icumodelstream.io import TableRef, scan_table
from icumodelstream.pipeline import (
    RICH_ASSESSMENT_CATEGORIES,
    RICH_LAB_CATEGORIES,
    RICH_RESPIRATORY_DEVICES,
    RICH_VITAL_CATEGORIES,
)


@dataclass(frozen=True)
class SequenceTensors:
    """Per-hospitalization tensors ready for a sequence model.

    Attributes
    ----------
    X:
        Float32 array of shape ``(n_hospitalizations, window_hours, n_channels)``.
        Numeric channels are forward-filled with leading-null = 0.0;
        respiratory indicator channels are 0/1 (no fill).
    mask:
        Int8 array of shape ``(n_hospitalizations, window_hours, n_numeric_channels)``.
        ``1`` where the source table had at least one observation in that
        ``(hospitalization, hour, channel)`` cell, else ``0``. NOT forward-filled.
        Indicator (respiratory) channels are intentionally excluded from the
        mask -- the indicator value itself encodes presence.
    hospitalization_ids:
        1-D object array (length ``n_hospitalizations``) of hospitalization IDs
        in the order they appear along axis 0 of ``X`` and ``mask``.
        Sorted lexicographically for deterministic ordering.
    channel_names:
        Length ``n_channels`` list, one name per channel column of ``X``.
        Ordering: vitals, then labs, then assessments, then respiratory
        indicators -- each block in the order of the corresponding category
        tuple passed to :func:`build_sequence_tensors`.
    numeric_channel_names:
        Subset of ``channel_names`` corresponding to columns that have a mask
        channel. ``numeric_channel_names == channel_names[:n_numeric]``.
    """

    X: np.ndarray
    mask: np.ndarray
    hospitalization_ids: np.ndarray
    channel_names: list[str]
    numeric_channel_names: list[str]


def _validate_anchors(anchors: pl.DataFrame) -> None:
    required_anchor_cols = {"hospitalization_id", "anchor_dttm"}
    missing = required_anchor_cols - set(anchors.columns)
    if missing:
        raise ValueError(
            f"anchors is missing required columns: {sorted(missing)}. "
            f"Expected {sorted(required_anchor_cols)}; got {anchors.columns}."
        )
    anchor_dtype = anchors.schema["anchor_dttm"]
    if not isinstance(anchor_dtype, pl.Datetime):
        raise ValueError(f"anchors.anchor_dttm must be Datetime, got {anchor_dtype}.")


def _per_hour_means(
    tables: dict[str, TableRef],
    table_name: str,
    category_column: str,
    categories: tuple[str, ...],
    anchors: pl.DataFrame,
    cohort_ids: pl.DataFrame,
    window_hours: int,
) -> pl.DataFrame:
    """Per-hospitalization per-category per-hour mean from a CLIF table.

    Returns long-format columns ``[hospitalization_id, hour, category, value, observed]``
    restricted to ``[anchor, anchor + window_hours)`` and to ``cohort_ids``.
    ``observed`` is 1 if the mean was computed from at least one row in that
    hour-bin, else 0.
    """
    lf = scan_table(tables, table_name)
    schema = lf.collect_schema()
    columns = set(schema.names())

    value_col = _first_existing(columns, VALUE_CANDIDATES)
    if value_col is None:
        raise ValueError(
            f"{table_name} has no recognized value column. "
            f"Expected one of {VALUE_CANDIDATES}; observed: {sorted(columns)}."
        )
    if "hospitalization_id" not in columns:
        raise ValueError(f"{table_name} must contain hospitalization_id")
    if category_column not in columns:
        raise ValueError(f"{table_name}.{category_column} not found; observed: {sorted(columns)}.")

    ts_col = _first_existing(columns, DATETIME_CANDIDATES)
    if ts_col is None:
        raise ValueError(
            f"{table_name} has no recognized datetime column for windowing. "
            f"Expected one of {DATETIME_CANDIDATES}; observed: {sorted(columns)}."
        )
    ts_dtype = schema[ts_col]
    if not isinstance(ts_dtype, pl.Datetime):
        raise ValueError(f"{table_name}.{ts_col} must be Datetime for windowing, got {ts_dtype}.")

    anchor_dtype = anchors.schema["anchor_dttm"]
    if ts_dtype.time_zone != anchor_dtype.time_zone:
        raise ValueError(
            f"Timezone mismatch between {table_name}.{ts_col} ({ts_dtype}) "
            f"and anchors.anchor_dttm ({anchor_dtype}). Both must be tz-aware "
            f"(same tz) or both naive."
        )
    src_id_dtype = schema["hospitalization_id"]
    anchor_id_dtype = anchors.schema["hospitalization_id"]
    if src_id_dtype != anchor_id_dtype:
        raise ValueError(
            f"hospitalization_id dtype mismatch joining anchors to {table_name}: "
            f"anchors={anchor_id_dtype}, {table_name}={src_id_dtype}."
        )

    anchors_lf = anchors.select("hospitalization_id", "anchor_dttm").lazy()
    cohort_lf = cohort_ids.lazy()

    long = (
        lf.join(cohort_lf, on="hospitalization_id", how="inner")
        .join(anchors_lf, on="hospitalization_id", how="inner")
        .filter(
            (pl.col(ts_col) >= pl.col("anchor_dttm"))
            & (pl.col(ts_col) < pl.col("anchor_dttm") + pl.duration(hours=window_hours))
            & pl.col(category_column).is_in(list(categories))
        )
        .with_columns(
            pl.col(value_col).cast(pl.Float64, strict=False).alias("_value"),
            # Hour index: integer floor of elapsed hours since anchor.
            ((pl.col(ts_col) - pl.col("anchor_dttm")).dt.total_seconds() // 3600)
            .cast(pl.Int64)
            .alias("hour"),
        )
        .group_by(["hospitalization_id", "hour", category_column])
        .agg(
            pl.col("_value").mean().alias("value"),
            pl.lit(1, dtype=pl.Int8).alias("observed"),
        )
        .rename({category_column: "category"})
        .collect()
    )
    return long


def _per_hour_indicator(
    tables: dict[str, TableRef],
    table_name: str,
    category_column: str,
    categories: tuple[str, ...],
    anchors: pl.DataFrame,
    cohort_ids: pl.DataFrame,
    window_hours: int,
) -> pl.DataFrame:
    """Per-hospitalization per-category per-hour presence flag (0/1) from a CLIF table.

    Returns long-format columns ``[hospitalization_id, hour, category, value]``
    where ``value == 1`` for every (hosp, hour, category) cell that had at
    least one source row. No mean; no mask.
    """
    lf = scan_table(tables, table_name)
    schema = lf.collect_schema()
    columns = set(schema.names())

    if "hospitalization_id" not in columns or category_column not in columns:
        raise ValueError(
            f"{table_name} must contain hospitalization_id and {category_column}; "
            f"observed: {sorted(columns)}."
        )
    ts_col = _first_existing(columns, DATETIME_CANDIDATES)
    if ts_col is None:
        raise ValueError(
            f"{table_name} has no recognized datetime column; "
            f"expected one of {DATETIME_CANDIDATES}; observed: {sorted(columns)}."
        )

    anchors_lf = anchors.select("hospitalization_id", "anchor_dttm").lazy()
    cohort_lf = cohort_ids.lazy()

    long = (
        lf.join(cohort_lf, on="hospitalization_id", how="inner")
        .join(anchors_lf, on="hospitalization_id", how="inner")
        .filter(
            (pl.col(ts_col) >= pl.col("anchor_dttm"))
            & (pl.col(ts_col) < pl.col("anchor_dttm") + pl.duration(hours=window_hours))
            & pl.col(category_column).is_in(list(categories))
        )
        .with_columns(
            ((pl.col(ts_col) - pl.col("anchor_dttm")).dt.total_seconds() // 3600)
            .cast(pl.Int64)
            .alias("hour"),
        )
        .group_by(["hospitalization_id", "hour", category_column])
        .agg(pl.lit(1.0, dtype=pl.Float64).alias("value"))
        .rename({category_column: "category"})
        .collect()
    )
    return long


def _forward_fill_with_zero_leading(values: np.ndarray) -> np.ndarray:
    """Forward-fill NaNs along the time axis (axis=1), filling leading NaNs with 0.

    ``values`` has shape ``(n_hosps, window_hours, n_channels)``. Operates
    independently per (hospitalization, channel) trajectory.
    """
    out = values.copy()
    n_hosps, n_t, n_c = out.shape
    # Forward-fill: for each (i, c), walk t from 0 to n_t-1 carrying last non-nan.
    # Vectorize using the standard "valid index" trick along axis=1.
    is_valid = ~np.isnan(out)
    # idx[i, t, c] = t if valid else 0; then cummax along t gives last seen valid index.
    t_grid = np.arange(n_t).reshape(1, n_t, 1)
    idx = np.where(is_valid, t_grid, 0)
    last_valid = np.maximum.accumulate(idx, axis=1)
    # Pull values at last_valid; broadcast-friendly gather along axis=1.
    # Use np.take_along_axis with a 3-D index.
    gathered = np.take_along_axis(out, last_valid, axis=1)
    # Where the trajectory has NEVER seen a valid value up to time t,
    # cummax stays 0 AND values[..., 0, ...] may still be NaN — fill those with 0.
    gathered = np.where(np.isnan(gathered), 0.0, gathered)
    return gathered


def build_sequence_tensors(
    tables: dict[str, TableRef],
    cohort: pl.DataFrame,
    anchors: pl.DataFrame,
    window_hours: int = 24,
    vital_categories: tuple[str, ...] = RICH_VITAL_CATEGORIES,
    lab_categories: tuple[str, ...] = RICH_LAB_CATEGORIES,
    assessment_categories: tuple[str, ...] = RICH_ASSESSMENT_CATEGORIES,
    respiratory_devices: tuple[str, ...] = RICH_RESPIRATORY_DEVICES,
) -> SequenceTensors:
    """Build per-hospitalization sequence tensors from CLIF tables.

    For each hospitalization in ``cohort``, builds an ``(window_hours, n_channels)``
    matrix where each channel is one of:

    1. A numeric vital category (mean per hour, mask channel records presence)
    2. A numeric lab category (mean per hour, mask channel records presence)
    3. A numeric patient_assessment category (mean per hour, mask channel records
       presence)
    4. A respiratory device indicator (1 if any source row in that hour, else 0;
       NO mask channel -- the indicator itself encodes presence)

    Numeric channels (1-3) are forward-filled along the time axis and any
    remaining leading NaNs are replaced with ``0.0``. The mask tensor is NOT
    forward-filled; it reflects original observation locations.

    Parameters
    ----------
    tables:
        Discovered CLIF parquet tables (from :func:`discover_tables`).
    cohort:
        DataFrame with at least ``hospitalization_id``. Determines which
        hospitalizations appear along axis 0 of the output tensors.
    anchors:
        DataFrame with ``hospitalization_id`` and ``anchor_dttm`` (Datetime).
        The window is ``[anchor_dttm, anchor_dttm + window_hours)``.
    window_hours:
        Length of the per-hospitalization window in integer hours. Must be > 0.
    vital_categories, lab_categories, assessment_categories, respiratory_devices:
        Category tuples that drive channel ordering. Defaults are imported from
        :mod:`icumodelstream.pipeline` so they stay in sync with the baseline
        feature set.

    Returns
    -------
    SequenceTensors

    Raises
    ------
    ValueError
        - Empty ``cohort``.
        - ``window_hours <= 0``.
        - ``anchors.anchor_dttm`` is not Datetime, or its timezone disagrees
          with the source timestamp column's timezone (silent timezone
          mismatches mis-align windows -- fail loudly per CLAUDE.md rule 7).
        - ``cohort.hospitalization_id`` dtype disagrees with the source's
          ``hospitalization_id`` dtype.
    """
    if window_hours <= 0:
        raise ValueError(f"window_hours must be > 0; got {window_hours}.")
    if "hospitalization_id" not in cohort.columns:
        raise ValueError(f"cohort is missing hospitalization_id; got columns: {cohort.columns}.")
    if cohort.height == 0:
        raise ValueError("Cohort is empty; cannot build sequence tensors.")
    _validate_anchors(anchors)

    cohort_id_dtype = cohort.schema["hospitalization_id"]
    anchor_id_dtype = anchors.schema["hospitalization_id"]
    if cohort_id_dtype != anchor_id_dtype:
        raise ValueError(
            f"hospitalization_id dtype mismatch between cohort ({cohort_id_dtype}) "
            f"and anchors ({anchor_id_dtype}); cast both sides to a common type."
        )

    # Restrict to cohort hospitalizations that also have an anchor.
    cohort_with_anchor = (
        cohort.select("hospitalization_id")
        .unique()
        .join(
            anchors.select("hospitalization_id").unique(),
            on="hospitalization_id",
            how="inner",
        )
    )
    if cohort_with_anchor.height == 0:
        raise ValueError("No cohort hospitalizations have anchors; cannot build sequence tensors.")

    # Deterministic row order: sort by hospitalization_id (lexicographic for str,
    # ascending for numeric).
    cohort_with_anchor = cohort_with_anchor.sort("hospitalization_id")
    hosp_ids = cohort_with_anchor["hospitalization_id"].to_list()
    hosp_id_to_idx: dict[Any, int] = {hid: i for i, hid in enumerate(hosp_ids)}

    cohort_ids = cohort_with_anchor.select("hospitalization_id")

    # ---- Channel layout ----
    vital_channel_names = [f"vitals_{_safe_category_name(c)}" for c in vital_categories]
    lab_channel_names = [f"labs_{_safe_category_name(c)}" for c in lab_categories]
    assess_channel_names = [f"assess_{_safe_category_name(c)}" for c in assessment_categories]
    resp_channel_names = [f"resp_{_safe_category_name(c)}" for c in respiratory_devices]

    numeric_channel_names = vital_channel_names + lab_channel_names + assess_channel_names
    indicator_channel_names = resp_channel_names
    channel_names = numeric_channel_names + indicator_channel_names

    n_numeric = len(numeric_channel_names)
    n_indicator = len(indicator_channel_names)
    n_hosps = len(hosp_ids)

    # Allocate the full output tensor. Numeric region starts as NaN so
    # forward-fill can identify "never observed". Indicator region starts as 0
    # (presence flag default).
    X_numeric = np.full((n_hosps, window_hours, n_numeric), np.nan, dtype=np.float64)
    mask_numeric = np.zeros((n_hosps, window_hours, n_numeric), dtype=np.int8)
    X_indicator = np.zeros((n_hosps, window_hours, n_indicator), dtype=np.float64)

    # ---- Scatter numeric channels ----
    numeric_specs = [
        ("vitals", "vital_category", vital_categories, 0),
        ("labs", "lab_category", lab_categories, len(vital_categories)),
        (
            "patient_assessments",
            "assessment_category",
            assessment_categories,
            len(vital_categories) + len(lab_categories),
        ),
    ]
    for table_name, cat_col, categories, channel_offset in numeric_specs:
        if not categories:
            continue
        if table_name not in tables:
            # Missing table is allowed; that block stays all-NaN -> forward-fills to 0.
            continue
        long = _per_hour_means(
            tables,
            table_name,
            category_column=cat_col,
            categories=categories,
            anchors=anchors,
            cohort_ids=cohort_ids,
            window_hours=window_hours,
        )
        if long.height == 0:
            continue
        cat_to_global_idx = {cat: channel_offset + i for i, cat in enumerate(categories)}
        for row in long.iter_rows(named=True):
            hosp = row["hospitalization_id"]
            h = row["hour"]
            cat = row["category"]
            if hosp not in hosp_id_to_idx or cat not in cat_to_global_idx:
                continue
            if h is None or h < 0 or h >= window_hours:
                continue
            i = hosp_id_to_idx[hosp]
            c = cat_to_global_idx[cat]
            v = row["value"]
            if v is not None:
                X_numeric[i, int(h), c] = v
            mask_numeric[i, int(h), c] = 1

    # ---- Scatter indicator channels ----
    if respiratory_devices and "respiratory_support" in tables:
        long_resp = _per_hour_indicator(
            tables,
            "respiratory_support",
            category_column="device_category",
            categories=respiratory_devices,
            anchors=anchors,
            cohort_ids=cohort_ids,
            window_hours=window_hours,
        )
        cat_to_resp_idx = {cat: i for i, cat in enumerate(respiratory_devices)}
        for row in long_resp.iter_rows(named=True):
            hosp = row["hospitalization_id"]
            h = row["hour"]
            cat = row["category"]
            if hosp not in hosp_id_to_idx or cat not in cat_to_resp_idx:
                continue
            if h is None or h < 0 or h >= window_hours:
                continue
            i = hosp_id_to_idx[hosp]
            c = cat_to_resp_idx[cat]
            X_indicator[i, int(h), c] = 1.0

    # ---- Forward-fill numeric ----
    X_numeric_filled = _forward_fill_with_zero_leading(X_numeric)

    # ---- Combine into final tensor ----
    X = np.concatenate([X_numeric_filled, X_indicator], axis=2).astype(np.float32)

    hospitalization_ids = np.array(hosp_ids, dtype=object)

    return SequenceTensors(
        X=X,
        mask=mask_numeric,
        hospitalization_ids=hospitalization_ids,
        channel_names=channel_names,
        numeric_channel_names=numeric_channel_names,
    )
