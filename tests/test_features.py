from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import polars as pl

from icumodelstream.features import (
    aggregate_numeric_table,
    aggregate_numeric_table_windowed,
)
from icumodelstream.io import discover_tables


def test_aggregate_numeric_table_happy_path(tmp_path: Path) -> None:
    pl.DataFrame(
        {"hospitalization_id": [1, 1, 2], "value": [10.0, 20.0, 30.0]}
    ).write_parquet(tmp_path / "vitals.parquet")

    tables = discover_tables(tmp_path)
    result = aggregate_numeric_table(tables, "vitals", "vitals")
    result = result.sort("hospitalization_id")

    assert result["vitals_mean"].to_list() == [15.0, 30.0]
    assert result["vitals_min"].to_list() == [10.0, 30.0]
    assert result["vitals_max"].to_list() == [20.0, 30.0]
    assert result["vitals_n"].to_list() == [2, 1]


def test_aggregate_numeric_table_vital_value_column(tmp_path: Path) -> None:
    pl.DataFrame(
        {"hospitalization_id": [1], "vital_value": [42.5]}
    ).write_parquet(tmp_path / "vitals.parquet")

    tables = discover_tables(tmp_path)
    result = aggregate_numeric_table(tables, "vitals", "vitals")

    assert len(result) == 1
    assert result["vitals_mean"][0] == pytest.approx(42.5)


def test_aggregate_numeric_table_lab_value_numeric_column(tmp_path: Path) -> None:
    pl.DataFrame(
        {"hospitalization_id": [1], "lab_value_numeric": [99.0]}
    ).write_parquet(tmp_path / "labs.parquet")

    tables = discover_tables(tmp_path)
    result = aggregate_numeric_table(tables, "labs", "labs")

    assert len(result) == 1
    assert result["labs_mean"][0] == pytest.approx(99.0)


def test_aggregate_numeric_table_cohort_filter(tmp_path: Path) -> None:
    pl.DataFrame(
        {"hospitalization_id": [1, 2], "value": [5.0, 99.0]}
    ).write_parquet(tmp_path / "vitals.parquet")

    tables = discover_tables(tmp_path)
    cohort = pl.DataFrame({"hospitalization_id": [1]})
    result = aggregate_numeric_table(tables, "vitals", "vitals", cohort=cohort)

    assert len(result) == 1
    assert result["hospitalization_id"][0] == 1
    assert result["vitals_mean"][0] == pytest.approx(5.0)


def test_aggregate_numeric_table_raises_on_missing_value_col(tmp_path: Path) -> None:
    pl.DataFrame(
        {"hospitalization_id": [1]}
    ).write_parquet(tmp_path / "vitals.parquet")

    tables = discover_tables(tmp_path)

    with pytest.raises(ValueError):
        aggregate_numeric_table(tables, "vitals", "vitals")


def test_aggregate_numeric_table_prefers_numeric_over_legacy_value_col(tmp_path: Path) -> None:
    """When both lab_value (legacy, possibly string) and lab_value_numeric are present,
    the numeric column must win — otherwise CLIF 2.1 labs aggregate to all-null."""
    pl.DataFrame({
        "hospitalization_id": [1, 1, 2],
        "lab_value": ["12.3 mg/dL", "15.1 mg/dL", "5.0 mg/dL"],
        "lab_value_numeric": [12.3, 15.1, 5.0],
    }).write_parquet(tmp_path / "labs.parquet")

    tables = discover_tables(tmp_path)
    result = aggregate_numeric_table(tables, "labs", "labs").sort("hospitalization_id")

    assert result["labs_mean"].to_list() == [pytest.approx(13.7), pytest.approx(5.0)]
    assert result["labs_n"].to_list() == [2, 1]


def test_aggregate_numeric_table_raises_on_unparseable_value_col(tmp_path: Path) -> None:
    """A string-typed value column with non-numeric content must raise, not silently null."""
    pl.DataFrame({
        "hospitalization_id": [1, 1, 2, 2],
        "value": ["120/80", "130/85", "118/78", "125/82"],
    }).write_parquet(tmp_path / "vitals.parquet")

    tables = discover_tables(tmp_path)

    with pytest.raises(ValueError, match="unparseable"):
        aggregate_numeric_table(tables, "vitals", "vitals")


def test_aggregate_numeric_table_n_uses_raw_row_count(tmp_path: Path) -> None:
    """_n should reflect raw row count, not null-excluding count — otherwise filters like
    `n >= 50` drop hospitalizations whose values mostly survived."""
    pl.DataFrame({
        "hospitalization_id": [1, 1, 1, 1],
        "numeric_value": [10.0, 20.0, None, None],
    }).write_parquet(tmp_path / "vitals.parquet")

    tables = discover_tables(tmp_path)
    result = aggregate_numeric_table(tables, "vitals", "vitals")

    assert result["vitals_n"][0] == 4


def test_aggregate_numeric_table_raises_on_id_dtype_mismatch(tmp_path: Path) -> None:
    """If cohort.hospitalization_id dtype != source.hospitalization_id dtype, raise with a
    helpful message rather than letting polars raise a cryptic SchemaError."""
    pl.DataFrame({
        "hospitalization_id": ["A1", "A2"],
        "value": [10.0, 20.0],
    }).write_parquet(tmp_path / "vitals.parquet")

    tables = discover_tables(tmp_path)
    cohort = pl.DataFrame({"hospitalization_id": [1, 2]})  # Int64 vs Utf8

    with pytest.raises(ValueError, match="hospitalization_id"):
        aggregate_numeric_table(tables, "vitals", "vitals", cohort=cohort)


def test_aggregate_numeric_table_cohort_left_join_preserves_missing(tmp_path: Path) -> None:
    """Cohort hospitalizations with no rows in the source table should be preserved.

    Distinguish "no data" (n=0) from "couldn't compute" (mean/min/max=null):
    - _n is coerced to 0 so downstream filters like `n >= 50` work correctly
    - mean/min/max stay null because they're genuinely undefined without data
    """
    pl.DataFrame(
        {"hospitalization_id": [1, 2], "value": [10.0, 20.0]}
    ).write_parquet(tmp_path / "vitals.parquet")

    tables = discover_tables(tmp_path)
    cohort = pl.DataFrame({"hospitalization_id": [1, 2, 3]})
    result = aggregate_numeric_table(tables, "vitals", "vitals", cohort=cohort).sort(
        "hospitalization_id"
    )

    assert result.height == 3
    row3 = result.filter(pl.col("hospitalization_id") == 3).row(0, named=True)
    assert row3["vitals_mean"] is None
    assert row3["vitals_min"] is None
    assert row3["vitals_max"] is None
    assert row3["vitals_n"] == 0


# ---------------------------------------------------------------------------
# aggregate_numeric_table_windowed
# ---------------------------------------------------------------------------


def _write_vitals_with_dttm(
    path: Path,
    hospitalization_ids: list[int],
    recorded_dttms: list[datetime],
    values: list[float],
) -> None:
    df = pl.DataFrame(
        {
            "hospitalization_id": hospitalization_ids,
            "recorded_dttm": recorded_dttms,
            "value": values,
        }
    ).with_columns(pl.col("recorded_dttm").cast(pl.Datetime(time_zone="UTC")))
    df.write_parquet(path)


def test_aggregate_windowed_happy_path(tmp_path: Path) -> None:
    """Rows within [anchor, anchor + 24h) are aggregated; rows outside are dropped."""
    anchor = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
    _write_vitals_with_dttm(
        tmp_path / "vitals.parquet",
        hospitalization_ids=[1, 1, 1, 1],
        recorded_dttms=[
            anchor + timedelta(hours=1),   # in
            anchor + timedelta(hours=12),  # in
            anchor + timedelta(hours=23),  # in
            anchor + timedelta(hours=25),  # out
        ],
        values=[10.0, 20.0, 30.0, 999.0],
    )

    tables = discover_tables(tmp_path)
    anchors = pl.DataFrame(
        {"hospitalization_id": [1], "anchor_dttm": [anchor]}
    ).with_columns(pl.col("anchor_dttm").cast(pl.Datetime(time_zone="UTC")))

    result = aggregate_numeric_table_windowed(
        tables, "vitals", "vitals", anchors=anchors, window_hours=24
    )

    assert result.height == 1
    row = result.row(0, named=True)
    assert row["vitals_n"] == 3
    assert row["vitals_mean"] == pytest.approx(20.0)
    assert row["vitals_min"] == pytest.approx(10.0)
    assert row["vitals_max"] == pytest.approx(30.0)


def test_aggregate_windowed_excludes_boundary(tmp_path: Path) -> None:
    """Row at exactly anchor + window_hours is excluded (half-open interval)."""
    anchor = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
    _write_vitals_with_dttm(
        tmp_path / "vitals.parquet",
        hospitalization_ids=[1, 1],
        recorded_dttms=[
            anchor + timedelta(hours=0),   # exactly anchor: in
            anchor + timedelta(hours=24),  # exactly anchor + window: out
        ],
        values=[10.0, 99.0],
    )

    tables = discover_tables(tmp_path)
    anchors = pl.DataFrame(
        {"hospitalization_id": [1], "anchor_dttm": [anchor]}
    ).with_columns(pl.col("anchor_dttm").cast(pl.Datetime(time_zone="UTC")))

    result = aggregate_numeric_table_windowed(
        tables, "vitals", "vitals", anchors=anchors, window_hours=24
    )

    assert result.height == 1
    assert result["vitals_n"][0] == 1
    assert result["vitals_max"][0] == pytest.approx(10.0)


def test_aggregate_windowed_excludes_pre_anchor(tmp_path: Path) -> None:
    """Row at anchor - 1h is excluded; pre-anchor data must not leak in."""
    anchor = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)
    _write_vitals_with_dttm(
        tmp_path / "vitals.parquet",
        hospitalization_ids=[1, 1],
        recorded_dttms=[
            anchor - timedelta(hours=1),  # before anchor: out
            anchor + timedelta(hours=2),  # in
        ],
        values=[99.0, 42.0],
    )

    tables = discover_tables(tmp_path)
    anchors = pl.DataFrame(
        {"hospitalization_id": [1], "anchor_dttm": [anchor]}
    ).with_columns(pl.col("anchor_dttm").cast(pl.Datetime(time_zone="UTC")))

    result = aggregate_numeric_table_windowed(
        tables, "vitals", "vitals", anchors=anchors, window_hours=24
    )

    assert result.height == 1
    assert result["vitals_n"][0] == 1
    assert result["vitals_mean"][0] == pytest.approx(42.0)


def test_aggregate_windowed_cohort_left_join_preserves_empty(tmp_path: Path) -> None:
    """Cohort hospitalizations with zero in-window rows get _n=0 and null stats."""
    anchor = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
    _write_vitals_with_dttm(
        tmp_path / "vitals.parquet",
        hospitalization_ids=[1, 2, 3],
        recorded_dttms=[
            anchor + timedelta(hours=1),   # h1: in-window
            anchor + timedelta(hours=99),  # h2: out-of-window
            anchor + timedelta(hours=2),   # h3: in-window but not in cohort
        ],
        values=[10.0, 20.0, 30.0],
    )

    tables = discover_tables(tmp_path)
    # Cohort has h1, h2 only. h2 has 0 in-window rows -> _n=0 in result.
    cohort = pl.DataFrame({"hospitalization_id": [1, 2]})
    anchors = pl.DataFrame(
        {"hospitalization_id": [1, 2], "anchor_dttm": [anchor, anchor]}
    ).with_columns(pl.col("anchor_dttm").cast(pl.Datetime(time_zone="UTC")))

    result = aggregate_numeric_table_windowed(
        tables, "vitals", "vitals", anchors=anchors, window_hours=24, cohort=cohort
    )

    assert result.height == 2
    assert result["hospitalization_id"].to_list() == [1, 2]
    row1 = result.filter(pl.col("hospitalization_id") == 1).row(0, named=True)
    row2 = result.filter(pl.col("hospitalization_id") == 2).row(0, named=True)
    assert row1["vitals_n"] == 1
    assert row1["vitals_mean"] == pytest.approx(10.0)
    assert row2["vitals_n"] == 0
    assert row2["vitals_mean"] is None
    assert row2["vitals_min"] is None
    assert row2["vitals_max"] is None


def test_aggregate_windowed_raises_on_tz_mismatch(tmp_path: Path) -> None:
    """Tz-naive anchors vs tz-aware source datetimes -> ValueError mentioning tz/dtype."""
    anchor = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
    _write_vitals_with_dttm(
        tmp_path / "vitals.parquet",
        hospitalization_ids=[1],
        recorded_dttms=[anchor + timedelta(hours=1)],
        values=[10.0],
    )

    tables = discover_tables(tmp_path)
    # Naive anchor (no tz_aware cast) -> mismatch.
    naive_anchor = datetime(2024, 1, 1, 0, 0)
    anchors = pl.DataFrame(
        {"hospitalization_id": [1], "anchor_dttm": [naive_anchor]}
    )

    with pytest.raises(ValueError, match="(?i)timezone|tz|dtype"):
        aggregate_numeric_table_windowed(
            tables, "vitals", "vitals", anchors=anchors, window_hours=24
        )


def test_aggregate_windowed_raises_on_missing_dttm_col(tmp_path: Path) -> None:
    """Source with no recognized *_dttm column -> ValueError."""
    pl.DataFrame(
        {"hospitalization_id": [1], "value": [10.0]}
    ).write_parquet(tmp_path / "vitals.parquet")

    tables = discover_tables(tmp_path)
    anchors = pl.DataFrame(
        {"hospitalization_id": [1], "anchor_dttm": [datetime(2024, 1, 1, tzinfo=timezone.utc)]}
    ).with_columns(pl.col("anchor_dttm").cast(pl.Datetime(time_zone="UTC")))

    with pytest.raises(ValueError, match="(?i)datetime|dttm"):
        aggregate_numeric_table_windowed(
            tables, "vitals", "vitals", anchors=anchors, window_hours=24
        )


# ---------------------------------------------------------------------------
# aggregate_numeric_table_per_category
# ---------------------------------------------------------------------------


def _write_vitals_with_category(
    path: Path,
    hospitalization_ids: list,
    categories: list[str],
    recorded_dttms: list[datetime],
    values: list[float],
) -> None:
    df = pl.DataFrame(
        {
            "hospitalization_id": hospitalization_ids,
            "vital_category": categories,
            "recorded_dttm": recorded_dttms,
            "vital_value": values,
        }
    ).with_columns(pl.col("recorded_dttm").cast(pl.Datetime(time_zone="UTC")))
    df.write_parquet(path)


def test_per_category_aggregation_happy_path(tmp_path: Path) -> None:
    """One column-block per category, each with mean/min/max/n."""
    from icumodelstream.features import aggregate_numeric_table_per_category

    anchor = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
    _write_vitals_with_category(
        tmp_path / "vitals.parquet",
        hospitalization_ids=[1, 1, 1, 1, 1],
        categories=["heart_rate", "heart_rate", "sbp", "sbp", "spo2"],
        recorded_dttms=[anchor + timedelta(hours=h) for h in [1, 2, 3, 4, 5]],
        values=[80.0, 90.0, 120.0, 130.0, 98.0],
    )
    tables = discover_tables(tmp_path)
    anchors = pl.DataFrame(
        {"hospitalization_id": [1], "anchor_dttm": [anchor]}
    ).with_columns(pl.col("anchor_dttm").cast(pl.Datetime(time_zone="UTC")))

    result = aggregate_numeric_table_per_category(
        tables, "vitals", "vital_category",
        categories=["heart_rate", "sbp", "spo2"],
        prefix_template="vitals_{category}",
        anchors=anchors, window_hours=24,
    )

    assert result.height == 1
    row = result.row(0, named=True)
    assert row["vitals_heart_rate_mean"] == pytest.approx(85.0)
    assert row["vitals_heart_rate_n"] == 2
    assert row["vitals_sbp_mean"] == pytest.approx(125.0)
    assert row["vitals_sbp_n"] == 2
    assert row["vitals_spo2_mean"] == pytest.approx(98.0)
    assert row["vitals_spo2_n"] == 1


def test_per_category_missing_category_is_zero_n(tmp_path: Path) -> None:
    """A category with no rows in window gets _n=0 and null mean/min/max."""
    from icumodelstream.features import aggregate_numeric_table_per_category

    anchor = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
    _write_vitals_with_category(
        tmp_path / "vitals.parquet",
        hospitalization_ids=[1, 1],
        categories=["heart_rate", "heart_rate"],
        recorded_dttms=[anchor + timedelta(hours=1), anchor + timedelta(hours=2)],
        values=[80.0, 90.0],
    )
    tables = discover_tables(tmp_path)
    anchors = pl.DataFrame(
        {"hospitalization_id": [1], "anchor_dttm": [anchor]}
    ).with_columns(pl.col("anchor_dttm").cast(pl.Datetime(time_zone="UTC")))

    result = aggregate_numeric_table_per_category(
        tables, "vitals", "vital_category",
        categories=["heart_rate", "lactate"],  # lactate has no rows
        prefix_template="vitals_{category}",
        anchors=anchors, window_hours=24,
    )
    row = result.row(0, named=True)
    assert row["vitals_heart_rate_n"] == 2
    assert row["vitals_lactate_n"] == 0
    assert row["vitals_lactate_mean"] is None


def test_per_category_safe_column_name(tmp_path: Path) -> None:
    """Categories with spaces/hyphens produce safe column names."""
    from icumodelstream.features import _safe_category_name

    assert _safe_category_name("heart_rate") == "heart_rate"
    assert _safe_category_name("High Flow NC") == "high_flow_nc"
    assert _safe_category_name("Nasal-Cannula") == "nasal_cannula"


def test_per_category_respects_window(tmp_path: Path) -> None:
    """Rows outside [anchor, anchor + window_hours) are excluded per category."""
    from icumodelstream.features import aggregate_numeric_table_per_category

    anchor = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
    _write_vitals_with_category(
        tmp_path / "vitals.parquet",
        hospitalization_ids=[1, 1, 1],
        categories=["heart_rate", "heart_rate", "heart_rate"],
        recorded_dttms=[
            anchor + timedelta(hours=1),    # in
            anchor + timedelta(hours=23),   # in
            anchor + timedelta(hours=25),   # OUT
        ],
        values=[80.0, 90.0, 9999.0],
    )
    tables = discover_tables(tmp_path)
    anchors = pl.DataFrame(
        {"hospitalization_id": [1], "anchor_dttm": [anchor]}
    ).with_columns(pl.col("anchor_dttm").cast(pl.Datetime(time_zone="UTC")))

    result = aggregate_numeric_table_per_category(
        tables, "vitals", "vital_category",
        categories=["heart_rate"],
        prefix_template="vitals_{category}",
        anchors=anchors, window_hours=24,
    )
    row = result.row(0, named=True)
    assert row["vitals_heart_rate_n"] == 2  # the +25h row excluded
    assert row["vitals_heart_rate_max"] == pytest.approx(90.0)


# ---------------------------------------------------------------------------
# respiratory_support_indicator
# ---------------------------------------------------------------------------


def test_respiratory_indicator_happy_path(tmp_path: Path) -> None:
    """One Int8 0/1 column per device_category, indicating presence in window."""
    from icumodelstream.features import respiratory_support_indicator

    anchor = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
    pl.DataFrame(
        {
            "hospitalization_id": [1, 1, 2],
            "recorded_dttm": [
                anchor + timedelta(hours=1),
                anchor + timedelta(hours=5),
                anchor + timedelta(hours=2),
            ],
            "device_category": ["IMV", "Nasal Cannula", "Nasal Cannula"],
        }
    ).with_columns(pl.col("recorded_dttm").cast(pl.Datetime(time_zone="UTC"))).write_parquet(
        tmp_path / "respiratory_support.parquet"
    )
    tables = discover_tables(tmp_path)
    anchors = pl.DataFrame(
        {"hospitalization_id": [1, 2], "anchor_dttm": [anchor, anchor]}
    ).with_columns(pl.col("anchor_dttm").cast(pl.Datetime(time_zone="UTC")))

    result = respiratory_support_indicator(
        tables,
        device_categories=["IMV", "Nasal Cannula", "CPAP"],
        anchors=anchors, window_hours=24,
    ).sort("hospitalization_id")

    # Patient 1: had IMV and Nasal Cannula in window
    row1 = result.filter(pl.col("hospitalization_id") == 1).row(0, named=True)
    assert row1["resp_imv"] == 1
    assert row1["resp_nasal_cannula"] == 1
    assert row1["resp_cpap"] == 0
    # Patient 2: had only Nasal Cannula
    row2 = result.filter(pl.col("hospitalization_id") == 2).row(0, named=True)
    assert row2["resp_imv"] == 0
    assert row2["resp_nasal_cannula"] == 1
    assert row2["resp_cpap"] == 0


def test_respiratory_indicator_no_rows_in_window(tmp_path: Path) -> None:
    """A hospitalization with no respiratory_support rows in window gets all-zero flags."""
    from icumodelstream.features import respiratory_support_indicator

    anchor = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
    pl.DataFrame(
        {
            "hospitalization_id": [1],
            "recorded_dttm": [anchor + timedelta(hours=48)],  # outside 24h window
            "device_category": ["IMV"],
        }
    ).with_columns(pl.col("recorded_dttm").cast(pl.Datetime(time_zone="UTC"))).write_parquet(
        tmp_path / "respiratory_support.parquet"
    )
    tables = discover_tables(tmp_path)
    anchors = pl.DataFrame(
        {"hospitalization_id": [1], "anchor_dttm": [anchor]}
    ).with_columns(pl.col("anchor_dttm").cast(pl.Datetime(time_zone="UTC")))

    result = respiratory_support_indicator(
        tables, device_categories=["IMV"],
        anchors=anchors, window_hours=24,
    )
    assert result.row(0, named=True)["resp_imv"] == 0
