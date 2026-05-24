from pathlib import Path

import pytest
import polars as pl

from icumodelstream.features import aggregate_numeric_table
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
