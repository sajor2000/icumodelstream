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
