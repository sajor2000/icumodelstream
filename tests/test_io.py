from pathlib import Path

import polars as pl

from icumodelstream.io import discover_tables, read_table, table_name_from_path


def test_table_name_from_path() -> None:
    assert table_name_from_path(Path("patient.parquet")) == "patient"
    assert table_name_from_path(Path("HOSPITALIZATION.parquet")) == "hospitalization"


def test_discover_and_read_tables(tmp_path: Path) -> None:
    pl.DataFrame({"patient_id": [1, 2]}).write_parquet(tmp_path / "patient.parquet")
    tables = discover_tables(tmp_path)
    assert "patient" in tables
    assert read_table(tables, "patient").height == 2
