from pathlib import Path

import polars as pl
import pytest

from icumodelstream.io import discover_tables, read_table, table_name_from_path


def test_table_name_from_path() -> None:
    assert table_name_from_path(Path("patient.parquet")) == "patient"
    assert table_name_from_path(Path("HOSPITALIZATION.parquet")) == "hospitalization"
    assert table_name_from_path(Path("clif_patient.parquet")) == "patient"
    assert table_name_from_path(Path("clif_medication_admin_continuous.parquet")) == "medication_admin_continuous"


def test_discover_and_read_tables(tmp_path: Path) -> None:
    pl.DataFrame({"patient_id": [1, 2]}).write_parquet(tmp_path / "patient.parquet")
    tables = discover_tables(tmp_path)
    assert "patient" in tables
    assert read_table(tables, "patient").height == 2


def test_discover_tables_with_clif_prefix(tmp_path: Path) -> None:
    pl.DataFrame({"patient_id": [1]}).write_parquet(tmp_path / "clif_patient.parquet")
    tables = discover_tables(tmp_path)
    assert "patient" in tables
    assert "clif_patient" not in tables
    assert read_table(tables, "patient").height == 1


def test_discover_tables_raises_on_duplicate_key(tmp_path: Path) -> None:
    pl.DataFrame({"patient_id": [1]}).write_parquet(tmp_path / "patient.parquet")
    pl.DataFrame({"patient_id": [2]}).write_parquet(tmp_path / "clif_patient.parquet")
    with pytest.raises(ValueError, match="Duplicate table name"):
        discover_tables(tmp_path)


def test_table_name_from_path_strips_known_prefix_variants() -> None:
    """CLIF exports use varied prefixes. Strip them so downstream code can look up by bare key."""
    assert table_name_from_path(Path("clif2_patient.parquet")) == "patient"
    assert table_name_from_path(Path("mimic_clif_hospitalization.parquet")) == "hospitalization"
    assert table_name_from_path(Path("rush_clif_adt.parquet")) == "adt"
