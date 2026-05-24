from pathlib import Path

import polars as pl

from icumodelstream.io import discover_tables
from icumodelstream.schema import validate_table_contracts


def test_validate_core_contracts(tmp_path: Path) -> None:
    pl.DataFrame({"patient_id": [1]}).write_parquet(tmp_path / "patient.parquet")
    pl.DataFrame({"patient_id": [1], "hospitalization_id": [10]}).write_parquet(
        tmp_path / "hospitalization.parquet"
    )
    pl.DataFrame({"hospitalization_id": [10], "location_category": ["ICU"]}).write_parquet(
        tmp_path / "adt.parquet"
    )

    tables = discover_tables(tmp_path)
    results = validate_table_contracts(tables)
    assert all(result.ok for result in results)
