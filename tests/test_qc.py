from pathlib import Path

import polars as pl

from icumodelstream.io import discover_tables
from icumodelstream.qc import build_qc_report


def test_build_qc_report(tmp_path: Path) -> None:
    pl.DataFrame({"patient_id": [1, 2], "x": [None, "ok"]}).write_parquet(
        tmp_path / "patient.parquet"
    )
    tables = discover_tables(tmp_path)
    report = build_qc_report(tables)
    assert report["n_tables"] == 1
    assert report["tables"][0]["n_rows"] == 2
    assert report["tables"][0]["missing_first_columns"]["x"] == 1
