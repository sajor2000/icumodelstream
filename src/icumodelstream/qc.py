from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl

from icumodelstream.io import TableRef


def summarize_table(ref: TableRef, max_missing_columns: int = 50) -> dict[str, Any]:
    """Summarize one parquet table with row count, columns, and missingness."""
    lf = pl.scan_parquet(ref.path)
    schema = lf.collect_schema()
    row_count = lf.select(pl.len().alias("n_rows")).collect().item()

    missing_exprs = [
        pl.col(col).null_count().alias(col) for col in schema.names()[:max_missing_columns]
    ]
    missing = lf.select(missing_exprs).collect().to_dicts()[0] if missing_exprs else {}

    return {
        "table": ref.name,
        "path": str(ref.path),
        "n_rows": int(row_count),
        "n_columns": len(schema),
        "columns": schema.names(),
        "missing_first_columns": {k: int(v) for k, v in missing.items()},
    }


def build_qc_report(tables: dict[str, TableRef]) -> dict[str, Any]:
    """Build a JSON-serializable QC report for all discovered CLIF tables."""
    table_summaries = [summarize_table(ref) for _, ref in sorted(tables.items())]
    return {
        "n_tables": len(table_summaries),
        "tables": table_summaries,
    }


def write_qc_report(report: dict[str, Any], out: str | Path) -> Path:
    """Write a QC report to JSON."""
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    return out_path
