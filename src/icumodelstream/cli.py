from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from icumodelstream.cohorts import CohortSpec, build_adult_icu_cohort
from icumodelstream.io import discover_tables, table_inventory
from icumodelstream.qc import build_qc_report, write_qc_report
from icumodelstream.schema import validate_table_contracts, validation_results_to_frame

app = typer.Typer(help="Local-first CLIF-MIMIC parquet pipeline.")
console = Console()


def _resolve_data_root(data_root: Path | None) -> Path:
    if data_root is None:
        raise typer.BadParameter("Pass --data-root or set CLIF_DATA_ROOT before running commands.")
    return data_root.expanduser().resolve()


@app.command()
def inspect(
    data_root: Path = typer.Option(..., help="Directory containing CLIF parquet files."),
) -> None:
    """Inventory parquet tables and validate minimal CLIF contracts."""
    tables = discover_tables(_resolve_data_root(data_root))
    inventory = table_inventory(tables)

    rich_table = Table(title="Discovered CLIF parquet tables")
    for column in inventory.columns:
        rich_table.add_column(column)
    for row in inventory.iter_rows(named=True):
        rich_table.add_row(*(str(row[column]) for column in inventory.columns))
    console.print(rich_table)

    validation = validation_results_to_frame(validate_table_contracts(tables))
    validation_table = Table(title="Minimal core contract validation")
    for column in validation.columns:
        validation_table.add_column(column)
    for row in validation.iter_rows(named=True):
        validation_table.add_row(*(str(row[column]) for column in validation.columns))
    console.print(validation_table)


@app.command()
def qc(
    data_root: Path = typer.Option(..., help="Directory containing CLIF parquet files."),
    out: Path = typer.Option(Path("reports/qc_summary.json"), help="Output JSON path."),
) -> None:
    """Write a JSON QC report for discovered CLIF parquet tables."""
    tables = discover_tables(_resolve_data_root(data_root))
    report = build_qc_report(tables)
    out_path = write_qc_report(report, out)
    console.print(f"Wrote QC report: {out_path}")


@app.command()
def cohort(
    data_root: Path = typer.Option(..., help="Directory containing CLIF parquet files."),
    out: Path = typer.Option(Path("reports/adult_icu_cohort.csv"), help="Output cohort CSV path."),
    min_age: int = typer.Option(18, help="Minimum age for adult cohort."),
    require_icu_location: bool = typer.Option(
        True, help="Require an ICU-like ADT location when possible."
    ),
) -> None:
    """Build the initial adult ICU cohort CSV."""
    tables = discover_tables(_resolve_data_root(data_root))
    df = build_adult_icu_cohort(
        tables, CohortSpec(min_age=min_age, require_icu_location=require_icu_location)
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(out)
    console.print(f"Wrote cohort with {df.height} rows: {out}")


if __name__ == "__main__":
    app()
