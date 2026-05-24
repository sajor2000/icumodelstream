from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl

from icumodelstream.io import TableRef

# Minimal first-pass CLIF table expectations. These are deliberately conservative.
CORE_TABLES: dict[str, set[str]] = {
    "patient": {"patient_id"},
    "hospitalization": {"hospitalization_id", "patient_id"},
    "adt": {"hospitalization_id"},
}

OPTIONAL_HIGH_VALUE_TABLES: dict[str, set[str]] = {
    "vitals": {"hospitalization_id"},
    "labs": {"hospitalization_id"},
    "medication_admin_continuous": {"hospitalization_id"},
    "medication_admin_intermittent": {"hospitalization_id"},
    "respiratory_support": {"hospitalization_id"},
}


@dataclass(frozen=True)
class TableValidationResult:
    table: str
    present: bool
    required_columns: tuple[str, ...]
    observed_columns: tuple[str, ...]
    missing_columns: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return self.present and not self.missing_columns


def collect_columns(path: Path) -> tuple[str, ...]:
    """Collect parquet column names without loading table rows."""
    schema = pl.scan_parquet(path).collect_schema()
    return tuple(schema.names())


def validate_table_contracts(
    tables: dict[str, TableRef], contracts: dict[str, set[str]] | None = None
) -> list[TableValidationResult]:
    """Validate that expected tables and required columns are present."""
    selected_contracts = contracts or CORE_TABLES
    results: list[TableValidationResult] = []
    for table, required in selected_contracts.items():
        ref = tables.get(table)
        observed = collect_columns(ref.path) if ref else tuple()
        missing = tuple(sorted(required.difference(observed)))
        results.append(
            TableValidationResult(
                table=table,
                present=ref is not None,
                required_columns=tuple(sorted(required)),
                observed_columns=observed,
                missing_columns=missing,
            )
        )
    return results


def validation_results_to_frame(results: list[TableValidationResult]) -> pl.DataFrame:
    """Convert validation results to a dataframe for display or export."""
    return pl.DataFrame(
        [
            {
                "table": result.table,
                "present": result.present,
                "ok": result.ok,
                "required_columns": ",".join(result.required_columns),
                "missing_columns": ",".join(result.missing_columns),
                "n_observed_columns": len(result.observed_columns),
            }
            for result in results
        ]
    )
