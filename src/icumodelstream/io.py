from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl


@dataclass(frozen=True)
class TableRef:
    """Reference to a discovered CLIF parquet table."""

    name: str
    path: Path


def table_name_from_path(path: Path) -> str:
    """Infer a CLIF table name from a parquet path."""
    name = path.name
    for suffix in (".parquet", ".parq"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name.lower()


def discover_tables(data_root: str | Path, table_glob: str = "*.parquet") -> dict[str, TableRef]:
    """Discover parquet tables under a CLIF data root."""
    root = Path(data_root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"CLIF data root does not exist: {root}")
    paths = sorted(root.glob(table_glob))
    tables = {
        table_name_from_path(path): TableRef(name=table_name_from_path(path), path=path)
        for path in paths
    }
    if not tables:
        raise FileNotFoundError(f"No parquet tables found in {root} with glob {table_glob!r}")
    return tables


def scan_table(tables: dict[str, TableRef], name: str) -> pl.LazyFrame:
    """Lazily scan a discovered parquet table by name."""
    key = name.lower()
    if key not in tables:
        available = ", ".join(sorted(tables))
        raise KeyError(f"Missing table {name!r}. Available tables: {available}")
    return pl.scan_parquet(tables[key].path)


def read_table(tables: dict[str, TableRef], name: str, limit: int | None = None) -> pl.DataFrame:
    """Read a parquet table, optionally limiting rows for local inspection."""
    lf = scan_table(tables, name)
    if limit is not None:
        lf = lf.limit(limit)
    return lf.collect()


def table_inventory(tables: dict[str, TableRef]) -> pl.DataFrame:
    """Return a small inventory of discovered tables."""
    rows: list[dict[str, str | int]] = []
    for name, ref in sorted(tables.items()):
        schema = pl.scan_parquet(ref.path).collect_schema()
        rows.append({"table": name, "path": str(ref.path), "n_columns": len(schema)})
    return pl.DataFrame(rows)
