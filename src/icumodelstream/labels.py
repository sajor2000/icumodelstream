"""In-hospital mortality label extraction from the CLIF hospitalization table.

CLAUDE.md rule 1 (think before coding): CLIF-MIMIC version differences mean the
column that carries discharge disposition may appear as ``discharge_category``,
``discharge_disposition``, or ``discharge_to``. We resolve the actual name with the
same tolerant-candidate pattern used by ``cohorts.py`` rather than hard-coding one.

CLAUDE.md rule 7 (fail loudly on data assumptions): if none of the candidate
columns are present we raise ValueError naming both the candidates we looked for
and the columns actually observed, so the caller can fix the data or extend the
candidate list rather than receiving an empty/silent label table.
"""

from __future__ import annotations

import polars as pl

from icumodelstream.cohorts import first_existing_column
from icumodelstream.io import TableRef, scan_table

DISCHARGE_CATEGORY_CANDIDATES = (
    "discharge_category",
    "discharge_disposition",
    "discharge_to",
)
MORTALITY_VALUES = frozenset({"expired"})
HOSPICE_VALUES = frozenset({"hospice"})


def extract_mortality_labels(
    tables: dict[str, TableRef], include_hospice: bool = False
) -> pl.DataFrame:
    """Return one row per hospitalization with an in-hospital mortality label.

    Parameters
    ----------
    tables:
        Discovered CLIF parquet tables (must include ``hospitalization``).
    include_hospice:
        If True, hospitalizations discharged to hospice are counted as mortality=1.
        Default False keeps the strict "died in hospital" definition.

    Returns
    -------
    DataFrame with two columns: ``hospitalization_id`` and ``mortality`` (Int8).
    """
    lf = scan_table(tables, "hospitalization")
    columns = set(lf.collect_schema().names())
    discharge_col = first_existing_column(columns, DISCHARGE_CATEGORY_CANDIDATES)
    if discharge_col is None:
        raise ValueError(
            "hospitalization table is missing a discharge disposition column. "
            f"Looked for any of {list(DISCHARGE_CATEGORY_CANDIDATES)}; "
            f"observed columns: {sorted(columns)}."
        )

    positive_values = set(MORTALITY_VALUES)
    if include_hospice:
        positive_values |= HOSPICE_VALUES

    return (
        lf.select(
            pl.col("hospitalization_id"),
            pl.col(discharge_col)
            .cast(pl.Utf8)
            .str.to_lowercase()
            .str.strip_chars()
            .is_in(list(positive_values))
            .cast(pl.Int8)
            .alias("mortality"),
        )
        .with_columns(pl.col("mortality").fill_null(0))
        .collect()
    )
