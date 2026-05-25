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
MORTALITY_VALUES = frozenset({
    "expired",          # CLIF 2.x / MIMIC canonical
    "died",
    "deceased",
    "death",
    "dead/expired",     # Rush-CLIF variant
    "expired in hospital",
})
HOSPICE_VALUES = frozenset({"hospice", "discharged to hospice"})


def extract_mortality_labels(
    tables: dict[str, TableRef], include_hospice: bool = False
) -> pl.DataFrame:
    """Return one row per hospitalization with an in-hospital mortality label.

    Rows with a NULL discharge value (e.g., patients still admitted) are EXCLUDED rather
    than silently coerced to mortality=0; the cohort-join then drops them, and the caller
    can compare cohort.height vs label.height to see how many were dropped.

    Raises ValueError if no rows match any known mortality vocabulary, which indicates
    either a label-vocabulary mismatch or a cohort with zero deaths.

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

    result = (
        lf.filter(pl.col(discharge_col).is_not_null())
        .select(
            pl.col("hospitalization_id"),
            pl.col(discharge_col)
            .cast(pl.Utf8)
            .str.to_lowercase()
            .str.strip_chars()
            .is_in(list(positive_values))
            .cast(pl.Int8)
            .alias("mortality"),
        )
        .unique(subset=["hospitalization_id"])
        .collect()
    )

    if result.height > 0 and int(result["mortality"].sum()) == 0:
        observed = sorted({
            v for v in lf.select(pl.col(discharge_col).cast(pl.Utf8)).collect().to_series().to_list()
            if v is not None
        })
        raise ValueError(
            f"extract_mortality_labels: no rows match any known mortality vocabulary "
            f"in column {discharge_col!r}. Looked for any of {sorted(positive_values)}; "
            f"observed values: {observed[:20]}{' ...' if len(observed) > 20 else ''}. "
            f"Extend MORTALITY_VALUES if the source uses a different vocabulary."
        )

    return result
