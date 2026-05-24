from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from icumodelstream.io import TableRef, scan_table

AGE_CANDIDATES = ("age_at_admission", "admission_age", "age", "anchor_age")
ICU_TEXT_CANDIDATES = ("location_category", "location_type", "care_unit", "unit_type", "department")


@dataclass(frozen=True)
class CohortSpec:
    """Specification for the initial adult ICU cohort."""

    min_age: int = 18
    require_icu_location: bool = True


def first_existing_column(columns: set[str], candidates: tuple[str, ...]) -> str | None:
    """Return the first candidate column that exists in a table."""
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


def build_adult_icu_cohort(
    tables: dict[str, TableRef], spec: CohortSpec | None = None
) -> pl.DataFrame:
    """Build a first-pass adult ICU cohort.

    The function is intentionally tolerant of CLIF-MIMIC version differences. It requires
    patient and hospitalization identifiers, applies an age filter if an age-like column is
    present, and applies an ICU-location filter if ADT contains a recognizable location column.
    """
    cohort_spec = spec or CohortSpec()
    patient = scan_table(tables, "patient")
    hospitalization = scan_table(tables, "hospitalization")

    patient_cols = set(patient.collect_schema().names())
    hosp_cols = set(hospitalization.collect_schema().names())
    age_col = first_existing_column(patient_cols | hosp_cols, AGE_CANDIDATES)

    base = hospitalization.join(patient, on="patient_id", how="left")
    if age_col is not None:
        base = base.filter(pl.col(age_col) >= cohort_spec.min_age)

    if cohort_spec.require_icu_location and "adt" in tables:
        adt = scan_table(tables, "adt")
        adt_cols = set(adt.collect_schema().names())
        location_col = first_existing_column(adt_cols, ICU_TEXT_CANDIDATES)
        if location_col is not None:
            icu_stays = (
                adt.filter(
                    pl.col(location_col).cast(pl.Utf8).str.to_lowercase().str.contains("icu")
                )
                .select("hospitalization_id")
                .unique()
            )
            base = base.join(icu_stays, on="hospitalization_id", how="inner")

    select_cols = [col for col in ("patient_id", "hospitalization_id", age_col) if col is not None]
    return base.select(select_cols).unique().sort("hospitalization_id").collect()
