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


@dataclass(frozen=True)
class CohortWaterfall:
    """Row-count waterfall for cohort construction.

    Each count is unique hospitalizations after the corresponding filter step.
    age_col_used and icu_location_col_used capture which tolerant-name candidates
    were resolved against the real data, so a reader can audit what actually ran.
    """

    total_hospitalizations: int
    after_age_filter: int
    after_icu_filter: int
    final: int
    age_col_used: str | None
    icu_location_col_used: str | None
    icu_filter_applied: bool


def first_existing_column(columns: set[str], candidates: tuple[str, ...]) -> str | None:
    """Return the first candidate column that exists in a table."""
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


def build_cohort_with_waterfall(
    tables: dict[str, TableRef], spec: CohortSpec | None = None
) -> tuple[pl.DataFrame, CohortWaterfall]:
    """Build the adult ICU cohort and return a structured row-count waterfall.

    Counts are unique hospitalizations at each step so they stay comparable to the
    final cohort.height (which is also unique). Use this when you need to render a
    filter waterfall; use build_adult_icu_cohort when you only need the cohort.
    """
    cohort_spec = spec or CohortSpec()
    patient = scan_table(tables, "patient")
    hospitalization = scan_table(tables, "hospitalization")

    patient_cols = set(patient.collect_schema().names())
    hosp_cols = set(hospitalization.collect_schema().names())
    age_col = first_existing_column(patient_cols | hosp_cols, AGE_CANDIDATES)

    total = hospitalization.select(pl.col("hospitalization_id").n_unique()).collect().item()

    base = hospitalization.join(patient, on="patient_id", how="left")
    if age_col is not None:
        base = base.filter(pl.col(age_col) >= cohort_spec.min_age)
    after_age = base.select(pl.col("hospitalization_id").n_unique()).collect().item()

    location_col: str | None = None
    icu_applied = False
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
            icu_applied = True

    after_icu = base.select(pl.col("hospitalization_id").n_unique()).collect().item()

    select_cols = [col for col in ("patient_id", "hospitalization_id", age_col) if col is not None]
    cohort = (
        base.select(select_cols)
        .unique(subset=["patient_id", "hospitalization_id"])
        .sort("hospitalization_id")
        .collect()
    )

    waterfall = CohortWaterfall(
        total_hospitalizations=total,
        after_age_filter=after_age,
        after_icu_filter=after_icu,
        final=cohort.height,
        age_col_used=age_col,
        icu_location_col_used=location_col,
        icu_filter_applied=icu_applied,
    )
    return cohort, waterfall


def build_adult_icu_cohort(
    tables: dict[str, TableRef], spec: CohortSpec | None = None
) -> pl.DataFrame:
    """Build a first-pass adult ICU cohort.

    Tolerant of CLIF-MIMIC version differences: requires patient and hospitalization
    identifiers, applies an age filter when an age-like column is present, and applies
    an ICU-location filter when ADT contains a recognizable location column.

    For callers that also need the waterfall counts, prefer build_cohort_with_waterfall.
    """
    cohort, _ = build_cohort_with_waterfall(tables, spec)
    return cohort
