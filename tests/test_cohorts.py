from pathlib import Path

import polars as pl

from icumodelstream.cohorts import CohortSpec, build_adult_icu_cohort
from icumodelstream.io import discover_tables


def test_build_adult_icu_cohort_filters_age_and_icu(tmp_path: Path) -> None:
    pl.DataFrame({"patient_id": [1, 2], "age": [17, 55]}).write_parquet(
        tmp_path / "patient.parquet"
    )
    pl.DataFrame({"patient_id": [1, 2], "hospitalization_id": [10, 20]}).write_parquet(
        tmp_path / "hospitalization.parquet"
    )
    pl.DataFrame(
        {"hospitalization_id": [10, 20], "location_category": ["ICU", "Medical ICU"]}
    ).write_parquet(tmp_path / "adt.parquet")

    tables = discover_tables(tmp_path)
    cohort = build_adult_icu_cohort(tables, CohortSpec(min_age=18, require_icu_location=True))
    assert cohort["hospitalization_id"].to_list() == [20]


def test_build_adult_icu_cohort_unique_on_hospitalization_id(tmp_path: Path) -> None:
    """Duplicate patient rows (different ages for the same patient_id) must NOT inflate the cohort.

    Without subset=, .unique() treats the age column as part of the key and keeps both rows.
    """
    pl.DataFrame(
        {"patient_id": [1, 1, 2], "age": [55, 56, 60]}  # duplicate patient_id with two ages
    ).write_parquet(tmp_path / "patient.parquet")
    pl.DataFrame({"patient_id": [1, 2], "hospitalization_id": [10, 20]}).write_parquet(
        tmp_path / "hospitalization.parquet"
    )

    tables = discover_tables(tmp_path)
    cohort = build_adult_icu_cohort(tables, CohortSpec(min_age=18, require_icu_location=False))

    # Two unique hospitalizations, not four
    assert cohort["hospitalization_id"].to_list() == [10, 20]
    assert cohort.height == 2
