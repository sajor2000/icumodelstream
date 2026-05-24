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
