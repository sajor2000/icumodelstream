from pathlib import Path

import polars as pl

from icumodelstream.cohorts import CohortSpec, build_adult_icu_cohort, build_cohort_with_waterfall
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


def test_build_cohort_with_waterfall_reports_each_filter_step(tmp_path: Path) -> None:
    """Waterfall counts must be monotonically non-increasing and reflect each filter applied."""
    pl.DataFrame({"patient_id": [1, 2, 3, 4], "age": [10, 25, 50, 80]}).write_parquet(
        tmp_path / "patient.parquet"
    )
    pl.DataFrame(
        {"patient_id": [1, 2, 3, 4], "hospitalization_id": [10, 20, 30, 40]}
    ).write_parquet(tmp_path / "hospitalization.parquet")
    pl.DataFrame(
        {
            "hospitalization_id": [10, 20, 30, 40],
            "location_category": ["Floor", "ICU", "ICU", "Ward"],
        }
    ).write_parquet(tmp_path / "adt.parquet")

    tables = discover_tables(tmp_path)
    cohort, waterfall = build_cohort_with_waterfall(
        tables, CohortSpec(min_age=18, require_icu_location=True)
    )

    assert waterfall.total_hospitalizations == 4
    assert waterfall.after_age_filter == 3  # drops patient 1 (age 10)
    assert waterfall.after_icu_filter == 2  # drops hospitalizations 10 (Floor) and 40 (Ward)
    assert waterfall.final == 2
    assert cohort.height == waterfall.final
    assert waterfall.age_col_used == "age"
    assert waterfall.icu_location_col_used == "location_category"
    assert waterfall.icu_filter_applied is True


def test_build_cohort_with_waterfall_marks_skipped_filters(tmp_path: Path) -> None:
    """When ADT table is absent and require_icu_location=True, icu_filter_applied is False
    and after_icu_filter equals after_age_filter (no rows dropped)."""
    pl.DataFrame({"patient_id": [1, 2], "age": [25, 50]}).write_parquet(
        tmp_path / "patient.parquet"
    )
    pl.DataFrame({"patient_id": [1, 2], "hospitalization_id": [10, 20]}).write_parquet(
        tmp_path / "hospitalization.parquet"
    )
    # No adt.parquet -- ICU filter should be skipped gracefully

    tables = discover_tables(tmp_path)
    _, waterfall = build_cohort_with_waterfall(
        tables, CohortSpec(min_age=18, require_icu_location=True)
    )

    assert waterfall.after_age_filter == 2
    assert waterfall.after_icu_filter == 2
    assert waterfall.icu_filter_applied is False
    assert waterfall.icu_location_col_used is None
