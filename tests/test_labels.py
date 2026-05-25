from pathlib import Path

import polars as pl
import pytest

from icumodelstream.io import discover_tables
from icumodelstream.labels import extract_mortality_labels


def test_extract_mortality_labels_happy_path(tmp_path: Path) -> None:
    pl.DataFrame(
        {
            "hospitalization_id": ["h1", "h2", "h3"],
            "discharge_category": ["Expired", "Home", "SNF"],
        }
    ).write_parquet(tmp_path / "hospitalization.parquet")

    tables = discover_tables(tmp_path)
    result = extract_mortality_labels(tables).sort("hospitalization_id")

    assert result["hospitalization_id"].to_list() == ["h1", "h2", "h3"]
    assert result["mortality"].to_list() == [1, 0, 0]


def test_extract_mortality_labels_case_insensitive(tmp_path: Path) -> None:
    pl.DataFrame(
        {
            "hospitalization_id": ["h1", "h2", "h3"],
            "discharge_category": ["EXPIRED", "expired", "Home"],
        }
    ).write_parquet(tmp_path / "hospitalization.parquet")

    tables = discover_tables(tmp_path)
    result = extract_mortality_labels(tables).sort("hospitalization_id")

    assert result["mortality"].to_list() == [1, 1, 0]


def test_extract_mortality_labels_alternative_column_name(tmp_path: Path) -> None:
    """When discharge_category is absent but discharge_disposition is present, still works."""
    pl.DataFrame(
        {
            "hospitalization_id": ["h1", "h2"],
            "discharge_disposition": ["Expired", "Home"],
        }
    ).write_parquet(tmp_path / "hospitalization.parquet")

    tables = discover_tables(tmp_path)
    result = extract_mortality_labels(tables).sort("hospitalization_id")

    assert result["mortality"].to_list() == [1, 0]


def test_extract_mortality_labels_raises_on_missing_column(tmp_path: Path) -> None:
    """If no discharge candidate column is present, raise ValueError naming candidates and
    actual columns (CLAUDE.md rule 7: fail loudly on missing required data)."""
    pl.DataFrame(
        {"hospitalization_id": ["h1"], "patient_id": ["p1"]}
    ).write_parquet(tmp_path / "hospitalization.parquet")

    tables = discover_tables(tmp_path)

    with pytest.raises(ValueError) as exc_info:
        extract_mortality_labels(tables)

    message = str(exc_info.value)
    # Must surface what we looked for AND what was observed so the operator can diagnose.
    assert "discharge_category" in message
    assert "discharge_disposition" in message
    assert "patient_id" in message


def test_extract_mortality_labels_hospice_default_excluded(tmp_path: Path) -> None:
    pl.DataFrame(
        {
            "hospitalization_id": ["h1", "h2"],
            "discharge_category": ["Hospice", "Expired"],
        }
    ).write_parquet(tmp_path / "hospitalization.parquet")

    tables = discover_tables(tmp_path)
    result = extract_mortality_labels(tables, include_hospice=False).sort("hospitalization_id")

    assert result["mortality"].to_list() == [0, 1]


def test_extract_mortality_labels_hospice_opt_in(tmp_path: Path) -> None:
    pl.DataFrame(
        {
            "hospitalization_id": ["h1", "h2"],
            "discharge_category": ["Hospice", "Expired"],
        }
    ).write_parquet(tmp_path / "hospitalization.parquet")

    tables = discover_tables(tmp_path)
    result = extract_mortality_labels(tables, include_hospice=True).sort("hospitalization_id")

    assert result["mortality"].to_list() == [1, 1]


def test_extract_mortality_labels_excludes_open_admissions(tmp_path: Path) -> None:
    """Hospitalizations with NULL discharge_category (still in hospital) must NOT be labeled 0.

    Per CLAUDE.md rule 7, silently coercing open admissions to mortality=0 poisons training.
    The right behavior is to exclude those rows from the label set so the cohort join drops
    them or surfaces the loss to the operator.
    """
    pl.DataFrame(
        {
            "hospitalization_id": ["h1", "h2", "h3"],
            "discharge_category": ["Expired", "Home", None],
        }
    ).write_parquet(tmp_path / "hospitalization.parquet")

    tables = discover_tables(tmp_path)
    result = extract_mortality_labels(tables).sort("hospitalization_id")

    # h3 (still admitted) must NOT appear with mortality=0
    assert "h3" not in result["hospitalization_id"].to_list()
    assert result.height == 2


def test_extract_mortality_labels_died_and_deceased_vocabularies(tmp_path: Path) -> None:
    """Non-MIMIC CLIF variants encode death as 'Died', 'Deceased', 'Dead/Expired'.

    The matcher must recognize these in addition to 'Expired' or it silently labels every
    death as mortality=0.
    """
    pl.DataFrame(
        {
            "hospitalization_id": ["h1", "h2", "h3", "h4", "h5"],
            "discharge_category": ["Died", "Deceased", "Dead/Expired", "Expired", "Home"],
        }
    ).write_parquet(tmp_path / "hospitalization.parquet")

    tables = discover_tables(tmp_path)
    result = extract_mortality_labels(tables).sort("hospitalization_id")

    assert result["mortality"].to_list() == [1, 1, 1, 1, 0]


def test_extract_mortality_labels_raises_when_no_values_match(tmp_path: Path) -> None:
    """If the discharge column exists but NO rows match any mortality vocabulary, the
    extractor should fail loudly rather than silently return all zeros (which the
    pipeline only catches downstream as a confusing 'no deaths in cohort' error).
    """
    pl.DataFrame(
        {
            "hospitalization_id": ["h1", "h2"],
            "discharge_category": ["Home", "SNF"],  # neither is a known death vocabulary
        }
    ).write_parquet(tmp_path / "hospitalization.parquet")

    tables = discover_tables(tmp_path)

    with pytest.raises(ValueError, match="no rows match"):
        extract_mortality_labels(tables)


def test_extract_mortality_labels_deduplicates_hospitalization_id(tmp_path: Path) -> None:
    """Duplicate hospitalization_id rows in the source must collapse to a single label
    row so downstream joins don't fan out the cohort.
    """
    pl.DataFrame(
        {
            "hospitalization_id": ["h1", "h1", "h2"],  # h1 duplicated
            "discharge_category": ["Expired", "Expired", "Home"],
        }
    ).write_parquet(tmp_path / "hospitalization.parquet")

    tables = discover_tables(tmp_path)
    result = extract_mortality_labels(tables).sort("hospitalization_id")

    assert result["hospitalization_id"].to_list() == ["h1", "h2"]
    assert result.height == 2


def test_extract_mortality_labels_schema(tmp_path: Path) -> None:
    """Output schema must be exactly hospitalization_id + integer mortality column."""
    pl.DataFrame(
        {
            "hospitalization_id": ["h1", "h2"],
            "discharge_category": ["Expired", "Home"],
            "patient_id": ["p1", "p2"],  # extra columns must not leak through
        }
    ).write_parquet(tmp_path / "hospitalization.parquet")

    tables = discover_tables(tmp_path)
    result = extract_mortality_labels(tables)

    assert result.columns == ["hospitalization_id", "mortality"]
    assert result.schema["mortality"] in (pl.Int8, pl.Int64)
