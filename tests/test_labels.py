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
