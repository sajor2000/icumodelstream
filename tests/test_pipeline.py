"""Tests for the end-to-end Phase 4 baseline pipeline (U1 of CLI work).

Pins:
* CLAUDE.md rule 4: every test has a verifiable success criterion.
* CLAUDE.md rule 5: toy parquet fixtures only -- no real CLIF data.
* CLAUDE.md rule 7: missing tables / degenerate inputs raise loudly.
* CLAUDE.md rule 10: same seed -> identical metrics (reproducibility).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from icumodelstream.cohorts import CohortSpec
from icumodelstream.io import discover_tables
from icumodelstream.models import BaselineResult
from icumodelstream.pipeline import (
    BaselinePipelineResult,
    get_admission_anchors,
    run_baseline_pipeline,
)

EXPECTED_METRIC_KEYS = {
    "auroc",
    "auprc",
    "brier_score",
    "prevalence",
    "calibration_intercept",
    "calibration_slope",
}


def _build_toy_clif(
    tmp_path: Path,
    n_patients: int = 20,
    prevalence: float = 0.3,
    seed: int = 0,
    write_vitals: bool = True,
    write_labs: bool = True,
) -> None:
    """Write a minimal CLIF dataset under tmp_path.

    Tables written:
      * patient: ``patient_id`` (String), ``age`` (Int64, all >= 18)
      * hospitalization: ``patient_id``, ``hospitalization_id`` (both String),
        ``admission_dttm`` (Datetime[UTC]), ``discharge_dttm`` (Datetime[UTC]),
        ``discharge_category`` (String, "Expired" with rate ~prevalence else "Home")
      * adt: ``hospitalization_id``, ``location_category`` (all "ICU")
      * vitals (if write_vitals): 12 readings per hospitalization spread across 24h
      * labs (if write_labs): 5 readings per hospitalization spread across 24h

    Mortality label is deterministic given seed so reproducibility tests are tight.
    """
    rng = np.random.default_rng(seed)
    patient_ids = [f"P{i:03d}" for i in range(n_patients)]
    hospitalization_ids = [f"H{i:03d}" for i in range(n_patients)]
    ages = [25 + (i % 50) for i in range(n_patients)]  # all adult, 25..74
    admission_base = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
    admission_dttms = [admission_base + timedelta(days=i) for i in range(n_patients)]
    discharge_dttms = [adm + timedelta(days=5) for adm in admission_dttms]

    # Deterministic mortality flags driven by rng so the same seed gives the same labels.
    is_expired = rng.uniform(size=n_patients) < prevalence
    discharge_categories = ["Expired" if flag else "Home" for flag in is_expired]

    pl.DataFrame(
        {
            "patient_id": patient_ids,
            "age": ages,
        }
    ).write_parquet(tmp_path / "patient.parquet")

    pl.DataFrame(
        {
            "patient_id": patient_ids,
            "hospitalization_id": hospitalization_ids,
            "admission_dttm": admission_dttms,
            "discharge_dttm": discharge_dttms,
            "discharge_category": discharge_categories,
        }
    ).with_columns(
        pl.col("admission_dttm").cast(pl.Datetime(time_zone="UTC")),
        pl.col("discharge_dttm").cast(pl.Datetime(time_zone="UTC")),
    ).write_parquet(tmp_path / "hospitalization.parquet")

    pl.DataFrame(
        {
            "hospitalization_id": hospitalization_ids,
            "location_category": ["ICU"] * n_patients,
        }
    ).write_parquet(tmp_path / "adt.parquet")

    if write_vitals:
        vital_ids: list[str] = []
        vital_dttms: list[datetime] = []
        vital_values: list[float] = []
        for hid, adm, expired in zip(hospitalization_ids, admission_dttms, is_expired):
            # 12 readings spaced 2h apart inside the 24h window.
            base_value = 100.0 if expired else 80.0
            for hours in range(0, 24, 2):
                vital_ids.append(hid)
                vital_dttms.append(adm + timedelta(hours=hours))
                vital_values.append(base_value + rng.normal(scale=5.0))
        pl.DataFrame(
            {
                "hospitalization_id": vital_ids,
                "recorded_dttm": vital_dttms,
                "value": vital_values,
            }
        ).with_columns(pl.col("recorded_dttm").cast(pl.Datetime(time_zone="UTC"))).write_parquet(
            tmp_path / "vitals.parquet"
        )

    if write_labs:
        lab_ids: list[str] = []
        lab_dttms: list[datetime] = []
        lab_values: list[float] = []
        for hid, adm, expired in zip(hospitalization_ids, admission_dttms, is_expired):
            # 5 readings spaced ~4.8h apart inside the 24h window.
            base_value = 2.5 if expired else 1.0
            for step in range(5):
                lab_ids.append(hid)
                lab_dttms.append(adm + timedelta(hours=step * 4.8))
                lab_values.append(base_value + rng.normal(scale=0.3))
        pl.DataFrame(
            {
                "hospitalization_id": lab_ids,
                "lab_result_dttm": lab_dttms,
                "value": lab_values,
            }
        ).with_columns(
            pl.col("lab_result_dttm").cast(pl.Datetime(time_zone="UTC"))
        ).write_parquet(tmp_path / "labs.parquet")


def test_run_baseline_pipeline_happy_path(tmp_path: Path) -> None:
    """End-to-end: 20 patients, all-ICU, vitals + labs both present."""
    _build_toy_clif(tmp_path, n_patients=20, prevalence=0.3, seed=0)
    tables = discover_tables(tmp_path)

    result = run_baseline_pipeline(
        tables,
        CohortSpec(min_age=18, require_icu_location=True),
        window_hours=24.0,
        test_size=0.2,
        seed=42,
    )

    assert isinstance(result, BaselinePipelineResult)
    assert isinstance(result.lightgbm, BaselineResult)
    assert isinstance(result.logistic, BaselineResult)
    assert set(result.lightgbm.metrics.keys()) == EXPECTED_METRIC_KEYS
    assert set(result.logistic.metrics.keys()) == EXPECTED_METRIC_KEYS
    # Group-aware split rounds whole patients; n_train + n_test must equal the cohort.
    assert result.n_train + result.n_test == 20
    # Both feature tables are present, so both prefixes appear in feature_names.
    assert any(c.startswith("vitals_") for c in result.feature_names)
    assert any(c.startswith("labs_") for c in result.feature_names)
    assert "vitals_mean" in result.feature_names
    assert "labs_mean" in result.feature_names
    assert result.n_features == len(result.feature_names)
    assert result.warnings == []
    # No patient leakage: one patient per hospitalization here, so n_*_patients matches n_*.
    assert result.n_train_patients + result.n_test_patients == 20


def test_get_admission_anchors_returns_renamed_columns(tmp_path: Path) -> None:
    """Standalone test for the public anchors helper."""
    expected_dttms = [
        datetime(2024, 1, 1, 8, 0, tzinfo=timezone.utc),
        datetime(2024, 2, 15, 12, 30, tzinfo=timezone.utc),
        datetime(2024, 3, 20, 23, 45, tzinfo=timezone.utc),
    ]
    pl.DataFrame(
        {
            "patient_id": ["P1", "P2", "P3"],
            "hospitalization_id": ["H1", "H2", "H3"],
            "admission_dttm": expected_dttms,
        }
    ).with_columns(pl.col("admission_dttm").cast(pl.Datetime(time_zone="UTC"))).write_parquet(
        tmp_path / "hospitalization.parquet"
    )

    tables = discover_tables(tmp_path)
    anchors = get_admission_anchors(tables).sort("hospitalization_id")

    assert anchors.columns == ["hospitalization_id", "anchor_dttm"]
    assert anchors.height == 3
    assert anchors["hospitalization_id"].to_list() == ["H1", "H2", "H3"]
    assert anchors["anchor_dttm"].to_list() == expected_dttms


def test_run_baseline_pipeline_vitals_missing_runs_on_labs(tmp_path: Path) -> None:
    """vitals.parquet absent -> pipeline still runs from labs alone, with a warning."""
    _build_toy_clif(tmp_path, n_patients=20, prevalence=0.3, seed=1, write_vitals=False)
    tables = discover_tables(tmp_path)
    assert "vitals" not in tables  # sanity check on the fixture

    result = run_baseline_pipeline(
        tables,
        CohortSpec(min_age=18, require_icu_location=True),
        window_hours=24.0,
        test_size=0.2,
        seed=42,
    )

    # No vitals features survived; labs features did.
    assert not any(c.startswith("vitals_") for c in result.feature_names)
    assert any(c.startswith("labs_") for c in result.feature_names)
    assert any("vitals" in w for w in result.warnings)


def test_run_baseline_pipeline_no_features_raises(tmp_path: Path) -> None:
    """vitals AND labs both missing -> ValueError (no features = no model)."""
    _build_toy_clif(
        tmp_path,
        n_patients=20,
        prevalence=0.3,
        seed=2,
        write_vitals=False,
        write_labs=False,
    )
    tables = discover_tables(tmp_path)

    with pytest.raises(ValueError, match="(?i)feature"):
        run_baseline_pipeline(
            tables,
            CohortSpec(min_age=18, require_icu_location=True),
            window_hours=24.0,
            test_size=0.2,
            seed=42,
        )


def test_run_baseline_pipeline_reproducible_with_same_seed(tmp_path: Path) -> None:
    """Same seed + same toy data -> identical metrics for both baselines."""
    _build_toy_clif(tmp_path, n_patients=20, prevalence=0.3, seed=0)
    tables = discover_tables(tmp_path)
    spec = CohortSpec(min_age=18, require_icu_location=True)

    r1 = run_baseline_pipeline(tables, spec, window_hours=24.0, test_size=0.2, seed=42)
    r2 = run_baseline_pipeline(tables, spec, window_hours=24.0, test_size=0.2, seed=42)

    assert r1.lightgbm.metrics["auroc"] == r2.lightgbm.metrics["auroc"]
    assert r1.lightgbm.metrics["brier_score"] == r2.lightgbm.metrics["brier_score"]
    assert r1.logistic.metrics["auroc"] == r2.logistic.metrics["auroc"]
    assert r1.logistic.metrics["brier_score"] == r2.logistic.metrics["brier_score"]
    # Patient-split sizes must also be deterministic.
    assert r1.n_train == r2.n_train
    assert r1.n_test == r2.n_test
    assert r1.n_train_patients == r2.n_train_patients
    assert r1.n_test_patients == r2.n_test_patients


def test_run_baseline_pipeline_config_snapshot_records_inputs(tmp_path: Path) -> None:
    """config_snapshot is the reproducibility record; assert it contains the inputs."""
    _build_toy_clif(tmp_path, n_patients=20, prevalence=0.3, seed=0)
    tables = discover_tables(tmp_path)
    spec = CohortSpec(min_age=21, require_icu_location=True)

    result = run_baseline_pipeline(
        tables,
        spec,
        window_hours=12.0,
        test_size=0.25,
        seed=7,
        include_hospice=True,
    )

    snap = result.config_snapshot
    assert snap["window_hours"] == 12.0
    assert snap["test_size"] == 0.25
    assert snap["seed"] == 7
    assert snap["include_hospice"] is True
    # CohortSpec fields are serialized under "cohort_spec".
    assert snap["cohort_spec"]["min_age"] == 21
    assert snap["cohort_spec"]["require_icu_location"] is True
