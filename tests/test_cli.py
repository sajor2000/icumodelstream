"""Smoke tests for the `icumodelstream baseline` CLI command (U2 of CLI work).

Pins:
* CLAUDE.md rule 4: each test has a verifiable success criterion.
* CLAUDE.md rule 5: toy parquet fixture, no real CLIF data.
* CLAUDE.md rule 10: same seed -> identical metrics across invocations.

The toy CLIF builder is intentionally duplicated from tests/test_pipeline.py
rather than factored into conftest.py -- surgical changes only (CLAUDE.md
rule 3), and the helper is small enough to read inline.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
from typer.testing import CliRunner

from icumodelstream.cli import app


def _build_toy_clif(
    tmp_path: Path,
    n_patients: int = 20,
    prevalence: float = 0.3,
    seed: int = 0,
) -> None:
    """Mirror of tests/test_pipeline.py:_build_toy_clif (vitals + labs both present)."""
    rng = np.random.default_rng(seed)
    patient_ids = [f"P{i:03d}" for i in range(n_patients)]
    hospitalization_ids = [f"H{i:03d}" for i in range(n_patients)]
    ages = [25 + (i % 50) for i in range(n_patients)]
    admission_base = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
    admission_dttms = [admission_base + timedelta(days=i) for i in range(n_patients)]
    discharge_dttms = [adm + timedelta(days=5) for adm in admission_dttms]

    is_expired = rng.uniform(size=n_patients) < prevalence
    discharge_categories = ["Expired" if flag else "Home" for flag in is_expired]

    pl.DataFrame({"patient_id": patient_ids, "age": ages}).write_parquet(
        tmp_path / "patient.parquet"
    )

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

    vital_ids: list[str] = []
    vital_dttms: list[datetime] = []
    vital_values: list[float] = []
    for hid, adm, expired in zip(hospitalization_ids, admission_dttms, is_expired):
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

    lab_ids: list[str] = []
    lab_dttms: list[datetime] = []
    lab_values: list[float] = []
    for hid, adm, expired in zip(hospitalization_ids, admission_dttms, is_expired):
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


def test_baseline_command_happy_path(tmp_path: Path) -> None:
    """End-to-end CLI invocation produces metrics JSON, summary MD, and a loadable model."""
    data_root = tmp_path / "data"
    data_root.mkdir()
    _build_toy_clif(data_root, n_patients=20, prevalence=0.3, seed=0)

    metrics_path = tmp_path / "out" / "metrics.json"
    summary_path = tmp_path / "out" / "summary.md"
    model_path = tmp_path / "out" / "model.txt"

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "baseline",
            "--data-root",
            str(data_root),
            "--metrics-out",
            str(metrics_path),
            "--summary-out",
            str(summary_path),
            "--model-out",
            str(model_path),
        ],
    )

    assert result.exit_code == 0, f"CLI failed:\nstdout={result.stdout}\nexc={result.exception}"
    assert metrics_path.exists()
    assert summary_path.exists()
    assert model_path.exists()

    payload = json.loads(metrics_path.read_text())
    for key in (
        "config",
        "cohort_waterfall",
        "n_features",
        "feature_names",
        "split",
        "models",
        "warnings",
        "generated_at",
        "code_version",
    ):
        assert key in payload, f"missing top-level key: {key}"
    assert "lightgbm" in payload["models"]
    assert "logistic" in payload["models"]
    assert "metrics" in payload["models"]["lightgbm"]
    assert "calibration_table" in payload["models"]["lightgbm"]
    # Config snapshot preserves CLI inputs.
    assert payload["config"]["seed"] == 42
    assert payload["config"]["window_hours"] == 24.0
    assert payload["config"]["cohort_spec"]["min_age"] == 18

    summary = summary_path.read_text()
    assert "## Cohort waterfall" in summary
    assert "## Model metrics" in summary

    # The LightGBM model artifact must be a valid Booster on disk.
    booster = lgb.Booster(model_file=str(model_path))
    assert booster.num_trees() > 0


def test_baseline_command_missing_data_root_fails_loudly(tmp_path: Path) -> None:
    """Nonexistent --data-root surfaces a non-zero exit + a message naming the path."""
    missing = tmp_path / "does_not_exist"
    runner = CliRunner()
    result = runner.invoke(app, ["baseline", "--data-root", str(missing)])

    assert result.exit_code != 0
    combined_output = (result.stdout or "") + (result.stderr or "")
    assert "does_not_exist" in combined_output or "does not exist" in combined_output.lower()


def test_baseline_command_reproducible_across_invocations(tmp_path: Path) -> None:
    """Two runs with the same seed against the same toy data -> identical AUROC."""
    data_root = tmp_path / "data"
    data_root.mkdir()
    _build_toy_clif(data_root, n_patients=20, prevalence=0.3, seed=0)

    runner = CliRunner()

    def _run(suffix: str) -> dict:
        metrics_path = tmp_path / f"metrics_{suffix}.json"
        summary_path = tmp_path / f"summary_{suffix}.md"
        model_path = tmp_path / f"model_{suffix}.txt"
        invocation = runner.invoke(
            app,
            [
                "baseline",
                "--data-root",
                str(data_root),
                "--metrics-out",
                str(metrics_path),
                "--summary-out",
                str(summary_path),
                "--model-out",
                str(model_path),
                "--seed",
                "42",
            ],
        )
        assert invocation.exit_code == 0, invocation.stdout
        return json.loads(metrics_path.read_text())

    p1 = _run("a")
    p2 = _run("b")

    assert p1["models"]["lightgbm"]["metrics"]["auroc"] == p2["models"]["lightgbm"]["metrics"]["auroc"]
    assert p1["models"]["logistic"]["metrics"]["auroc"] == p2["models"]["logistic"]["metrics"]["auroc"]
    assert p1["split"]["n_train"] == p2["split"]["n_train"]
    assert p1["split"]["n_test"] == p2["split"]["n_test"]
