"""Tests for icumodelstream.sequences.build_sequence_tensors (U1 of Phase 5).

Toy-data only (CLAUDE.md rule 5). Each test builds a fresh tmp_path with the
minimum CLIF parquet files needed to exercise one behavior.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from icumodelstream.io import discover_tables
from icumodelstream.sequences import SequenceTensors, build_sequence_tensors


def _write_vitals(
    path: Path,
    hospitalization_ids: list,
    categories: list[str],
    recorded_dttms: list[datetime],
    values: list[float],
) -> None:
    pl.DataFrame(
        {
            "hospitalization_id": hospitalization_ids,
            "vital_category": categories,
            "recorded_dttm": recorded_dttms,
            "vital_value": values,
        }
    ).with_columns(
        pl.col("recorded_dttm").cast(pl.Datetime(time_zone="UTC"))
    ).write_parquet(path)


def _write_labs(
    path: Path,
    hospitalization_ids: list,
    categories: list[str],
    recorded_dttms: list[datetime],
    values: list[float],
) -> None:
    pl.DataFrame(
        {
            "hospitalization_id": hospitalization_ids,
            "lab_category": categories,
            "lab_collect_dttm": recorded_dttms,
            "lab_value_numeric": values,
        }
    ).with_columns(
        pl.col("lab_collect_dttm").cast(pl.Datetime(time_zone="UTC"))
    ).write_parquet(path)


def _write_resp(
    path: Path,
    hospitalization_ids: list,
    devices: list[str],
    recorded_dttms: list[datetime],
) -> None:
    pl.DataFrame(
        {
            "hospitalization_id": hospitalization_ids,
            "device_category": devices,
            "recorded_dttm": recorded_dttms,
        }
    ).with_columns(
        pl.col("recorded_dttm").cast(pl.Datetime(time_zone="UTC"))
    ).write_parquet(path)


def _anchors(ids: list, anchor: datetime) -> pl.DataFrame:
    return pl.DataFrame(
        {"hospitalization_id": ids, "anchor_dttm": [anchor] * len(ids)}
    ).with_columns(pl.col("anchor_dttm").cast(pl.Datetime(time_zone="UTC")))


def test_tensor_shape_happy_path(tmp_path: Path) -> None:
    """Two hospitalizations with vitals + labs in window produce a (2, 24, n_channels) tensor."""
    anchor = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
    _write_vitals(
        tmp_path / "vitals.parquet",
        hospitalization_ids=[1, 1, 2],
        categories=["heart_rate", "heart_rate", "sbp"],
        recorded_dttms=[
            anchor + timedelta(hours=1),
            anchor + timedelta(hours=5),
            anchor + timedelta(hours=0),
        ],
        values=[80.0, 90.0, 120.0],
    )
    _write_labs(
        tmp_path / "labs.parquet",
        hospitalization_ids=[1],
        categories=["sodium"],
        recorded_dttms=[anchor + timedelta(hours=3)],
        values=[140.0],
    )

    tables = discover_tables(tmp_path)
    cohort = pl.DataFrame({"hospitalization_id": [1, 2]})
    anchors = _anchors([1, 2], anchor)

    out = build_sequence_tensors(tables, cohort, anchors, window_hours=24)

    assert isinstance(out, SequenceTensors)
    # 7 vitals + 12 labs + 2 assessments + 5 resp = 26 channels with defaults.
    assert out.X.shape == (2, 24, 26)
    assert out.mask.shape == (2, 24, 21)  # 21 numeric channels (vitals+labs+assess)
    assert out.X.dtype == np.float32
    assert out.mask.dtype == np.int8

    hr_idx = out.channel_names.index("vitals_heart_rate")
    # Hosp 1 had hr=80 at hour 1 and hr=90 at hour 5. After forward-fill:
    # hour 0 -> 0.0 (leading null)
    # hour 1 -> 80.0
    # hours 2-4 -> 80.0 (forward-filled from hour 1)
    # hour 5 -> 90.0
    # hours 6-23 -> 90.0
    i_hosp1 = list(out.hospitalization_ids).index(1)
    assert out.X[i_hosp1, 0, hr_idx] == pytest.approx(0.0)
    assert out.X[i_hosp1, 1, hr_idx] == pytest.approx(80.0)
    assert out.X[i_hosp1, 4, hr_idx] == pytest.approx(80.0)
    assert out.X[i_hosp1, 5, hr_idx] == pytest.approx(90.0)
    assert out.X[i_hosp1, 23, hr_idx] == pytest.approx(90.0)


def test_mask_is_not_forward_filled(tmp_path: Path) -> None:
    """Mask reflects original observations only; forward-fill is for X, not mask."""
    anchor = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
    _write_vitals(
        tmp_path / "vitals.parquet",
        hospitalization_ids=[1, 1],
        categories=["heart_rate", "heart_rate"],
        recorded_dttms=[
            anchor + timedelta(hours=1),
            anchor + timedelta(hours=5),
        ],
        values=[80.0, 90.0],
    )
    tables = discover_tables(tmp_path)
    cohort = pl.DataFrame({"hospitalization_id": [1]})
    anchors = _anchors([1], anchor)

    out = build_sequence_tensors(tables, cohort, anchors, window_hours=24)
    hr_idx = out.numeric_channel_names.index("vitals_heart_rate")

    # Mask: 1 at hours 1 and 5, 0 elsewhere.
    expected = np.zeros(24, dtype=np.int8)
    expected[1] = 1
    expected[5] = 1
    np.testing.assert_array_equal(out.mask[0, :, hr_idx], expected)


def test_window_boundary_excludes_exact_endpoint(tmp_path: Path) -> None:
    """A row at exactly anchor + 24h is NOT included (half-open interval)."""
    anchor = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
    _write_vitals(
        tmp_path / "vitals.parquet",
        hospitalization_ids=[1, 1],
        categories=["heart_rate", "heart_rate"],
        recorded_dttms=[
            anchor + timedelta(hours=0),   # in
            anchor + timedelta(hours=24),  # out -- exactly on the boundary
        ],
        values=[80.0, 9999.0],
    )
    tables = discover_tables(tmp_path)
    cohort = pl.DataFrame({"hospitalization_id": [1]})
    anchors = _anchors([1], anchor)

    out = build_sequence_tensors(tables, cohort, anchors, window_hours=24)
    hr_idx = out.numeric_channel_names.index("vitals_heart_rate")

    # The 9999 at hour 24 should not appear anywhere in X (would be hour 24,
    # outside [0, 24) range), and the mask shouldn't have a 1 anywhere except hour 0.
    assert out.X[0, :, hr_idx].max() == pytest.approx(80.0)
    expected_mask = np.zeros(24, dtype=np.int8)
    expected_mask[0] = 1
    np.testing.assert_array_equal(out.mask[0, :, hr_idx], expected_mask)


def test_forward_fill_leading_null_to_zero(tmp_path: Path) -> None:
    """Hours before the first observation are filled with 0 (NOT backward-filled)."""
    anchor = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
    _write_vitals(
        tmp_path / "vitals.parquet",
        hospitalization_ids=[1],
        categories=["heart_rate"],
        recorded_dttms=[anchor + timedelta(hours=10)],
        values=[80.0],
    )
    tables = discover_tables(tmp_path)
    cohort = pl.DataFrame({"hospitalization_id": [1]})
    anchors = _anchors([1], anchor)

    out = build_sequence_tensors(tables, cohort, anchors, window_hours=24)
    hr_idx = out.numeric_channel_names.index("vitals_heart_rate")

    # Hours 0..9: 0.0 (leading null fill). Hours 10..23: 80.0 (observation + ffill).
    np.testing.assert_allclose(out.X[0, 0:10, hr_idx], 0.0)
    np.testing.assert_allclose(out.X[0, 10:24, hr_idx], 80.0)

    # Mask is 0 everywhere except hour 10.
    expected_mask = np.zeros(24, dtype=np.int8)
    expected_mask[10] = 1
    np.testing.assert_array_equal(out.mask[0, :, hr_idx], expected_mask)


def test_respiratory_indicator_no_forward_fill(tmp_path: Path) -> None:
    """Respiratory device flags are 1 only at hours with at least one row; no forward-fill."""
    anchor = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
    # vitals must exist for the cohort to have channels; write a dummy row.
    _write_vitals(
        tmp_path / "vitals.parquet",
        hospitalization_ids=[1],
        categories=["heart_rate"],
        recorded_dttms=[anchor],
        values=[70.0],
    )
    _write_resp(
        tmp_path / "respiratory_support.parquet",
        hospitalization_ids=[1, 1],
        devices=["IMV", "IMV"],
        recorded_dttms=[
            anchor + timedelta(hours=2),
            anchor + timedelta(hours=5),
        ],
    )
    tables = discover_tables(tmp_path)
    cohort = pl.DataFrame({"hospitalization_id": [1]})
    anchors = _anchors([1], anchor)

    out = build_sequence_tensors(tables, cohort, anchors, window_hours=24)

    imv_idx = out.channel_names.index("resp_imv")
    # imv channel should NOT have a mask counterpart (it's an indicator).
    assert "resp_imv" not in out.numeric_channel_names

    # Values: 1 at hours 2 and 5, 0 elsewhere -- specifically not forward-filled
    # from hour 2 into hour 3.
    expected = np.zeros(24, dtype=np.float32)
    expected[2] = 1.0
    expected[5] = 1.0
    np.testing.assert_allclose(out.X[0, :, imv_idx], expected)


def test_channel_ordering_deterministic(tmp_path: Path) -> None:
    """Same input -> same channel_names order across two consecutive calls.

    Channel layout: all vitals (in given tuple order), then all labs,
    then all assessments, then all respiratory devices.
    """
    anchor = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
    _write_vitals(
        tmp_path / "vitals.parquet",
        hospitalization_ids=[1],
        categories=["heart_rate"],
        recorded_dttms=[anchor + timedelta(hours=1)],
        values=[80.0],
    )
    tables = discover_tables(tmp_path)
    cohort = pl.DataFrame({"hospitalization_id": [1]})
    anchors = _anchors([1], anchor)

    out1 = build_sequence_tensors(tables, cohort, anchors, window_hours=24)
    out2 = build_sequence_tensors(tables, cohort, anchors, window_hours=24)

    assert out1.channel_names == out2.channel_names

    # Vitals come first.
    assert out1.channel_names[0] == "vitals_heart_rate"
    # Numeric block then indicator block.
    assert out1.numeric_channel_names == out1.channel_names[: len(out1.numeric_channel_names)]
    # Indicator names disjoint from numeric.
    assert set(out1.numeric_channel_names).isdisjoint(
        set(out1.channel_names[len(out1.numeric_channel_names):])
    )
    # Lengths: 7 + 12 + 2 = 21 numeric; +5 resp = 26 total (defaults).
    assert len(out1.numeric_channel_names) == 21
    assert len(out1.channel_names) == 26


def test_empty_cohort_raises(tmp_path: Path) -> None:
    """Empty cohort -> ValueError."""
    anchor = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
    _write_vitals(
        tmp_path / "vitals.parquet",
        hospitalization_ids=[1],
        categories=["heart_rate"],
        recorded_dttms=[anchor + timedelta(hours=1)],
        values=[80.0],
    )
    tables = discover_tables(tmp_path)
    cohort = pl.DataFrame({"hospitalization_id": pl.Series([], dtype=pl.Int64)})
    anchors = _anchors([1], anchor)

    with pytest.raises(ValueError, match="(?i)empty"):
        build_sequence_tensors(tables, cohort, anchors, window_hours=24)


def test_hospitalization_ids_sorted(tmp_path: Path) -> None:
    """Hospitalizations are sorted in the output regardless of cohort row order."""
    anchor = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
    # 3 hospitalizations with string IDs, deliberately unsorted in cohort.
    _write_vitals(
        tmp_path / "vitals.parquet",
        hospitalization_ids=["A", "B", "C"],
        categories=["heart_rate", "heart_rate", "heart_rate"],
        recorded_dttms=[anchor + timedelta(hours=1)] * 3,
        values=[70.0, 80.0, 90.0],
    )
    tables = discover_tables(tmp_path)
    cohort = pl.DataFrame({"hospitalization_id": ["B", "A", "C"]})
    anchors = pl.DataFrame(
        {"hospitalization_id": ["B", "A", "C"], "anchor_dttm": [anchor] * 3}
    ).with_columns(pl.col("anchor_dttm").cast(pl.Datetime(time_zone="UTC")))

    out = build_sequence_tensors(tables, cohort, anchors, window_hours=24)

    assert list(out.hospitalization_ids) == ["A", "B", "C"]
    hr_idx = out.numeric_channel_names.index("vitals_heart_rate")
    # Hour 1: A=70, B=80, C=90 (in sorted order).
    assert out.X[0, 1, hr_idx] == pytest.approx(70.0)
    assert out.X[1, 1, hr_idx] == pytest.approx(80.0)
    assert out.X[2, 1, hr_idx] == pytest.approx(90.0)


def test_window_hours_must_be_positive(tmp_path: Path) -> None:
    """window_hours <= 0 raises."""
    anchor = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
    _write_vitals(
        tmp_path / "vitals.parquet",
        hospitalization_ids=[1],
        categories=["heart_rate"],
        recorded_dttms=[anchor],
        values=[80.0],
    )
    tables = discover_tables(tmp_path)
    cohort = pl.DataFrame({"hospitalization_id": [1]})
    anchors = _anchors([1], anchor)

    with pytest.raises(ValueError, match="window_hours"):
        build_sequence_tensors(tables, cohort, anchors, window_hours=0)


def test_id_dtype_mismatch_raises(tmp_path: Path) -> None:
    """cohort.hospitalization_id Int vs anchors.hospitalization_id Utf8 -> ValueError."""
    anchor = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
    _write_vitals(
        tmp_path / "vitals.parquet",
        hospitalization_ids=[1],
        categories=["heart_rate"],
        recorded_dttms=[anchor],
        values=[80.0],
    )
    tables = discover_tables(tmp_path)
    cohort = pl.DataFrame({"hospitalization_id": [1]})  # Int64
    anchors = pl.DataFrame(
        {"hospitalization_id": ["1"], "anchor_dttm": [anchor]}  # Utf8
    ).with_columns(pl.col("anchor_dttm").cast(pl.Datetime(time_zone="UTC")))

    with pytest.raises(ValueError, match="hospitalization_id"):
        build_sequence_tensors(tables, cohort, anchors, window_hours=24)
