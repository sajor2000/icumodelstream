from __future__ import annotations

import math

import polars as pl

from icumodelstream.metrics import (
    auroc_rank,
    binary_classification_metrics,
    calibration_table,
    expected_calibration_error,
)


def test_binary_metrics_known_values() -> None:
    metrics = binary_classification_metrics([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8])
    assert metrics.n == 4
    assert metrics.prevalence == 0.5
    assert math.isclose(metrics.auroc or -1, 0.75)
    assert math.isclose(metrics.brier, 0.158125)


def test_auroc_returns_none_for_single_class() -> None:
    assert auroc_rank([1, 1, 1], [0.2, 0.7, 0.9]) is None


def test_calibration_table_and_ece() -> None:
    frame = pl.DataFrame({"y": [0, 0, 1, 1], "p": [0.1, 0.2, 0.8, 0.9]})
    cal = calibration_table(frame, "y", "p", n_bins=2)
    assert cal.height == 2
    assert math.isclose(expected_calibration_error(cal), 0.15)
