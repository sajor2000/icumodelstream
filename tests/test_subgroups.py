from __future__ import annotations

import polars as pl

from icumodelstream.subgroups import subgroup_performance


def test_subgroup_performance_keeps_unknown_and_single_class_warning() -> None:
    frame = pl.DataFrame(
        {
            "y": [0, 1, 1, 1],
            "p": [0.1, 0.8, 0.7, 0.9],
            "sex": ["F", "F", "M", None],
        }
    )
    results = subgroup_performance(frame, "y", "p", ["sex"])
    values = {row["subgroup_value"] for row in results}
    assert values == {"F", "M", "Unknown"}
    warned = [row for row in results if row["subgroup_value"] in {"M", "Unknown"}]
    assert all("warning" in row for row in warned)
