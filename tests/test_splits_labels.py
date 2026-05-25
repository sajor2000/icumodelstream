from __future__ import annotations

import polars as pl

from icumodelstream.labels import derive_in_hospital_mortality
from icumodelstream.splits import add_stable_split, assign_split


def test_assign_split_is_stable() -> None:
    first = assign_split("patient-123")
    second = assign_split("patient-123")
    assert first == second
    assert first in {"train", "validation", "test"}


def test_add_stable_split_adds_column() -> None:
    frame = pl.DataFrame({"patient_id": ["a", "b", "c"]})
    out = add_stable_split(frame, "patient_id")
    assert "split" in out.columns
    assert set(out["split"].to_list()).issubset({"train", "validation", "test"})


def test_derive_in_hospital_mortality_from_disposition_and_death_time() -> None:
    frame = pl.DataFrame(
        {
            "hospitalization_id": [1, 2, 3],
            "discharge_disposition": ["Home", "Expired", None],
            "death_dttm": [None, None, "2020-01-01"],
        }
    )
    labels = derive_in_hospital_mortality(
        frame,
        disposition_col="discharge_disposition",
        death_time_col="death_dttm",
    ).sort("hospitalization_id")
    assert labels["in_hospital_mortality"].to_list() == [0, 1, 1]
