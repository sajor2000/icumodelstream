from __future__ import annotations

import polars as pl

from icumodelstream.decision_curve import decision_curve


def test_decision_curve_outputs_reference_strategies() -> None:
    frame = pl.DataFrame({"y": [0, 0, 1, 1], "p": [0.1, 0.2, 0.8, 0.9]})
    out = decision_curve(frame, "y", "p", thresholds=[0.2, 0.5])
    assert out.columns == [
        "threshold",
        "n",
        "prevalence",
        "true_positives",
        "false_positives",
        "net_benefit_model",
        "net_benefit_treat_all",
        "net_benefit_treat_none",
    ]
    assert out.height == 2
    assert out["net_benefit_treat_none"].to_list() == [0.0, 0.0]
