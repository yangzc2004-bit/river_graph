import pandas as pd

from scripts.evaluate_phase4_stage2_gate import _k_curve_checks


def test_k_curve_gate_requires_k5_not_worse_than_k0_and_k1():
    frame = pd.DataFrame(
        [
            {"model_name": "h2x_full", "analyte": "doc", "basin": "a", "k": 0, "mae": 2.0},
            {"model_name": "h2x_full", "analyte": "doc", "basin": "a", "k": 1, "mae": 1.5},
            {"model_name": "h2x_full", "analyte": "doc", "basin": "a", "k": 5, "mae": 1.0},
        ]
    )
    assert _k_curve_checks(frame).iloc[0]["pass"]
    frame.loc[frame["k"].eq(5), "mae"] = 3.0
    assert not _k_curve_checks(frame).iloc[0]["pass"]
