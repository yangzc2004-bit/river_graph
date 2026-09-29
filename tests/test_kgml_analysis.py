"""Paired inference must estimate the same cell-weighted error as the table."""

import numpy as np
import pandas as pd
import pytest

from scripts.analyze_kgml_local_transport import station_bootstrap as k1_bootstrap
from scripts.analyze_kgml_source_isolation import paired_predictions, station_bootstrap


@pytest.mark.parametrize("bootstrap", [k1_bootstrap, station_bootstrap])
def test_cluster_point_retains_cell_weighting(bootstrap):
    # The station-equal value is 5.0; the cell-weighted endpoint is 2.0.
    delta = pd.Series([10., 0., 0., 0., 0.])
    station = pd.Series(["A", "B", "B", "B", "B"])
    point, low, high = bootstrap(delta, station)
    assert point == 2.0
    assert low <= point <= high


def test_pairing_averages_errors_not_predictions(tmp_path):
    rows = []
    for arm in ("residual_msgdelta", "residual_msgnull"):
        for seed, value in zip((42, 43, 44), (-3., 3., 0.), strict=True):
            path = tmp_path / f"{arm}_{seed}.parquet"
            pred = value if arm == "residual_msgdelta" else 1.
            pd.DataFrame({"cell": [0], "station": ["A"], "month": ["2020-01"],
                          "y_true": [0.], "final_pred": [pred], "graph_delta": [0.],
                          "seed": [seed]}).to_parquet(path)
            rows.append({"arm": arm, "seed": seed, "mask": "test",
                         "prediction_path": str(path)})
    paired = paired_predictions(pd.DataFrame(rows), "test")
    assert paired.residual_msgdelta_pred.iloc[0] == 0.
    assert paired.residual_msgdelta_error.iloc[0] == 2.
    assert paired.residual_msgnull_error.iloc[0] == 1.
    np.testing.assert_array_equal(paired.cell, [0])
