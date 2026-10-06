"""Station units and geographic prediction isolation for shape/DOC analysis."""
import numpy as np
import pandas as pd

from river_graph.analysis.river_planform_doc import (
    CONTROLS,
    SHAPE,
    blocked_gain,
    blocked_predictions,
    shape_features,
)


def test_shape_features_do_not_read_doc_or_area():
    frame = pd.DataFrame({"basin_aspect": [2., 3.], "network_axis_ratio": [3., 4.],
                          "drainage_density": [.5, .8], "mainstem_share": [.2, .1],
                          "mainstem_sinuosity": [1.1, 1.4], "doc": [10., 20.],
                          "basin_area_km2": [100., 200.]})
    a = shape_features(frame)
    frame[["doc", "basin_area_km2"]] *= 1000
    pd.testing.assert_frame_equal(a, shape_features(frame))


def test_blocked_prediction_does_not_fit_held_region_doc():
    rng = np.random.default_rng(42)
    frame = pd.DataFrame(rng.normal(size=(60, len(CONTROLS)+len(SHAPE))), columns=[*CONTROLS, *SHAPE])
    frame["station"] = [f"s{i}" for i in range(60)]
    frame["huc4"] = [f"h{i//6}" for i in range(60)]
    frame["huc2"] = "10"
    frame["eligible"] = True
    frame["cluster"] = np.tile([1, 2, 3], 20)
    frame["doc_median"] = np.exp(rng.normal(size=60))
    first = blocked_predictions(frame)
    ids = first.loc[first.fold.eq(0), "station"].unique()
    frame.loc[frame.station.isin(ids), "doc_median"] *= 100000
    second = blocked_predictions(frame)
    # Holdout y changes scoring only; preprocessing and fitted prediction stay.
    np.testing.assert_array_equal(first.loc[first.fold.eq(0), "y_pred_log1p"],
                                  second.loc[second.fold.eq(0), "y_pred_log1p"])


def test_paired_gain_identity_and_units():
    rows = []
    for i in range(20):
        for model in ("environment_area", "environment_area_classes", "environment_area_shape"):
            rows.append({"station": f"s{i}", "huc4": f"h{i//2}", "fold": (i//2) % 5,
                         "model": model, "y_true": 2., "y_pred": 3.,
                         "y_log1p": np.log(3), "y_pred_log1p": np.log(4)})
    gains = blocked_gain(pd.DataFrame(rows), draws=30)
    assert set(gains.resampling_unit) == {"station", "huc4"}
    assert (gains[["relative_gain_pct", "gain_ci_low_pct", "gain_ci_high_pct"]] == 0).all().all()
    assert gains.loc[gains.space.eq("native"), "reference_mae"].eq(1).all()
