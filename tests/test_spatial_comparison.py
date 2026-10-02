import numpy as np
import pandas as pd
import pytest

from river_graph.experiments.spatial_comparison import paired_station_comparison


def test_bootstrap_keeps_cell_weights_and_resamples_denominator():
    # One station contributes three cells, the other only one. A macro mean
    # would incorrectly give delta=-4 instead of the cell-weighted delta=-1.
    ref = pd.DataFrame({"seed": [42]*4, "cell": [0, 1, 2, 3],
                        "station": [0, 0, 0, 1], "y_true": [0.0]*4,
                        "y_pred": [1., 1., 1., 10.]})
    new = ref.assign(y_pred=[3., 3., 3., 0.])
    stats = paired_station_comparison(new, ref)
    assert stats["delta_mae"] == -1
    assert stats["mae"] == 2.25
    assert stats["reference_mae"] == 3.25
    assert stats["reduction_pct"] == pytest.approx(100/3.25)
    # Each endpoint samples just one of the two stations twice.
    assert stats["reduction_lo"] == -200
    assert stats["reduction_hi"] == 100


def test_seed_losses_precede_averaging_and_pairing_is_strict():
    ref = pd.DataFrame({"seed": [42, 43], "cell": [0, 0], "station": [0, 0],
                        "y_true": [5., 5.], "y_pred": [3., 7.]})
    new = ref.assign(y_pred=[4., 6.])
    result = paired_station_comparison(new, ref)
    assert result["mae"] == 1
    assert result["reference_mae"] == 2
    assert np.isfinite(list(result.values())).all()
    with pytest.raises(ValueError, match="sets differ"):
        paired_station_comparison(new.iloc[:1], ref)
