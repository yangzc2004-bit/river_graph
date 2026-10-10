"""Scientific weighting and population checks for the paper-only consolidation."""
import numpy as np
import pandas as pd
import pytest

from scripts.consolidate_doc_spatiotemporal_v4 import (
    METRICS,
    aggregate_metrics,
    assert_aligned,
    compute_metrics,
)


def test_regions_have_equal_weight_despite_different_cell_counts():
    rows = []
    for region, errors in ((1, (1., 3.)), (2, (10., 10.))):
        for seed, error in enumerate(errors):
            rows.append({"split_seed": region, "seed": seed, "model_name": "model",
                         "n_cells": 100 if region == 1 else 1,
                         **dict.fromkeys(METRICS, error)})
    regions, summary = aggregate_metrics(pd.DataFrame(rows))
    assert regions.mae.tolist() == [2., 10.]
    assert summary.mae.iloc[0] == 6.


def test_station_weighting_and_bias_are_distinct_from_cell_mae():
    frame = pd.DataFrame({"cell": [0, 1, 2], "station": ["a", "a", "b"],
                          "y_true": [1., 1., 1.], "y_pred": [2., 2., 5.]})
    result = compute_metrics(frame, 4.)
    assert result["mae"] == 2.
    assert result["station_equal_mae"] == 2.5
    assert result["bias"] == 2.
    assert np.isnan(result["r2"])
    assert np.isnan(result["q90_mae"])


def test_alignment_rejects_a_model_that_loses_a_query_cell():
    frame = pd.DataFrame({"split_seed": [1, 1, 1], "seed": [42, 42, 42],
        "model_name": ["a", "a", "b"], "cell": [0, 1, 0], "station": ["s", "s", "s"],
        "month": ["2020-01", "2020-02", "2020-01"], "y_true": [1., 2., 1.]})
    with pytest.raises(AssertionError):
        assert_aligned(frame)


def test_repeated_seeds_do_not_change_the_query_population():
    frame = pd.DataFrame({"split_seed": [1, 1], "seed": [42, 43], "model_name": ["a", "a"],
                          "cell": [0, 0], "station": ["s", "s"], "month": ["2020-01", "2020-01"],
                          "y_true": [1., 1.]})
    assert_aligned(frame)
    frame.loc[1, "y_true"] = 2.
    with pytest.raises(AssertionError):
        assert_aligned(frame)
