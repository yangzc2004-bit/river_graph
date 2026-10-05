"""Keep zero-shot scoring separate from fixed-query support calibration."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from analyze_doc_geographical_confirmation_v1 import summaries
from run_doc_geographical_confirmation_v1 import support_curves


def test_support_curves_use_fixed_query_and_source_selected_adapter():
    n, months = 4, 10
    truth = np.full((n, months), 4.)
    dataset = {"y": truth.copy(), "site_no": np.array(["a", "b", "c", "d"]),
               "months": np.arange("2000-01", "2000-11", dtype="datetime64[M]")}
    split = {"train": np.arange(2*months), "val": np.arange(2*months, 3*months),
             "test": np.arange(3*months, 4*months), "context": np.array([], dtype=np.int64)}
    base = np.full((n, months), 2.)
    first, adapters = support_curves({"model": base}, dataset, split, truth)
    query = first[first.k.eq(0)].cell.to_numpy()
    assert len(query) == 5
    for k in (0, 1, 3, 5):
        np.testing.assert_array_equal(query, first[first.k.eq(k)].cell)
    np.testing.assert_array_equal(first[first.k.eq(0)].y_pred, 2.)
    np.testing.assert_allclose(first[first.k.eq(5)].y_pred, 4., atol=1e-12)
    # Alter query truth only. It can change the exported score labels, not
    # source-selected shrinkage or support-adapted predictions.
    changed = truth.copy()
    changed.ravel()[query] = 9000
    second, next_adapters = support_curves({"model": base}, dataset, split, changed)
    assert adapters == next_adapters
    np.testing.assert_array_equal(first.y_pred, second.y_pred)
    assert not np.array_equal(first.y_true, second.y_true)


def test_geographical_metrics_weight_five_regions_after_seed_average():
    rows, thresholds = [], {}
    for region in range(5):
        for seed in (42, 43):
            thresholds[(region, seed)] = .5
            for month in range(2*(region+1)):
                truth = 1.+month
                error = 1.+region+.2*(seed-42)
                rows.append({"split_seed": region, "seed": seed, "model_name": "model", "k": 0,
                    "station": f"region{region}", "y_true": truth, "y_pred": truth+error})
    panel = pd.DataFrame(rows)
    _, parts, summary, _ = summaries(panel, thresholds)
    assert len(parts) == 5
    np.testing.assert_allclose(summary.mae, 3.1)
    np.testing.assert_allclose(summary.q90_mae, 3.1)
    np.testing.assert_allclose(summary.station_equal_mae, 3.1)
    assert not np.isclose(np.abs(panel.y_pred-panel.y_true).mean(), 3.1)
