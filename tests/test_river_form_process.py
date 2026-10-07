import numpy as np
import pandas as pd
import pytest

from river_graph.analysis.river_form_process import (
    aggregate_receivers,
    anomaly_diagnostics,
    cluster_mean,
    conditional_association,
    log_sd_ratio,
)
from river_graph.analysis.river_mechanisms import permitted_doc


def signal_frame():
    t = np.arange(60)
    signal = .1*np.sin(2*np.pi*t/7)
    return pd.DataFrame({"date": pd.date_range("2000-01-01", periods=60, freq="MS"),
                         "doc_a": np.expm1(.6+signal), "doc_target": np.expm1(.6+.5*signal)})


def test_sd_ratio_measures_variability_and_not_mean_change():
    frame = signal_frame()
    result = anomaly_diagnostics(frame)
    np.testing.assert_allclose(result["log_anomaly_sd_ratio"], np.log(.5), atol=1e-12)
    shifted = frame.copy()
    shifted.doc_target = np.expm1(np.log1p(frame.doc_target)+1)
    changed = anomaly_diagnostics(shifted)
    np.testing.assert_allclose(changed["log_anomaly_sd_ratio"], result["log_anomaly_sd_ratio"])
    np.testing.assert_allclose(changed["mean_log_departure"]-result["mean_log_departure"], 1)


def test_constant_or_invalid_signal_is_missing_not_false_buffering():
    assert np.isnan(log_sd_ratio(0, 1)) and np.isnan(log_sd_ratio(1, 0))
    f = signal_frame().assign(doc_a=2.)
    assert np.isnan(anomaly_diagnostics(f)["log_anomaly_sd_ratio"])
    f.loc[1, "doc_a"] = -1
    with pytest.raises(ValueError, match="nonnegative"):
        anomaly_diagnostics(f)
    with pytest.raises(ValueError, match="unique"):
        anomaly_diagnostics(pd.concat([signal_frame(), signal_frame()]))


def test_mixing_identical_branches_preserves_synchrony_and_transmission():
    f = signal_frame()
    f["doc_b"] = f.doc_a
    f["doc_target"] = f.doc_a
    f["mixture_doc"] = f.doc_a
    result = anomaly_diagnostics(f, mixing=True)
    for term in ("log_anomaly_sd_ratio", "log_mixture_sd_ratio", "log_downstream_mixture_sd_ratio"):
        np.testing.assert_allclose(result[term], 0, atol=1e-12)
    np.testing.assert_allclose(result["source_anomaly_pearson"], 1)
    np.testing.assert_allclose(result["signal_rho"], 1)


def test_receiver_averaging_does_not_count_incoming_paths_as_replicates():
    f = pd.DataFrame({"target": ["a", "a", "b"], "huc4": ["01"]*3,
                      "component": [1]*3, "metric": [2., 4., 7.]})
    out = aggregate_receivers(f, ("metric",))
    assert len(out) == 2
    assert out.set_index("target").loc["a", "metric"] == 3
    result = cluster_mean(out.assign(cluster=1), "metric", "component", draws=100)
    assert result["estimate"] == 5
    assert np.isnan(result["ci_low"])
    with pytest.raises(ValueError, match="consistent"):
        aggregate_receivers(f.assign(component=[1, 2, 1]), ("metric",))


def test_class_contrast_does_not_get_ci_from_other_class_replication():
    f = pd.DataFrame({"target": list("abcd"), "huc4": ["01", "02", "03", "04"],
                      "component": [1, 2, 3, 4], "cluster": [1, 3, 3, 3], "metric": [0., 1., 2., 3.]})
    r = cluster_mean(f, "metric", "component", draws=100, contrast=True)
    assert r["estimate"] == 2 and np.isnan(r["ci_low"])
    assert r["valid_bootstrap_draws"] == 0


def test_cluster_regression_keeps_scale_and_recovers_known_effect():
    rng = np.random.default_rng(7)
    x, nuisance = rng.normal(size=(2, 60))
    f = pd.DataFrame({"x": x, "control": nuisance, "y": 2*x+5*nuisance,
                      "component": np.repeat(np.arange(12), 5), "huc4": np.repeat(np.arange(6), 10)})
    r, loco = conditional_association(f, "y", "x", ("control",), draws=100)
    np.testing.assert_allclose(r["estimate_per_receiver_sd"], 2*x.std(), atol=1e-10)
    np.testing.assert_allclose([r["ci_low"], r["ci_high"]], 2*x.std(), atol=1e-10)
    np.testing.assert_allclose(loco.estimate, 2*x.std(), atol=1e-10)
    assert r["n_receivers"] == 60 and r["n_blocks"] == 12
    assert r["valid_bootstrap_draws"] == 100


def test_permitted_source_cells_ignore_all_other_doc_truth():
    y = np.arange(24, dtype=float).reshape(3, 8)
    cells = np.array([0, 3, 9, 14])
    visible = permitted_doc({"y": y}, cells)
    modified = y.copy().ravel()
    modified[np.setdiff1d(np.arange(24), cells)] += 10000
    np.testing.assert_array_equal(permitted_doc({"y": modified.reshape(3, 8)}, cells), visible)
