"""Conservative placement contrasts and real, geometry-only source selection."""
import numpy as np
import pandas as pd
import pytest

from river_graph.analysis.river_storage_placement import (
    MATCHES,
    PLACEMENTS,
    blocked_summary,
    placement_parameters,
    placement_responses,
    select_confluence,
    selected_corridor,
)
from river_graph.analysis.river_whole_storage import StoragePaths


def tree(permutation=None):
    # Two branches join COMID3, then traverse the common downstream reach4.
    f = pd.DataFrame({"comid": [1, 2, 3, 4], "hydroseq": [4, 3, 2, 1],
        "dnhydroseq": [2, 2, 1, 0], "dnminorhyd": [0]*4,
        "lengthkm": [2., 6., 2., 2.], "areasqkm": [2., 3., 0., 1.],
        "wbareatype": [None]*4, "wbareacomi": [0]*4})
    d = np.array([5., 7., 3., 1.])
    if permutation is not None:
        f, d = f.iloc[permutation], d[permutation]
    return StoragePaths(f, 4, d), f.hydroseq.to_numpy()


def test_actual_confluence_and_midpoint_paths():
    paths, hydro = tree()
    s, reason = select_confluence(paths, hydro)
    assert reason == "included"
    assert s["junction_comid"] == 3
    assert s["source_a_comid"] == 2  # The larger subtree is A.
    assert s["source_b_comid"] == 1
    assert s["branch_a_km"] == 3
    assert s["branch_b_km"] == 1
    assert s["common_km"] == 4
    assert s["weight_a"] == pytest.approx(.6)
    assert s["selected_pair_area_share"] == pytest.approx(5/6)
    corridor = selected_corridor(paths, s)
    lengths = corridor.groupby("segment").length_km.sum()
    assert lengths.to_dict() == {"branch_a": 3., "branch_b": 1., "common": 4.}
    assert not corridor.comid.duplicated().any()


def test_source_selection_is_independent_of_dataframe_order():
    p, h = tree()
    q, j = tree([3, 0, 2, 1])
    a, _ = select_confluence(p, h)
    b, _ = select_confluence(q, j)
    for key in a:
        if not key.endswith("_index"):
            assert a[key] == pytest.approx(b[key])
    pd.testing.assert_frame_equal(selected_corridor(p, a), selected_corridor(q, b))


@pytest.mark.parametrize("n", [1, 3])
def test_no_real_junction_is_excluded(n):
    f = pd.DataFrame({"comid": np.arange(1, n+1), "hydroseq": np.arange(n, 0, -1),
        "dnhydroseq": np.arange(n-1, -1, -1), "dnminorhyd": [0]*n,
        "lengthkm": [1.]*n, "areasqkm": [1.]*n, "wbareatype": [None]*n,
        "wbareacomi": [0]*n})
    p = StoragePaths(f, n, np.arange(n-.5, 0, -1))
    assert select_confluence(p, f.hydroseq.to_numpy()) == (None, "no_two_positive_area_tributaries")


def test_invalid_hydro_alignment_rejected():
    p, h = tree()
    with pytest.raises(ValueError, match="unique hydroseq"):
        select_confluence(p, h[:2])


@pytest.mark.parametrize("match", MATCHES)
def test_equal_strength_fixed_mean_and_causal_budget(match):
    p = [placement_parameters(2., 6., 3., .3, match, k, .7) for k in PLACEMENTS]
    field = "storage_variance" if match == "variance_matched" else "allocated_mean_budget"
    np.testing.assert_allclose([x[field] for x in p], p[0][field], rtol=1e-14)
    for x in p:
        assert (x["translation"] >= 0).all()
        np.testing.assert_allclose(x["translation"]+x["taus"], x["delays"])
        assert x["weights"]@x["delays"] == pytest.approx(1.)


def test_variance_matched_sd_and_source_envelope_identity():
    f, traces = placement_responses(2., 6., 3., .3, keep_curves=True)
    assert len(f) == 19 and not traces.empty
    np.testing.assert_allclose(f.anomaly_area_fraction, 1., atol=1e-9)
    np.testing.assert_allclose(f.pulse_centroid, 1., atol=1e-9)
    np.testing.assert_allclose(f.pulse_sd, f.analytic_sd, atol=1e-8)
    for _, g in f[f.strength_match.eq("variance_matched")].groupby("fraction"):
        np.testing.assert_allclose(g.pulse_sd, g.pulse_sd.iloc[0], atol=1e-9)
    assert (f.individual_peak_envelope <= f.individual_peak_envelope.iloc[0]+1e-7).all()
    assert f.alignment_ratio.between(0, 1+1e-12).all()
    np.testing.assert_allclose(f.log_peak_change, f.log_envelope_change+f.log_alignment_change, atol=1e-14)
    assert (f.loc[f.placement.eq("shared_trunk"), "peak_reduction_pct"] >= -1e-7).all()


def test_early_late_names_survive_branch_swap_and_length_units():
    a, _ = placement_responses(2., 6., 3., .3)
    b, _ = placement_responses(6., 2., 3., .7)
    c, _ = placement_responses(20., 60., 30., .3)
    fields = ["pulse_peak", "pulse_sd", "duration_80", "allocated_mean_budget", "storage_variance"]
    np.testing.assert_allclose(a[fields], b[fields], atol=1e-12)
    np.testing.assert_allclose(a[fields], c[fields], atol=1e-12)


def test_zero_allocation_exactly_preserves_original_paths():
    for match in MATCHES:
        for where in PLACEMENTS:
            p = placement_parameters(2., 6., 3., .3, match, where, 0.)
            np.testing.assert_array_equal(p["taus"], [0., 0.])
            np.testing.assert_array_equal(p["translation"], p["delays"])


@pytest.mark.parametrize("fractions", [(), (-.1,), (1.1,), (.5, .5), ((.5,),)])
def test_invalid_allocation_sets_rejected(fractions):
    with pytest.raises(ValueError):
        placement_responses(2., 6., 3., .3, fractions=fractions)


def test_block_bootstrap_preserves_unit_equal_mean_with_unequal_block_sizes():
    f = pd.DataFrame({"station": ["A", "B", "C"], "huc4": ["01", "01", "02"], "value": [0., 0., 12.]})
    row = blocked_summary(f, ["value"], group="huc4", unit="station").iloc[0]
    assert row["mean"] == 4.  # Equal station weighting; not the block-equal mean6.
    assert row.ci_low == 0. and row.ci_high == 12.
    assert row.n_units == 3 and row.n_blocks == 2
    with pytest.raises(ValueError, match="inferential unit"):
        blocked_summary(pd.concat([f, f]), ["value"], group="huc4", unit="station")


def test_one_monitoring_system_has_no_estimable_cluster_interval():
    f = pd.DataFrame({"station": ["A", "B"], "system": [1, 1], "value": [2., 4.]})
    row = blocked_summary(f, ["value"], group="system", unit="station").iloc[0]
    assert row["mean"] == 3. and np.isnan(row.ci_low) and np.isnan(row.ci_high)


def test_equal_double_peaks_retain_time_ambiguity_instead_of_grid_switching():
    coarse, _ = placement_responses(2., 10., 1., .5, sigma=.075, dt=.00125)
    fine, _ = placement_responses(2., 10., 1., .5, sigma=.075, dt=.000625)
    a, b = coarse.iloc[0], fine.iloc[0]
    assert a.near_equal_peak_count == b.near_equal_peak_count == 2
    assert a.peak_time_is_ambiguous and b.peak_time_is_ambiguous
    assert a.last_near_equal_peak_time-a.peak_time > 1.
    assert a.peak_time == pytest.approx(b.peak_time, abs=1e-8)
    assert a.pulse_peak == pytest.approx(b.pulse_peak, abs=1e-10)
