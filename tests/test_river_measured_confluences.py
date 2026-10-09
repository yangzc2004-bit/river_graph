"""Water accounting, tracer limits and geometry sampling units."""

import numpy as np
import pandas as pd
import pytest

from river_graph.analysis.river_measured_confluences import (
    geometry_tables,
    mixing_ledger,
)


def example(qd=5, cd=6.8):
    return pd.DataFrame({"season": ["fall"], "location": ["Con-1"], "transect.location": ["L"],
                         "Q.Ls.main": [2], "Q.Ls.trib": [3], "Q.Ls": [qd],
                         "DOC.mgL.mean.main": [2], "DOC.mgL.mean.trib": [10], "DOC.mgL.mean": [cd],
                         "SpC.main": [100], "SpC.trib": [200], "SpC": [160],
                         "pQm.mtsum": [.4], "pQt.mtsum": [.6]})


def test_conservative_flux_units_and_tracer():
    row = mixing_ledger(example()).iloc[0]
    assert row.incoming_doc_mgs == 34
    assert row.downstream_doc_mgs == 34
    assert row.mix_doc_mgl == pytest.approx(6.8)
    assert row.doc_flux_discrepancy_mgs == 0
    assert row.tracer_main_fraction == pytest.approx(.4)
    assert row.tracer_doc_departure_pct == pytest.approx(0)


def test_added_water_dilutes_without_carbon_loss():
    row = mixing_ledger(example(qd=10, cd=3.4)).iloc[0]
    assert row.doc_departure_pct == pytest.approx(-50)
    assert row.doc_flux_discrepancy_pct == pytest.approx(0)
    assert row.doc_departure_measured_q_pct == pytest.approx(0)
    assert row.water_closure_pct == pytest.approx(100)


def test_do_not_clip_extrapolated_or_undefined_tracer():
    d = example()
    d.loc[0, "SpC"] = 300
    row = mixing_ledger(d).iloc[0]
    assert row.tracer_main_fraction == -1
    assert not row.tracer_in_bounds and np.isnan(row.tracer_mix_doc_mgl)
    d.loc[0, "SpC.main"] = 200
    assert np.isnan(mixing_ledger(d).iloc[0].tracer_main_fraction)


def test_provided_wrong_fraction_is_audited_not_used():
    d = example()
    d.loc[0, "pQm.mtsum"] = .2
    row = mixing_ledger(d).iloc[0]
    assert row.mix_doc_mgl == pytest.approx(6.8)
    assert row.provided_fraction_sum == pytest.approx(.8)
    assert row.provided_main_fraction_error == pytest.approx(-.2)


def test_width_repetitions_are_depth_positions_not_new_transects():
    d = pd.DataFrame({"location": ["1.CHC"] * 4, "reach": ["1.upstream"] * 4,
                      "transect": [1, 1, 1, 2], "wettedwidth.m": [1, 1, 1, 3],
                      "distance.to.confluence": [-5, -5, -5, -10],
                      "depth.m": [.1, .2, np.nan, .3]})
    transects, reaches = geometry_tables(d)
    assert len(transects) == 2
    assert reaches.iloc[0].width_mean_m == 2
    assert reaches.iloc[0].n_depth_points == 3
    assert reaches.iloc[0].n_depth_missing == 1
    assert reaches.iloc[0].depth_transect_mean_m == pytest.approx(.225)


def test_duplicate_campaign_position_rejected():
    d = pd.concat([example(), example()], ignore_index=True)
    with pytest.raises(ValueError, match="Repeated downstream"):
        mixing_ledger(d)
