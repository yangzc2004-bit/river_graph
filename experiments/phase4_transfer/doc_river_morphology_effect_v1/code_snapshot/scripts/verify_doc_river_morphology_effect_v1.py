"""Verify morphology pairing, routing identities and matched diagnostic scores."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from analyze_doc_river_morphology_effect_v1 import OUTCOMES, ROOT, block_gains

from river_graph.analysis.river_morphology_effect import (
    MATCH_COVARIATES,
    morphology_pairs,
    paired_response_summary,
)
from river_graph.experiments.provenance import sha256_file


def main():
    source = json.loads((ROOT/"analysis_sources.json").read_text())
    for name, expected in source["source_hashes"].items():
        if sha256_file(Path(name)) != expected:
            raise ValueError(f"source content changed: {name}")
    out = ROOT/"analysis"
    panel = pd.read_csv(out/"station_morphology_doc_panel.csv", dtype={"station": str, "huc4": str, "huc2": str})
    pair = pd.read_csv(out/"covariate_selected_pairs.csv", dtype={"station_a": str, "station_b": str, "huc4": str})
    match = panel.dropna(subset=list(MATCH_COVARIATES)).reset_index(drop=True)
    roots = match.comid.to_numpy(int)
    nested = np.zeros((len(match), len(match)), bool)
    for i, row in enumerate(match.itertuples()):
        with np.load(f"data/raw/river_source_placement_v1/routing/comid_{int(row.comid)}.npz") as z:
            nested[i] = np.isin(roots, z["comids"])
    nested |= nested.T.copy()
    rebuilt, changed = [], []
    modified = match.copy()
    modified[list(OUTCOMES)] = -999
    modified["mainstem_sinuosity"], modified["route_mean_scaled"] = 999, -999
    for a, b in ((1, 3), (1, 2), (2, 3)):
        selected, _ = morphology_pairs(match, a, b, nested=nested)
        perturbed, _ = morphology_pairs(modified, a, b, nested=nested)
        rebuilt.append(selected)
        changed.append(perturbed)
        assert not selected.station_a.duplicated().any() and not selected.station_b.duplicated().any()
    rebuilt, changed = pd.concat(rebuilt, ignore_index=True), pd.concat(changed, ignore_index=True)
    pd.testing.assert_frame_equal(pair, rebuilt, check_dtype=False, atol=1e-12)
    pd.testing.assert_frame_equal(rebuilt, changed)
    summary, _ = paired_response_summary(pair, panel, OUTCOMES, draws=source["bootstrap_draws"])
    saved = pd.read_csv(out/"paired_doc_contrasts.csv")
    pd.testing.assert_frame_equal(summary, saved, check_dtype=False, atol=1e-12)
    assert saved.loc[saved.n_huc4.eq(1), ["ci_low", "ci_high"]].isna().all().all()
    profile = pd.read_csv(out/"station_path_profiles.csv", dtype={"station": str})
    pulse = pd.read_parquet(out/"identical_input_routing.parquet")
    np.testing.assert_allclose(profile.groupby("station").area_mass.sum(), 1, atol=1e-12)
    sums = pulse.groupby("station")[["routed_anomaly", "no_delay_anomaly"]].sum()
    np.testing.assert_allclose(sums.routed_anomaly, sums.no_delay_anomaly, atol=1e-10)
    np.testing.assert_allclose(panel.routing_pulse_mass_ratio, 1, atol=1e-12)
    assert np.isfinite(pulse[["routed_anomaly", "no_delay_anomaly"]]).all().all()
    assert panel.route_area_coverage.min() >= .95
    prediction = pd.read_csv(out/"morphology_block_predictions.csv", dtype={"station": str, "huc4": str})
    assert not prediction.duplicated(["station", "model"]).any()
    assert prediction.groupby("station").model.nunique().eq(8).all()
    assert prediction.groupby("huc4").fold.nunique().eq(1).all()
    gains = block_gains(prediction, source["bootstrap_draws"])
    pd.testing.assert_frame_equal(gains, pd.read_csv(out/"morphology_block_gains.csv"), check_dtype=False, atol=1e-10)
    record = {"status": "passed", "source_hashes_checked": len(source["source_hashes"]),
              "stations": len(panel), "huc4": panel.huc4.nunique(), "all_route_mass_ratios": "one",
              "all_profile_area_weights": "sum one", "matching_DOC_and_shape_perturbation": "unchanged pairs",
              "non_nested_pairs": pair.groupby(["class_a", "class_b"]).size().to_dict(),
              "score_bootstrap_recomputed": True, "single_HUC4_interval": "not estimable, stored NA",
              "neural_training": False}
    record["non_nested_pairs"] = {f"{a}_vs_{b}": int(n) for (a, b), n in record["non_nested_pairs"].items()}
    (ROOT/"verification.json").write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
