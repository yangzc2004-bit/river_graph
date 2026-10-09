"""Check actual acquisition, independent river units and reported paired scores."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from river_graph.experiments.provenance import sha256_file

ROOT = Path("experiments/phase4_transfer/doc_river_neon_form_validation_v1")


def main():
    out = ROOT / "analysis"
    source = json.loads((ROOT / "analysis_sources.json").read_text())
    for name, digest in source["sources"].items():
        if sha256_file(Path(name)) != digest:
            raise ValueError(f"source changed: {name}")
    ledger = pd.read_csv(out / "eligibility_ledger.csv")
    forms = pd.read_csv(out / "network_forms.csv")
    locations = pd.read_csv(out / "registered_locations.csv")
    panel = pd.read_csv(out / "site_panel_primary.csv")
    if len(ledger) != 27 or ledger.site.duplicated().any():
        raise ValueError("candidate ledger is incomplete or duplicated")
    if panel.site.duplicated().any() or len(panel) != ledger.analysis_eligible.sum():
        raise ValueError("final population does not reconcile with candidate ledger")
    if not panel.geometry_coverage.eq(1).all() or not panel.n_reaches.ge(5).all():
        raise ValueError("eligible river lacks complete whole-network geometry")
    if not panel.registered_point_max_channel_distance_m.le(200).all():
        raise ValueError("routine chemistry location not represented by mapped channel")
    if not panel.basin_to_catchment_area_ratio.between(.8, 1.2).all():
        raise ValueError("basin and unique-catchment areas inconsistent")
    overlaps = pd.read_csv(out / "network_overlaps.csv")
    group = panel.set_index("site").network_group
    for r in overlaps[overlaps.shared_reaches.gt(0)].itertuples():
        if group[r.site_a] != group[r.site_b]:
            raise ValueError("overlapping upstream networks placed in different groups")
    if panel.groupby("network_group").domain.nunique().gt(1).any():
        raise ValueError("domain sensitivity separates overlapping networks; repair its grouping")
    for site in panel.site:
        f = locations[locations.site.eq(site)]
        if not f.status.eq("resolved_catchment").all() or f.comid.nunique() != 1:
            raise ValueError("unresolved historical sampling locations in analysis")
    p = out / "heldout_site_predictions.parquet"
    meta = json.loads(p.with_suffix(".meta.json").read_text())
    if sha256_file(p) != meta["prediction_hash"] or sha256_file(out / "site_panel_primary.csv") != meta["dataset_hash"]:
        raise ValueError("prediction or panel identity changed")
    pred = pd.read_parquet(p)
    if pred.duplicated(["population", "site", "outcome", "model"]).any() or not np.isfinite(pred[["y_true", "y_pred"]]).all().all():
        raise ValueError("invalid or duplicated site predictions")
    gains = pd.read_csv(out / "heldout_form_gains.csv")
    errors = pred.assign(error=abs(pred.y_true-pred.y_pred))
    for row in gains.itertuples():
        f = errors[errors.population.eq(row.population) & errors.outcome.eq(row.outcome)]
        a = f[f.model.eq(row.candidate)].error.mean()
        b = f[f.model.eq(row.reference)].error.mean()
        np.testing.assert_allclose([row.candidate_mae, row.reference_mae, row.gain_pct], [a, b, 100*(b-a)/b], rtol=1e-12)
    summary = {"status": "verified", "candidate_sites": len(ledger), "measured_sites": len(forms),
               "analysis_sites": len(panel), "upstream_overlap_groups": panel.network_group.nunique(),
               "prediction_rows": len(pred), "paired_gain_rows_recomputed": len(gains),
               "full_geometry_and_routine_location_checks": True,
               "raw_sample_individual_coordinates_available": False,
               "original_quality_flags_verified": False,
               "interval_scope": "group-bootstrap of fixed held-out predictions, not a causal coefficient interval"}
    (ROOT / "verification.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
