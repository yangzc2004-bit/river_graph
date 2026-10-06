"""Recompute paired mixing metrics and check actual source-role membership."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from river_graph.analysis.river_mechanisms import permitted_doc
from river_graph.experiments.provenance import sha256_file

ROOT = Path("experiments/phase4_transfer/doc_river_mechanisms_v1")


def main():
    meta = json.loads((ROOT/"sources.json").read_text())
    for name, digest in meta["source_hashes"].items():
        if sha256_file(name) != digest:
            raise ValueError(f"changed source: {name}")
    gauge_meta = json.loads((ROOT/"metadata/sources.json").read_text())
    for r in gauge_meta["files"]:
        assert sha256_file(r["file"]) == r["sha256"]
    out = ROOT/"analysis"
    dataset = torch.load("data/processed/mississippi_graph_graphfix_st357.pt", map_location="cpu", weights_only=False)
    expected = []
    for split in (142, 143, 144):
        with np.load(f"experiments/phase4_transfer/unified_doc_spatial_v1/confirmation/masks/split{split}.npz") as z:
            expected.extend(z["train"])
    cells = np.load(out/"source_cells.npy")
    np.testing.assert_array_equal(cells, np.unique(expected))
    visible = permitted_doc(dataset, cells)
    assert np.asarray(dataset["y_mask"], bool).ravel()[cells].all()
    altered = dict(dataset)
    y = np.asarray(dataset["y"]).copy()
    hidden = np.ones(y.size, bool)
    hidden[cells] = False
    y.ravel()[hidden] = 1e8
    altered["y"] = y
    np.testing.assert_array_equal(visible, permitted_doc(altered, cells))
    months = pd.DatetimeIndex(dataset["months"]).to_period("M")
    lookup = {str(s): i for i, s in enumerate(dataset["site_no"])}
    raw = pd.read_parquet(out/"source_sample_days.parquet")
    assert not raw.duplicated(["site_no", "date"]).any()
    for station, frame in raw.groupby("site_no"):
        allowed = months[np.isfinite(visible[lookup[station]])]
        assert frame.date.dt.to_period("M").isin(allowed).all()
    inventory = pd.read_csv(out/"confluence_inventory.csv", dtype={"source_a": str, "source_b": str, "target": str})
    assert inventory.pair_id.is_unique
    eligible = inventory[inventory.eligible_monthly]
    assert eligible.branch_status.eq("independent").all() and eligible.n_common_doc_months.ge(12).all()
    assert eligible.area_overlap_fraction.le(.01).all()
    assert eligible.source_drainage_coverage.le(1.02).all()
    assert not eligible.mapping_gross_mismatch.any() and eligible.reported_drainage_order_ok.all()
    assert not eligible.target.eq("03201720").any()
    records = pd.read_parquet(out/"mixing_records.parquet")
    cases = pd.read_csv(out/"mixing_cases.csv", dtype={"target": str})
    tested = 0
    for r in cases.itertuples():
        population = r.population.removesuffix("_near_complete")
        f = records[(records.pair_id == r.pair_id) & records.population.eq(population) & records.weighting.eq(r.weighting)]
        assert len(f) == r.n_records and f.date.dt.to_period("M").nunique() == r.n_months
        info = inventory[inventory.pair_id.eq(r.pair_id)].iloc[0]
        if r.weighting == "area":
            qa, qb = info.source_a_area_km2, info.source_b_area_km2
        else:
            qa, qb = f.q_a.astype(float), f.q_b.astype(float)
            assert qa.gt(0).all() and qb.gt(0).all()
        mixture = (qa*f.doc_a+qb*f.doc_b)/(qa+qb)
        np.testing.assert_allclose(f.mixture_doc, mixture, rtol=1e-12)
        baseline = .5*(abs(f.doc_a-f.doc_target)+abs(f.doc_b-f.doc_target))
        np.testing.assert_allclose([r.reference_mae, r.mixture_mae, r.signed_departure_mg_L],
                                   [baseline.mean(), abs(mixture-f.doc_target).mean(), (f.doc_target-mixture).mean()])
        np.testing.assert_allclose(r.unweighted_mix_mae, abs(.5*(f.doc_a+f.doc_b)-f.doc_target).mean())
        assert np.isfinite(f[["doc_a", "doc_b", "doc_target", "mixture_doc"]]).all().all()
        for station, column in ((info.source_a, "doc_a"), (info.source_b, "doc_b"), (info.target, "doc_target")):
            t = months.get_indexer(f.date.dt.to_period("M"))
            assert (t >= 0).all() and np.isfinite(visible[lookup[station], t]).all()
            if population == "monthly":
                np.testing.assert_allclose(f[column], visible[lookup[station], t])
            else:
                truth = raw[raw.site_no.eq(station)].set_index("date").doc.reindex(f.date).to_numpy()
                np.testing.assert_allclose(f[column], truth)
        tested += 1
    receivers = pd.read_csv(out/"mixing_receivers.csv", dtype={"target": str})
    result = pd.read_csv(out/"mixing_results.csv")
    for r in result.itertuples():
        s = receivers[receivers.population.eq(r.population) & receivers.weighting.eq(r.weighting)]
        if r.group != "all":
            s = s[s.cluster.eq(int(r.group[-1]))]
        if r.metric in ("mae_gain_pct", "weighting_gain_pct"):
            reference = s.reference_mae if r.metric == "mae_gain_pct" else s.unweighted_mix_mae
            point = 100*(1-s.mixture_mae.mean()/reference.mean())
        else:
            point = s[r.metric].mean()
        np.testing.assert_allclose(point, r.estimate)
        assert len(s) == r.n_receivers and s.huc4.nunique() == r.n_huc4
        if len(s) == 1 or (r.resampling_unit == "huc4" and r.n_huc4 == 1) or (r.resampling_unit == "component" and r.n_components == 1):
            assert np.isnan(r.ci_low) and np.isnan(r.ci_high)
    edges = pd.read_csv(out/"pathway_edges.csv", dtype={"source": str, "target": str})
    for r in edges.itertuples():
        a, b = visible[lookup[r.source]], visible[lookup[r.target]]
        common = np.isfinite(a+b)
        assert common.sum() == r.n_common_months
        np.testing.assert_allclose(r.log_doc_change, (np.log1p(b[common])-np.log1p(a[common])).mean())
    simulation = pd.read_csv(out/"simulation_timeseries.csv")
    np.testing.assert_allclose(simulation.outlet_flow, 1.)
    assert np.isfinite(simulation.doc_normalized).all()
    summary = {"status": "passed", "source_files_checked": len(meta["source_hashes"]),
               "source_cells": len(cells), "mixing_cases_recomputed": tested,
               "paired_source_records": len(records), "source_sample_days": len(raw),
               "pathway_edges_recomputed": len(edges), "hidden_month_perturbation": "unchanged source DOC",
               "raw_date_populations": "all within permitted source cells; same-day exact, no widening",
               "reference": "individual sources averaged; equal-concentration mixture additionally reported",
               "simulation": "all post-warmup outlet flows exactly equal source flow",
               "external_or_geographical_test_read": False}
    (ROOT/"verification.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
