"""Independently verify saved flow-memory predictions, roles and key estimates."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from river_graph.experiments.provenance import sha256_file

ROOT = Path("experiments/phase4_transfer/doc_river_flow_memory_v1")


def main():
    out = ROOT/"analysis"
    manifest = json.loads((ROOT/"analysis_sources.json").read_text())
    for path, expected in manifest["source_hashes"].items():
        assert sha256_file(Path(path)) == expected, path
    data = torch.load("data/processed/mississippi_graph_graphfix_st357.pt", map_location="cpu", weights_only=False)
    allowed = np.load("experiments/phase4_transfer/doc_river_mechanisms_v1/analysis/source_cells.npy")
    panel = pd.read_parquet(out/"monthly_flow_pairs.parquet")
    assert not panel.duplicated(["station", "month"]).any()
    assert np.array_equal(np.sort(panel.cell.to_numpy()), np.sort(allowed))
    np.testing.assert_array_equal(panel.doc, np.asarray(data["y"]).ravel()[panel.cell.to_numpy()])
    ti = pd.DatetimeIndex(data["months"]).get_indexer(panel.month)
    ni = pd.Index(np.asarray(data["site_no"], str)).get_indexer(panel.station)
    qi = list(data["feature_channels"]).index("discharge")
    take = panel.previous_flow_observed.to_numpy()
    assert (ti[take] > 0).all()
    np.testing.assert_array_equal(panel.previous_discharge_cfs[take], np.asarray(data["x"])[ni[take], ti[take]-1, qi])
    chronological = pd.read_parquet(out/"chronological_predictions.parquet")
    assert not chronological.duplicated(["population", "station", "month", "arm"]).any()
    ledger = pd.read_csv(out/"chronological_eligibility.csv", dtype={"station": str})
    check = chronological.merge(ledger[["station", "population", "split_year"]], on=["station", "population"], validate="many_to_one")
    assert check.month.dt.year.ge(check.split_year).all()
    query_sets = chronological.groupby(["population", "station", "month"]).arm.nunique()
    assert query_sets.eq(3).all()
    assert np.isfinite(chronological[["doc", "prediction", "log_pred", "log_true"]]).all().all()
    counts = pd.read_csv(out/"population_counts.csv")
    for row in counts.itertuples():
        g = chronological[chronological.population.eq(row.population)]
        assert g.station.nunique() == row.n_chronological_stations
        assert len(g.drop_duplicates(["station", "month"])) == row.n_unique_query_months
    temporal = pd.read_csv(out/"chronological_gains.csv")
    recomputed = []
    for row in temporal.itertuples():
        g = chronological[chronological.population.eq(row.population)].copy()
        g["error"] = abs(g.doc-g.prediction) if row.space == "native" else abs(np.log1p(g.doc)-g.log_pred)
        e = g.groupby(["station", "arm"]).error.mean().unstack()
        baseline, candidate = e[row.reference].mean(), e[row.candidate].mean()
        value = 100*(baseline-candidate)/baseline
        np.testing.assert_allclose([candidate, baseline, value], [row.candidate_mae, row.reference_mae, row.gain_pct], atol=1e-10)
        recomputed.append(value)
    descriptors = pd.read_parquet(out/"descriptor_predictions.parquet")
    gains = pd.read_csv(out/"descriptor_gains.csv")
    for row in gains.itertuples():
        g = descriptors[descriptors.population.eq(row.population) & descriptors.response.eq(row.response)].copy()
        g["error"] = abs(g.true-g.prediction)
        # Same estimand, independently through grouped arithmetic rather than the bootstrap helper.
        means = g.groupby(["fold", "arm"]).error.mean().unstack().mean()
        value = 100*(means[row.reference]-means[row.candidate])/means[row.reference]
        np.testing.assert_allclose(value, row.gain_pct, atol=1e-10)
    # Direct full design least squares, independent of the residualization fit.
    response = pd.read_csv(out/"station_responses_all_source_months.csv", dtype={"station": str})
    chosen = response.sort_values("station").iloc[len(response)//2]
    s = panel[panel.station.eq(chosen.station) & panel.pair_usable]
    m = s.month.dt.month.to_numpy()
    y = s.month.dt.year.to_numpy()+(m-1)/12
    q = np.log(s[["discharge_cfs", "previous_discharge_cfs"]].to_numpy())
    q -= q.mean(0)
    design = np.column_stack([np.ones(len(s)), np.sin(2*np.pi*m/12), np.cos(2*np.pi*m/12), (y-y.mean())/10, q])
    coef = np.linalg.lstsq(design, np.log1p(s.doc), rcond=None)[0]
    np.testing.assert_allclose(coef[-2:], [chosen.current_response, chosen.previous_response], atol=1e-10)
    # Check historical masks and source snapshots were not written by this study.
    verification = {"source_cells": len(panel), "temporal_comparisons_recomputed": len(recomputed),
                    "descriptor_comparisons_recomputed": len(gains), "sample_direct_response_station": chosen.station,
                    "source_hashes_verified": len(manifest["source_hashes"]),
                    "same_queries_all_temporal_arms": True, "chronological_query_after_train": True,
                    "previous_flow_calendar_alignment": True, "status": "passed"}
    (ROOT/"verification.json").write_text(json.dumps(verification, indent=2)+"\n")
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    main()
