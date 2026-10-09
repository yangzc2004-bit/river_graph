"""Study measured DOC-flow memory and its actual river-structure information."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from river_graph.analysis.river_flow_memory import (
    ARMS,
    chronological_prediction,
    descriptor_predictions,
    fit_response,
    flow_month_pairs,
    paired_gains,
    response_moderation,
)
from river_graph.experiments.provenance import sha256_file

ROOT = Path("experiments/phase4_transfer/doc_river_flow_memory_v1")
DATASET = Path("data/processed/mississippi_graph_graphfix_st357.pt")
CELLS = Path("experiments/phase4_transfer/doc_river_mechanisms_v1/analysis/source_cells.npy")
METADATA = Path("experiments/phase4_transfer/doc_river_morphology_effect_v1/analysis/station_morphology_doc_panel.csv")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    args = parser.parse_args()
    out = ROOT/"analysis"
    out.mkdir(parents=True, exist_ok=True)
    data = torch.load(DATASET, weights_only=False, map_location="cpu")
    cells = np.load(CELLS)
    metadata = pd.read_csv(METADATA, dtype={"station": str, "huc4": str, "huc2": str})
    panel = flow_month_pairs(data, cells, metadata)
    panel.to_parquet(out/"monthly_flow_pairs.parquet", index=False)
    fits, residuals, chronological, time_ledger, descriptors, temporal_gains, structure_gains, moderators = [], [], [], [], [], [], [], []
    sizes, diagnostics = [], []
    for population, subset, temperature in (
        ("all_source_months", panel, False),
        ("observed_temperature", panel[panel.temperature_usable], True),
        ("source_months_since_2009", panel[panel.month.dt.year.ge(2009)], False),
    ):
        fitted, resc, pred, ledger = [], [], [], []
        for station, raw in subset.groupby("station", sort=True):
            s = raw[raw.pair_usable].copy()
            record, r = fit_response(s, temperature=temperature)
            fitted.append({"station": station, "population": population, **record})
            if r is not None:
                resc.append(r)
            record, p = chronological_prediction(s, temperature=temperature)
            ledger.append({"station": station, "population": population, **record})
            if not p.empty:
                pred.append(p)
        f = pd.DataFrame(fitted)
        fits.append(f)
        time_ledger.append(pd.DataFrame(ledger))
        good = f[f.status.eq("included")].copy()
        meta = metadata.drop(columns=metadata.columns.intersection(good.columns).difference(["station"]))
        stations = meta.merge(good, on="station", validate="one_to_one")
        stations["log_flow"] = np.log(stations.median_flow)
        stations.to_csv(out/f"station_responses_{population}.csv", index=False)
        residual = pd.concat(resc, ignore_index=True)
        residual["population"] = population
        residuals.append(residual)
        d = descriptor_predictions(stations)
        d["population"] = population
        descriptors.append(d)
        for response, g in d.groupby("response", sort=True):
            errors = g.copy()
            errors["error"] = abs(g.true-g.prediction)
            gain = paired_gains(errors, [(a, "context") for a in ARMS if a != "context"],
                                draws=args.bootstrap_draws, equal_folds=True)
            gain["population"], gain["response"] = population, response
            structure_gains.append(gain)
        coef, diagnostic = response_moderation(residual, stations, draws=args.bootstrap_draws)
        coef["population"] = population
        moderators.append(coef)
        diagnostics.append({"population": population, **diagnostic})
        p = pd.concat(pred, ignore_index=True) if pred else pd.DataFrame()
        if not p.empty:
            p["population"] = population
            chronological.append(p)
            for space in ("native", "log1p"):
                p["error"] = abs(p.prediction-p.doc) if space == "native" else abs(p.log_pred-p.log_true)
                e = p.groupby(["station", "huc4", "arm"], as_index=False).error.mean()
                gain = paired_gains(e, [("current_flow", "season_trend"), ("flow_memory", "current_flow"),
                                       ("flow_memory", "season_trend")], draws=args.bootstrap_draws)
                gain["space"], gain["population"] = space, population
                temporal_gains.append(gain)
        sizes.append({"population": population, "n_response_stations": len(stations),
                      "n_response_months": len(residual), "n_response_huc4": stations.huc4.nunique(),
                      "n_chronological_stations": p.station.nunique() if not p.empty else 0,
                      "n_unique_query_months": len(p.drop_duplicates(["station", "month"])) if not p.empty else 0})
        print(json.dumps(sizes[-1]), flush=True)
    for name, frames in (("response_eligibility.csv", fits), ("chronological_eligibility.csv", time_ledger),
                         ("descriptor_gains.csv", structure_gains), ("chronological_gains.csv", temporal_gains),
                         ("structure_response_modifiers.csv", moderators)):
        pd.concat(frames, ignore_index=True).to_csv(out/name, index=False)
    for name, frames in (("response_residuals.parquet", residuals), ("chronological_predictions.parquet", chronological),
                         ("descriptor_predictions.parquet", descriptors)):
        pd.concat(frames, ignore_index=True).to_parquet(out/name, index=False)
    pd.DataFrame(sizes).to_csv(out/"population_counts.csv", index=False)
    (out/"moderation_identification.json").write_text(json.dumps(diagnostics, indent=2)+"\n")
    inputs = [DATASET, CELLS, METADATA, ROOT/"study_plan.md", Path(__file__),
              Path("src/river_graph/analysis/river_flow_memory.py"),
              Path("src/river_graph/analysis/river_hydrologic_activation.py")]
    manifest = {"source_hashes": {str(p): sha256_file(p) for p in inputs},
                "bootstrap_draws": args.bootstrap_draws, "bootstrap_unit": "whole HUC4, paired; no refitting of held-out predictions",
                "roles": "DOC from source-training union 142/143/144 only", "neural_training": False,
                "previous_flow": "observed preceding calendar month, no preceding DOC or imputation",
                "temporal_evaluation": "all preprocessing and coefficients fitted to earlier years",
                "structure_evaluation": "5 HUC4-held-out folds, fixed Ridge alpha=10; measured response descriptors",
                "interpretation": "monthly hydrologic memory; not physical travel time or event peak width"}
    for p in (out/"descriptor_predictions.parquet", out/"chronological_predictions.parquet"):
        p.with_suffix(".meta.json").write_text(json.dumps(manifest, indent=2)+"\n")
    (ROOT/"analysis_sources.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(pd.concat(temporal_gains).to_string(index=False), flush=True)
    print(pd.concat(structure_gains).query("response == 'previous_response'").to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
