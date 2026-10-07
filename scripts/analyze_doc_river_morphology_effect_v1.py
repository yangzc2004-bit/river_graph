"""Compare actual river forms, matched DOC, and identical-input routing."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from river_graph.analysis.river_morphology_effect import (
    BRANCHING,
    ENVIRONMENT_CONTROLS,
    FOOTPRINT,
    MATCH_COVARIATES,
    PATHS,
    geometry_routing,
    group_descriptor_summary,
    morphology_pairs,
    paired_response_summary,
)
from river_graph.experiments.provenance import sha256_file

ROOT = Path("experiments/phase4_transfer/doc_river_morphology_effect_v1")
PANEL = Path("experiments/phase4_transfer/doc_river_source_placement_v1/analysis/station_doc_source_panel.csv")
FLOW = Path("experiments/phase4_transfer/doc_river_hydrologic_activation_v1/analysis/station_response_fits.csv")
ROUTING = Path("data/raw/river_source_placement_v1/routing")
ARMS = {"environment": (), "footprint": FOOTPRINT, "branching": BRANCHING, "paths": PATHS,
        "all_morphology": (*FOOTPRINT, *BRANCHING, *PATHS),
        "without_footprint": (*BRANCHING, *PATHS), "without_branching": (*FOOTPRINT, *PATHS),
        "without_paths": (*FOOTPRINT, *BRANCHING)}
COMPARISONS = [(a, "environment") for a in ("footprint", "branching", "paths", "all_morphology")]
COMPARISONS += [("all_morphology", "without_"+a) for a in ("footprint", "branching", "paths")]
OUTCOMES = {"doc_median": "raw", "log_doc_median": "raw", "harmonic_amplitude": "raw",
            "doc_cv": "raw", "q90_fraction": "raw", "interquartile_response": "raw"}


def block_predictions(panel):
    rows = []
    y = np.log1p(panel.doc_median.to_numpy())
    for fold, (train, test) in enumerate(GroupKFold(5).split(panel, groups=panel.huc4)):
        for model, extra in ARMS.items():
            prep = ColumnTransformer([
                ("numeric", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()),
                 [*ENVIRONMENT_CONTROLS, *extra]),
                ("region", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["huc2"])])
            fit = make_pipeline(prep, Ridge(alpha=10)).fit(panel.iloc[train], y[train])
            for i, p in zip(test, fit.predict(panel.iloc[test]), strict=True):
                rows.append({"station": panel.station.iloc[i], "huc4": panel.huc4.iloc[i], "fold": fold,
                             "model": model, "doc_true": panel.doc_median.iloc[i], "doc_pred": max(0, np.expm1(p)),
                             "log_doc_true": y[i], "log_doc_pred": p})
    return pd.DataFrame(rows)


def block_gains(predictions, draws):
    rows = []
    for space, true, pred in (("native", "doc_true", "doc_pred"), ("log1p", "log_doc_true", "log_doc_pred")):
        f = predictions.copy()
        f["error"] = abs(f[true]-f[pred])
        f = f.pivot(index=["station", "huc4", "fold"], columns="model", values="error").reset_index()
        _, gi = np.unique(f.huc4, return_inverse=True)
        n = gi.max()+1
        rng = np.random.default_rng(42)
        weights = []
        folds = [f.fold.to_numpy() == i for i in sorted(f.fold.unique())]
        while sum(len(w) for w in weights) < draws:
            w = rng.multinomial(n, np.full(n, 1/n), size=draws)[:, gi]
            valid = np.all(np.column_stack([w[:, fold].sum(axis=1) > 0 for fold in folds]), axis=1)
            weights.append(w[valid])
        w = np.concatenate(weights)[:draws]
        for candidate, reference in COMPARISONS:
            a, b = f[candidate].to_numpy(), f[reference].to_numpy()
            ma, mb = np.mean([a[v].mean() for v in folds]), np.mean([b[v].mean() for v in folds])
            ba = np.mean([w[:, v]@a[v]/w[:, v].sum(axis=1) for v in folds], axis=0)
            bb = np.mean([w[:, v]@b[v]/w[:, v].sum(axis=1) for v in folds], axis=0)
            lo, hi = np.quantile(100*(bb-ba)/bb, [.025, .975])
            rows.append({"space": space, "candidate": candidate, "reference": reference,
                         "gain_pct": 100*(mb-ma)/mb, "ci_low_pct": lo, "ci_high_pct": hi,
                         "candidate_mae": ma, "reference_mae": mb,
                         "positive_folds": sum(a[v].mean() < b[v].mean() for v in folds),
                         "n_stations": len(f), "n_huc4": n})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    args = parser.parse_args()
    out = ROOT/"analysis"
    out.mkdir(parents=True, exist_ok=True)
    vaa_path = Path("cache/nldplus_vaa.parquet")
    sources = [PANEL, FLOW, vaa_path, ROOT/"study_plan.md", Path(__file__),
               Path("src/river_graph/analysis/river_morphology_effect.py"),
               Path("src/river_graph/analysis/river_planform_doc.py")]
    panel = pd.read_csv(PANEL, dtype={"station": str, "huc2": str, "huc4": str})
    panel = panel[panel.included & panel.eligible].copy().reset_index(drop=True)
    flow = pd.read_csv(FLOW, dtype={"station": str})
    flow = flow[flow.population.eq("all_source_months") & flow.response_status.eq("included")]
    panel = panel.merge(flow[["station", "interquartile_response"]], on="station", how="left", validate="one_to_one")
    panel["log_doc_median"] = np.log1p(panel.doc_median)
    vaa = pd.read_parquet(vaa_path, columns=["comid", "areasqkm"]).set_index("comid")
    members, desc, profiles, pulses = {}, [], [], []
    receiving_comids = panel.comid.to_numpy(int)
    for row in panel.itertuples():
        cid = int(row.comid)
        p = ROUTING/f"comid_{cid}.npz"
        sources.append(p)
        with np.load(p) as z:
            membership, distance = z["comids"], z["distance_km"]
        members[cid] = set(receiving_comids[np.isin(receiving_comids, membership)].tolist())
        areas = vaa.reindex(membership).areasqkm.to_numpy(float)
        summary, profile, pulse = geometry_routing(areas, distance, row.basin_area_km2)
        desc.append({"station": row.station, **summary})
        for frame, collection in ((profile, profiles), (pulse, pulses)):
            frame["station"], frame["cluster"], frame["huc4"] = row.station, row.cluster, row.huc4
            collection.append(frame)
    panel = panel.merge(pd.DataFrame(desc), on="station", validate="one_to_one")
    panel.to_csv(out/"station_morphology_doc_panel.csv", index=False)
    pd.concat(profiles).to_csv(out/"station_path_profiles.csv", index=False)
    pd.concat(pulses).to_parquet(out/"identical_input_routing.parquet", index=False)
    metrics = ["route_mean_scaled", "route_distance_cv", "route_peak_bin_mass", "mainstem_sinuosity",
               "routing_pulse_peak", "routing_pulse_spread", "routing_pulse_centroid"]
    group_descriptor_summary(panel, metrics, draws=args.bootstrap_draws).to_csv(out/"class_routing_descriptors.csv", index=False)
    match = panel.dropna(subset=list(MATCH_COVARIATES)).reset_index(drop=True)
    ids = match.comid.to_numpy(int)
    nested = np.array([[a in members[b] or b in members[a] for b in ids] for a in ids])
    pairs, balance = [], []
    for a, b in ((1, 3), (1, 2), (2, 3)):
        selected, diagnostics = morphology_pairs(match, a, b, nested=nested)
        pairs.append(selected)
        balance.append(diagnostics)
    pairs = pd.concat(pairs, ignore_index=True)
    pd.concat(balance).to_csv(out/"matching_balance.csv", index=False)
    pairs.to_csv(out/"covariate_selected_pairs.csv", index=False)
    summary, evidence = paired_response_summary(pairs, panel, OUTCOMES, draws=args.bootstrap_draws)
    summary.to_csv(out/"paired_doc_contrasts.csv", index=False)
    evidence.to_csv(out/"paired_doc_evidence.csv", index=False)
    geometry_summary, geometry_evidence = paired_response_summary(
        pairs, panel, dict.fromkeys(metrics, "raw"), draws=args.bootstrap_draws)
    geometry_summary.to_csv(out/"paired_geometry_contrasts.csv", index=False)
    geometry_evidence.to_csv(out/"paired_geometry_evidence.csv", index=False)
    print('Covariate-selected non-nested pairs:', pairs.groupby(["class_a", "class_b"]).size().to_dict(), flush=True)
    print(summary.to_string(index=False), flush=True)
    prediction = block_predictions(panel)
    prediction.to_csv(out/"morphology_block_predictions.csv", index=False)
    gains = block_gains(prediction, args.bootstrap_draws)
    gains.to_csv(out/"morphology_block_gains.csv", index=False)
    print(gains.to_string(index=False), flush=True)
    hashes = {str(p): sha256_file(p) for p in sources}
    (ROOT/"analysis_sources.json").write_text(json.dumps({"source_hashes": hashes,
        "bootstrap_draws": args.bootstrap_draws, "bootstrap_unit": "HUC4; station pairs retained",
        "station_count": len(panel), "huc4_count": panel.huc4.nunique(), "matching_count": len(match),
        "routing": "uniform inputs, constant flow, maximum path delay one; no land-cover weighting",
        "roles": "source-training union 142/143/144; exploratory follow-up", "neural_training": False,
        "external_or_geographic_model_test_read": False}, indent=2)+'\n')


if __name__ == "__main__":
    main()
