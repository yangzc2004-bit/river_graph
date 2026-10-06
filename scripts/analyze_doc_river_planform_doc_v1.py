"""Relate real whole-network shape to source DOC dynamics and matched messages."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from analyze_unified_doc_spatial import joint_station_bootstrap
from build_doc_river_structure_atlas_v1 import load_historical, paired_errors

from river_graph.analysis.river_doc_structure import huc_prefix, source_station_response
from river_graph.analysis.river_planform_doc import (
    adjusted_associations,
    blocked_gain,
    blocked_predictions,
    group_summary,
    shape_features,
)
from river_graph.experiments.provenance import sha256_file

ROOT = Path("experiments/phase4_transfer/doc_river_planform_doc_v1")
GEOMETRY = Path("experiments/phase4_transfer/doc_river_planform_typology_v1/analysis")
ATLAS = Path("experiments/phase4_transfer/doc_river_structure_atlas_v1/analysis")
DATASET = Path("data/processed/mississippi_graph_graphfix_st357.pt")
MASKS = Path("experiments/phase4_transfer/unified_doc_spatial_v1/confirmation/masks")
K1 = Path("experiments/phase4_transfer/kgml_local_transport_v1/k1")


def message_results(dataset, geometry, manifest, draws):
    summaries, stations = [], []
    edges_path = Path("data/processed/graph_edges_graphfix_st357.csv")
    manifest[str(edges_path)] = sha256_file(edges_path)
    edges = pd.read_csv(edges_path, dtype={"source": str, "target": str})
    lookup = {str(s): i for i, s in enumerate(dataset["site_no"])}
    t = len(dataset["months"])
    for family, panel in load_historical(dataset, manifest).items():
        mask_path = K1/"masks"/f"doc__{family}.npz"
        with np.load(mask_path) as mask:
            visible = np.zeros(np.asarray(dataset["y"]).size, dtype=bool)
            visible[mask["train"]] = True
        visible = visible.reshape(np.asarray(dataset["y"]).shape)
        support = np.zeros_like(visible, dtype=int)
        for e in edges.itertuples():
            support[lookup[e.target]] += visible[lookup[e.source]]
        for arm in ("residual_upstream", "residual_both"):
            pair = paired_errors(panel, arm, "residual_nomsg").merge(
                geometry[["station", "cluster", "huc_cd"]], on="station", how="inner", validate="many_to_one")
            pair = pair[pair.cluster.notna()].copy()
            cells = pair.cell.to_numpy()
            pair["visible_upstream_count"] = support[cells//t, cells % t]
            groups = [("all_classified", np.ones(len(pair), dtype=bool))]
            for c in (1, 2, 3):
                groups.append((f"class_{c}", pair.cluster.eq(c).to_numpy()))
                for supported in (False, True):
                    groups.append((f"class_{c}_support_{int(supported)}",
                                   (pair.cluster.eq(c) & pair.visible_upstream_count.gt(0).eq(supported)).to_numpy()))
            for group, keep in groups:
                for tail in (False, True):
                    s = pair[keep & ((pair.y_true_candidate >= pair.q90_train).to_numpy() if tail else True)]
                    if s.empty:
                        continue
                    result = joint_station_bootstrap(s, draws=draws)
                    means = s.groupby(["seed", "split_seed"])[["candidate_error", "reference_error"]].mean()
                    summaries.append({"family": family, "candidate": arm, "reference": "residual_nomsg",
                                      "group": group, "tail": tail, **result,
                                      "positive_seeds": int((means.candidate_error < means.reference_error).sum()),
                                      "n_seeds": s.seed.nunique(),
                                      "few_stations": s.station.nunique() < 10,
                                      "tail_unstable": tail and result["n_station_months_unique"] < 20})
            cell = pair.groupby(["station", "cell", "cluster"], as_index=False).agg(
                candidate_error=("candidate_error", "mean"), reference_error=("reference_error", "mean"),
                visible_upstream_count=("visible_upstream_count", "first"))
            station = cell.groupby(["station", "cluster"], as_index=False).agg(
                candidate_mae=("candidate_error", "mean"), reference_mae=("reference_error", "mean"),
                n_cells=("cell", "size"), support_fraction=("visible_upstream_count", lambda x: x.gt(0).mean()))
            station["family"], station["candidate"] = family, arm
            station["delta_mae"] = station.candidate_mae-station.reference_mae
            stations.append(station)
    return pd.DataFrame(summaries), pd.concat(stations, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    args = parser.parse_args()
    out = ROOT/"analysis"
    out.mkdir(parents=True, exist_ok=True)
    paths = [GEOMETRY/"station_classes.csv", GEOMETRY/"classification_summary.json",
             DATASET, Path("data/processed/streamcat_attributes.csv"),
             Path("data/processed/graph_nodes_graphfix_st357.csv"), ROOT/"study_plan.md",
             Path(__file__), Path("src/river_graph/analysis/river_planform_doc.py"),
             Path("src/river_graph/analysis/river_doc_structure.py"),
             Path("scripts/build_doc_river_structure_atlas_v1.py"),
             Path("scripts/analyze_unified_doc_spatial.py")]
    geom = pd.read_csv(paths[0], dtype={"station": str, "huc_cd": str})
    if geom.station.duplicated().any():
        raise ValueError("duplicate geometry station")
    dataset = torch.load(DATASET, map_location="cpu", weights_only=False)
    cells = []
    for split in (142, 143, 144):
        p = MASKS/f"split{split}.npz"
        paths.append(p)
        with np.load(p) as mask:
            if set(mask["train"]//len(dataset["months"])) & set(mask["test"]//len(dataset["months"])):
                raise ValueError("source/test stations overlap")
            cells.extend(mask["train"].tolist())
    cells = np.unique(cells)
    np.save(out/"source_cells.npy", cells)
    responses, season, q90 = source_station_response(dataset, cells)
    nodes = pd.read_csv(paths[4], dtype={"site_no": str, "huc_cd": str}).rename(columns={"site_no": "station"})
    np.testing.assert_array_equal(nodes.station.to_numpy(), np.asarray(dataset["site_no"], str))
    inclusion = nodes[["station", "comid", "huc_cd"]].merge(
        geom[["station", "cluster", "classification_status"]], on="station", how="left", validate="one_to_one")
    inclusion["classification_status"] = inclusion.classification_status.fillna("geometry_unavailable")
    inclusion = inclusion.merge(responses[["station", "eligible", "n_doc"]], on="station", how="left", validate="one_to_one")
    inclusion["included_response"] = inclusion.cluster.notna() & inclusion.eligible.eq(True)
    inclusion.to_csv(out/"inclusion.csv", index=False)
    cat = pd.read_csv(paths[3])
    cat = cat.assign(wetland=cat.pcthbwet2019ws+cat.pctwdwet2019ws,
                     forest=cat.pctconif2019ws+cat.pctdecid2019ws+cat.pctmxfst2019ws,
                     agriculture=cat.pctcrop2019ws+cat.pcthay2019ws,
                     urban=cat.pcturbhi2019ws+cat.pcturblo2019ws+cat.pcturbmd2019ws+cat.pcturbop2019ws,
                     precipitation=cat.precip9120ws, climate_temperature=cat.tmean9120ws)
    panel = responses.merge(geom[geom.cluster.notna()], on="station", validate="one_to_one")
    panel = panel.merge(nodes[["station", "dec_lat_va", "dec_long_va"]].rename(
        columns={"dec_lat_va": "latitude", "dec_long_va": "longitude"}), on="station", validate="one_to_one")
    panel = panel.merge(cat[["comid", "wetland", "forest", "agriculture", "urban", "precipitation", "climate_temperature"]],
                        on="comid", how="left", validate="many_to_one")
    panel["huc2"] = panel.huc_cd.map(lambda x: huc_prefix(x, 2))
    panel["huc4"] = panel.huc_cd.map(huc_prefix)
    panel["log_flow"], panel["log_n_doc"] = np.log1p(panel.median_flow), np.log1p(panel.n_doc)
    panel["log_polygon_area"] = np.log1p(panel.basin_area_km2)
    panel = pd.concat([panel, shape_features(panel).drop(columns=["mainstem_share", "mainstem_sinuosity"])], axis=1)
    panel.to_csv(out/"station_doc_response.csv", index=False)
    group_summary(panel, args.bootstrap_draws).to_csv(out/"class_doc_response.csv", index=False)
    season = season.merge(panel.loc[panel.eligible, ["station", "cluster"]], on="station", validate="many_to_one")
    season.to_csv(out/"station_seasonal_response.csv", index=False)
    seasonal = season.groupby(["cluster", "month"], as_index=False).agg(
        doc_station_mean=("doc_median", "mean"), doc_station_median=("doc_median", "median"), n_stations=("station", "size"))
    seasonal.to_csv(out/"class_seasonal_response.csv", index=False)
    print(f"DOC panel: {len(panel)} classified source stations; {panel.eligible.sum()} eligible; Q90={q90}", flush=True)
    association, diagnostics = adjusted_associations(panel, args.bootstrap_draws)
    association.to_csv(out/"adjusted_associations.csv", index=False)
    diagnostics.to_csv(out/"regression_diagnostics.csv", index=False)
    print("Adjusted class and continuous associations completed", flush=True)
    predictions = blocked_predictions(panel)
    predictions.to_csv(out/"huc4_blocked_predictions.csv", index=False)
    gains = blocked_gain(predictions, args.bootstrap_draws)
    gains.to_csv(out/"huc4_blocked_gains.csv", index=False)
    print(gains.to_string(index=False), flush=True)
    manifest = {str(p): sha256_file(p) for p in paths}
    summary, station = message_results(dataset, geom, manifest, args.bootstrap_draws)
    summary.to_csv(out/"historical_message_gains.csv", index=False)
    station.to_csv(out/"historical_message_station_errors.csv", index=False)
    edge_path = ATLAS/"balanced_source_upstream_doc_associations.csv"
    manifest[str(edge_path)] = sha256_file(edge_path)
    edge = pd.read_csv(edge_path, dtype={"source": str, "target": str})
    edge = edge.groupby(["source", "target", "lag_months"], as_index=False).agg(
        rho_seasonal_anomaly=("rho_seasonal_anomaly", "mean"), path_length_km=("path_length_km", "first"))
    edge = edge.merge(geom[["station", "cluster"]], left_on="target", right_on="station", validate="many_to_one")
    edge.to_csv(out/"upstream_associations.csv", index=False)
    edge.groupby(["cluster", "lag_months"], as_index=False).agg(
        median_rho=("rho_seasonal_anomaly", "median"), n_edges=("rho_seasonal_anomaly", "size"),
        n_receivers=("target", "nunique")).to_csv(out/"upstream_class_summary.csv", index=False)
    (ROOT/"sources.json").write_text(json.dumps({"source_hashes": manifest,
        "bootstrap_draws": args.bootstrap_draws, "q90_mg_L": q90, "n_source_cells": len(cells),
        "n_classified_source_stations": len(panel), "n_eligible_source_stations": int(panel.eligible.sum()),
        "roles": "source-training union 142/143/144; historical KGML validation only",
        "neural_training": False, "external_or_geographical_test_read": False}, indent=2)+"\n")
    print("DOC planform analysis completed; no neural model retrained", flush=True)


if __name__ == "__main__":
    main()
