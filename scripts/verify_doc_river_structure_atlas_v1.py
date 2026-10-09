"""Independent joins, arithmetic and source-role checks for the river atlas."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from river_graph.experiments.provenance import sha256_file

ROOT = Path("experiments/phase4_transfer/doc_river_structure_atlas_v1")


def main():
    manifest = json.loads((ROOT/"sources.json").read_text())
    for name, expected in manifest["source_hashes"].items():
        if sha256_file(Path(name)) != expected:
            raise ValueError(f"atlas source changed: {name}")
    a = ROOT/"analysis"
    structures = pd.read_csv(a/"station_structure.csv", dtype={"station": str})
    nodes = pd.read_csv("data/processed/graph_nodes_graphfix_st357.csv", dtype={"site_no": str})
    vaa = pd.read_parquet("cache/nldplus_vaa.parquet", columns=["comid", "fromnode", "tonode", "startflag", "streamorde"])
    selected = nodes.merge(vaa, on="comid", validate="many_to_one")
    np.testing.assert_array_equal(structures.station, selected.site_no)
    np.testing.assert_array_equal(structures.physical_headwater, selected.startflag.eq(1))
    np.testing.assert_array_equal(structures.stream_order, selected.streamorde)
    incoming = vaa.loc[vaa.tonode > 0].groupby("tonode").size()
    np.testing.assert_array_equal(structures.inlet_reaches, selected.fromnode.map(incoming).fillna(0))
    if len(structures) != 357 or structures.station.duplicated().any() or structures.upstream_search_capped.any():
        raise ValueError("invalid structure population")
    paths = pd.read_csv(a/"station_edge_paths.csv", dtype={"source": str, "target": str})
    original_edges = pd.read_csv("data/processed/graph_edges_graphfix_st357.csv", dtype=str)
    pd.testing.assert_frame_equal(paths[["source", "target"]], original_edges)
    if not paths.mainstem_connected.all() or (paths.path_length_km < 0).any():
        raise ValueError("monitored edge is not reconstructed along physical mainstem")
    doc = pd.read_csv(a/"source_doc_station_summaries.csv", dtype={"station": str})
    dataset = torch.load("data/processed/mississippi_graph_graphfix_st357.pt", weights_only=False, map_location="cpu")
    names, truth, n_month = np.asarray(dataset["site_no"], str), np.asarray(dataset["y"]), len(dataset["months"])
    model_effects = pd.read_csv(a/"current_model_structure_effects.csv")
    direct = {name: [] for name in ("available_real_integrated", "station_hidden_trees")}
    val_union = set()
    for split in (142, 143, 144):
        with np.load(f"experiments/phase4_transfer/unified_doc_spatial_v1/confirmation/masks/split{split}.npz") as mask:
            train, val = mask["train"], mask["val"]
        per_station = np.bincount(train//n_month, minlength=len(names))
        expected = pd.DataFrame({"station": names[per_station > 0], "n_doc": per_station[per_station > 0]})
        actual = doc.loc[doc.split_seed.eq(split), ["station", "n_doc"]]
        pd.testing.assert_frame_equal(expected.sort_values("station").reset_index(drop=True),
                                      actual.sort_values("station").reset_index(drop=True))
        for row in actual.itertuples():
            i = int(np.flatnonzero(names == row.station)[0])
            t = train[train//n_month == i] % n_month
            median = np.median(truth[i, t].astype(float))
            recorded = doc.loc[doc.split_seed.eq(split) & doc.station.eq(row.station), "doc_median"].iloc[0]
            np.testing.assert_allclose(recorded, median, rtol=1e-12, atol=1e-12)
        val_union.update(names[np.unique(val//n_month)])
        for model, partition_errors in direct.items():
            seed_errors = []
            for seed in (42, 43, 44):
                path = Path(f"experiments/phase4_transfer/doc_current_availability_attention_v1/runs/split{split}_seed{seed}/predictions.parquet")
                frame = pd.read_parquet(path, filters=[("model_name", "==", model)])
                if set(frame.cell) != set(val) or not frame.visibility_role.eq("val").all():
                    raise ValueError("current atlas is not source validation")
                seed_errors.append(float(np.mean(np.abs(frame.y_pred.to_numpy()-frame.y_true.to_numpy()))))
            partition_errors.append(float(np.mean(seed_errors)))
    overall = model_effects[model_effects.group.eq("overall") & ~model_effects["tail"]
        & model_effects.reference.eq("station_hidden_trees")].iloc[0]
    current, trees = np.mean(direct["available_real_integrated"]), np.mean(direct["station_hidden_trees"])
    np.testing.assert_allclose([overall.candidate_mae, overall.reference_mae, overall.relative_gain_pct],
                               [current, trees, 100*(trees-current)/trees], rtol=1e-12, atol=1e-12)
    if overall.n_stations_unique != len(val_union):
        raise ValueError("bootstrap station population mismatch")
    balanced = pd.read_csv(a/"balanced_source_upstream_doc_associations.csv", dtype={"source": str, "target": str})
    if not balanced.groupby(["split_seed", "source", "target"]).lag_months.nunique().eq(5).all():
        raise ValueError("balanced lag panel changes its partition-edge population")
    for name in ("current_model_structure_effects.csv", "historical_message_structure_effects.csv"):
        frame = pd.read_csv(a/name)
        np.testing.assert_allclose(frame.relative_gain_pct, 100*(frame.reference_mae-frame.candidate_mae)/frame.reference_mae, rtol=1e-10)
        if not frame.bootstrap_draws.eq(5000).all():
            raise ValueError("atlas requires 5000 station draws")
    report = {"status": "passed", "physical_stations": len(structures), "physical_edges": len(paths),
              "source_validation_stations_unique": len(val_union), "source_hashes_checked": len(manifest["source_hashes"]),
              "independent_overall_current_mae": current, "independent_overall_trees_mae": trees,
              "checks": ["source content", "VAA joins and headwaters", "inlet degree", "physical station-edge paths",
                         "train-role observation counts and DOC medians", "source-val query identities",
                         "independent overall seed then partition arithmetic", "gain arithmetic and bootstrap count",
                         "fixed partition-edge population across all lags"],
              "no_fitting": True}
    (ROOT/"verification.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
