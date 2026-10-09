"""Build physical river settings and source-only DOC/model evidence; no fitting."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from analyze_unified_doc_spatial import joint_station_bootstrap

from river_graph.experiments.provenance import sha256_file
from river_graph.topology.river_structure import (
    PROFILES,
    VAA_COLUMNS,
    ReachNetwork,
    source_doc_summaries,
    source_pair_associations,
)

ROOT = Path("experiments/phase4_transfer/doc_river_structure_atlas_v1")
CURRENT = Path("experiments/phase4_transfer/doc_current_availability_attention_v1")
HISTORICAL = Path("experiments/phase4_transfer/kgml_local_transport_v1/k1")
DATASET = Path("data/processed/mississippi_graph_graphfix_st357.pt")
NODES = Path("data/processed/graph_nodes_graphfix_st357.csv")
EDGES = Path("data/processed/graph_edges_graphfix_st357.csv")
VAA = Path("cache/nldplus_vaa.parquet")
STREAMCAT = Path("data/processed/streamcat_attributes.csv")
MASKS = Path("experiments/phase4_transfer/unified_doc_spatial_v1/confirmation/masks")
SPLITS, SEEDS = (142, 143, 144), (42, 43, 44)
CURRENT_ARMS = ("available_real_integrated", "available_seasonal_integrated",
                "unmonitored_integrated", "station_hidden_trees")


def structure_table(nodes, edges, network):
    rows = []
    for i, node in enumerate(nodes.itertuples()):
        row = network.neighbourhood(node.comid, node.measure)
        row.update(station=node.site_no, huc_cd=node.huc_cd,
                   latitude=node.dec_lat_va, longitude=node.dec_long_va,
                   sampled_in_degree=int(edges.target.eq(node.site_no).sum()),
                   sampled_out_degree=int(edges.source.eq(node.site_no).sum()))
        rows.append(row)
        if (i+1) % 50 == 0:
            print(f"Physical river neighbourhoods: {i+1}/{len(nodes)}", flush=True)
    table = pd.DataFrame(rows)
    if table.upstream_search_capped.any():
        raise ValueError("upstream search exceeded its resource limit")
    return table


def edge_table(nodes, edges, network):
    lookup = nodes.set_index("site_no")
    rows = []
    for edge in edges.itertuples():
        a, b = lookup.loc[edge.source], lookup.loc[edge.target]
        path = network.mainstem_path(a.comid, a.measure, b.comid, b.measure)
        rows.append({"source": edge.source, "target": edge.target, **path})
    return pd.DataFrame(rows)


def group_masks(frame):
    yield "overall", np.ones(len(frame), dtype=bool)
    for name in PROFILES:
        yield name, frame[name].to_numpy(dtype=bool)
    for name in ("1–3", "4–6", "7–10"):
        yield f"order_{name}", frame.order_band.eq(name).to_numpy()
    yield "physical_headwater", frame.physical_headwater.to_numpy(dtype=bool)


def paired_errors(panel, candidate, reference):
    keys = ["split_seed", "seed", "station", "cell"]
    a = panel[panel.model_name.eq(candidate)][[*keys, "y_true", "y_pred", "q90_train"]]
    b = panel[panel.model_name.eq(reference)][[*keys, "y_true", "y_pred"]]
    pair = a.merge(b, on=keys, validate="one_to_one", suffixes=("_candidate", "_reference"))
    if len(pair) != len(a) or len(pair) != len(b) or pair.empty:
        raise ValueError("comparison changes the query population")
    np.testing.assert_array_equal(pair.y_true_candidate, pair.y_true_reference)
    pair["candidate_error"] = np.abs(pair.y_pred_candidate-pair.y_true_candidate)
    pair["reference_error"] = np.abs(pair.y_pred_reference-pair.y_true_reference)
    return pair


def structure_effects(panel, structures, comparisons, *, draws, scope):
    summary, stations = [], []
    for candidate, reference in comparisons:
        pair = paired_errors(panel, candidate, reference).merge(structures, on="station", validate="many_to_one")
        for group, selected in group_masks(pair):
            for tail in (False, True):
                keep = selected & ((pair.y_true_candidate >= pair.q90_train).to_numpy() if tail else True)
                sub = pair[keep]
                if sub.empty:
                    continue
                # Station IDs are sampled jointly across partitions, and losses
                # are averaged across seeds before the partition estimator.
                result = joint_station_bootstrap(sub, draws=draws)
                cells = sub.groupby(["split_seed", "station", "cell"], as_index=False)[["candidate_error", "reference_error"]].mean()
                station = cells.groupby(["split_seed", "station"], as_index=False).agg(
                    candidate_mae=("candidate_error", "mean"), reference_mae=("reference_error", "mean"),
                    n_cells=("cell", "size"))
                station["comparison"] = f"{candidate}_vs_{reference}"
                station["group"], station["tail"], station["scope"] = group, tail, scope
                stations.append(station)
                per_partition = cells.groupby("split_seed")[["candidate_error", "reference_error"]].mean()
                summary.append({"candidate": candidate, "reference": reference, "group": group, "tail": tail,
                                "scope": scope, **result,
                                "station_equal_candidate_mae": float(station.groupby("split_seed").candidate_mae.mean().mean()),
                                "station_equal_reference_mae": float(station.groupby("split_seed").reference_mae.mean().mean()),
                                "positive_partitions": int((per_partition.candidate_error < per_partition.reference_error).sum()),
                                "partial_partitions": len(per_partition) < panel.split_seed.nunique(),
                                "few_stations": result["n_stations_unique"] < 10,
                                "tail_small_sample": tail and result["n_station_months_unique"] < 20})
    return pd.DataFrame(summary), pd.concat(stations, ignore_index=True)


def load_current(dataset, manifest):
    frames = []
    for split in SPLITS:
        with np.load(MASKS/f"split{split}.npz", allow_pickle=False) as mask:
            allowed = set(mask["val"].tolist())
            q90 = np.quantile(np.asarray(dataset["y"]).ravel()[mask["train"]], .90)
            if set(mask["train"]//len(dataset["months"])) & set(mask["val"]//len(dataset["months"])):
                raise ValueError("source train and validation stations overlap")
        for seed in SEEDS:
            run = CURRENT/"runs"/f"split{split}_seed{seed}"
            if not (run/"complete.json").is_file():
                raise ValueError(f"incomplete source panel: {run}")
            cfg = json.loads((run/"config.json").read_text())
            if cfg["evaluation_role"] != "source_validation_only" or cfg["dataset_hash"] != sha256_file(DATASET):
                raise ValueError("source role or dataset mismatch")
            if not np.isclose(cfg["q90_threshold_train"], q90):
                raise ValueError("tail threshold is not source-training derived")
            frame = pd.read_parquet(run/"predictions.parquet", filters=[("model_name", "in", list(CURRENT_ARMS))])
            if (set(frame.model_name) != set(CURRENT_ARMS) or not frame.visibility_role.eq("val").all()
                    or not frame.k.eq(0).all() or set(frame.cell) != allowed):
                raise ValueError("source panel is not the fixed K0 validation population")
            if frame.duplicated(["model_name", "cell"]).any() or not np.isfinite(frame[["y_true", "y_pred"]]).all().all():
                raise ValueError("invalid source predictions")
            idx = frame.cell.to_numpy()
            np.testing.assert_array_equal(frame.station.to_numpy(str), np.asarray(dataset["site_no"], str)[idx//len(dataset["months"])])
            np.testing.assert_array_equal(frame.y_true.to_numpy(), np.asarray(dataset["y"]).ravel()[idx])
            frame["q90_train"] = cfg["q90_threshold_train"]
            frames.append(frame)
            for name in ("config.json", "predictions.parquet", "complete.json"):
                path = run/name
                manifest[str(path)] = sha256_file(path)
    return pd.concat(frames, ignore_index=True)


def load_historical(dataset, manifest):
    panels = {}
    for mask_name in ("e2a_strict", "e3_spatial_seed42"):
        frames = []
        mask_path = HISTORICAL/"masks"/f"doc__{mask_name}.npz"
        with np.load(mask_path, allow_pickle=False) as mask:
            allowed = set(mask["val"].tolist())
            threshold = float(np.quantile(np.asarray(dataset["y"]).ravel()[mask["train"]], .90))
        manifest[str(mask_path)] = sha256_file(mask_path)
        for arm in ("residual_upstream", "residual_both", "residual_nomsg"):
            for seed in SEEDS:
                run = HISTORICAL/"runs"/f"{arm}__doc__{mask_name}__seed{seed}"
                meta = json.loads((run/"meta.json").read_text())
                path = run/"val_predictions.parquet"
                frame = pd.read_parquet(path)
                if (not frame.visibility_role.eq("val").all() or frame.visible_input.any()
                        or set(frame.cell) != allowed or sha256_file(path) != meta["artifacts"]["val_predictions.parquet"]
                        or meta["config"]["dataset_sha256"] != sha256_file(DATASET)):
                    raise ValueError("historical validation artifact mismatch")
                if not np.isfinite(frame[["y_true", "y_pred"]]).all().all():
                    raise ValueError("nonfinite historical prediction")
                frame["q90_train"], frame["split_seed"] = threshold, 0
                frames.append(frame)
                manifest[str(path)], manifest[str(run/"meta.json")] = sha256_file(path), sha256_file(run/"meta.json")
        panels[mask_name] = pd.concat(frames, ignore_index=True)
    return panels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    output = args.root/"analysis"
    output.mkdir(exist_ok=True)
    manifest = {str(p): sha256_file(p) for p in (DATASET, NODES, EDGES, VAA, STREAMCAT,
                args.root/"study_plan.md", Path(__file__), Path("src/river_graph/topology/river_structure.py"),
                Path("scripts/analyze_unified_doc_spatial.py"))}
    nodes = pd.read_csv(NODES, dtype={"site_no": str, "huc_cd": str})
    edges = pd.read_csv(EDGES, dtype={"source": str, "target": str})
    dataset = torch.load(DATASET, weights_only=False, map_location="cpu")
    if len(nodes) != 357 or len(edges) != 324 or nodes.site_no.duplicated().any():
        raise ValueError("unexpected ST357 topology")
    np.testing.assert_array_equal(nodes.site_no.to_numpy(), np.asarray(dataset["site_no"], str))
    network = ReachNetwork(pd.read_parquet(VAA, columns=list(VAA_COLUMNS)))
    structures = structure_table(nodes, edges, network)
    cat = pd.read_csv(STREAMCAT)
    cat["wetland_cover_pct"] = cat.pcthbwet2019ws+cat.pctwdwet2019ws
    structures = structures.merge(cat[["comid", "wetland_cover_pct"]], on="comid", how="left", validate="many_to_one")
    structures.to_csv(output/"station_structure.csv", index=False)
    paths = edge_table(nodes, edges, network)
    paths.to_csv(output/"station_edge_paths.csv", index=False)
    del network
    doc, correlations = [], []
    for split in SPLITS:
        path = MASKS/f"split{split}.npz"
        manifest[str(path)] = sha256_file(path)
        with np.load(path, allow_pickle=False) as mask:
            train = mask["train"]
        summary = source_doc_summaries(dataset, train)
        summary["split_seed"] = split
        doc.append(summary)
        corr = source_pair_associations(dataset, train, edges)
        corr["split_seed"] = split
        correlations.append(corr)
    doc = pd.concat(doc, ignore_index=True).merge(structures, on="station", validate="many_to_one")
    doc.to_csv(output/"source_doc_station_summaries.csv", index=False)
    # Descriptive station distributions: average partition summaries first;
    # repeated source roles never become additional ecological replicates.
    metrics = ["doc_median", "doc_cv", "doc_iqr", "seasonal_amplitude", "cq_log1p_slope"]
    station_doc = doc.groupby("station", as_index=False)[metrics].mean().merge(structures, on="station", validate="one_to_one")
    behaviour = []
    for group, select in group_masks(station_doc):
        for metric in metrics:
            values = station_doc.loc[select, metric].dropna()
            behaviour.append({"group": group, "metric": metric, "n_stations": len(values),
                "median": values.median(), "q25": values.quantile(.25), "q75": values.quantile(.75)})
    pd.DataFrame(behaviour).to_csv(output/"doc_behaviour_by_structure.csv", index=False)
    correlation = pd.concat(correlations, ignore_index=True).merge(paths, on=["source", "target"], validate="many_to_one")
    correlation.to_csv(output/"source_upstream_doc_associations.csv", index=False)
    qualified = correlation[correlation.rho_seasonal_anomaly.notna()]
    keys = ["split_seed", "source", "target"]
    complete_pairs = qualified.groupby(keys).lag_months.nunique().eq(5)
    selected_pairs = complete_pairs[complete_pairs].reset_index()[keys]
    balanced = qualified.merge(selected_pairs, on=keys, validate="many_to_one")
    balanced.to_csv(output/"balanced_source_upstream_doc_associations.csv", index=False)
    current = load_current(dataset, manifest)
    comparisons = [("available_real_integrated", arm) for arm in CURRENT_ARMS[1:]]
    effects, station_effects = structure_effects(current, structures, comparisons,
        draws=args.bootstrap_draws, scope="source_validation_selected_similarity_model")
    effects.to_csv(output/"current_model_structure_effects.csv", index=False)
    station_effects.to_csv(output/"current_station_effects.csv", index=False)
    old, old_station = [], []
    for mask_name, panel in load_historical(dataset, manifest).items():
        a, b = structure_effects(panel, structures,
            [("residual_upstream", "residual_nomsg"), ("residual_both", "residual_nomsg")],
            draws=args.bootstrap_draws, scope=f"historical_validation_{mask_name}")
        a["mask"], b["mask"] = mask_name, mask_name
        old.append(a)
        old_station.append(b)
    pd.concat(old, ignore_index=True).to_csv(output/"historical_message_structure_effects.csv", index=False)
    pd.concat(old_station, ignore_index=True).to_csv(output/"historical_message_station_effects.csv", index=False)
    counts = [{"profile": name, "label": label, "n_all_stations": int(structures[name].sum()),
               "n_source_train_unique": int(doc.loc[doc[name], "station"].nunique()),
               "n_current_validation_unique": int(current.loc[current.station.isin(structures.loc[structures[name], "station"]), "station"].nunique())}
              for name, label in PROFILES.items()]
    pd.DataFrame(counts).to_csv(output/"profile_counts.csv", index=False)
    (args.root/"sources.json").write_text(json.dumps({"source_hashes": manifest,
        "bootstrap_draws": args.bootstrap_draws, "no_training": True,
        "roles": "source train descriptions; source val current errors; separate historical val message errors",
        "overlapping_profiles": True, "geometry_only_cohort": 357,
        "new_geographical_or_external_test_read": False}, indent=2)+"\n")
    print(effects[(effects.group == "overall") & ~effects["tail"]][[
        "candidate", "reference", "candidate_mae", "reference_mae", "relative_gain_pct",
        "gain_ci_low_pct", "gain_ci_high_pct"]].to_string(index=False), flush=True)
    print("Atlas tables completed; no model was trained.", flush=True)


if __name__ == "__main__":
    main()
