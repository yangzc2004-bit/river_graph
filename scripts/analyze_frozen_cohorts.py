"""Cohort and graph-position decomposition for current frozen predictions."""

from __future__ import annotations

import argparse
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import torch

from river_graph.experiments.evaluate import metrics

MASKS = [
    "e2a_strict",
    "e2b_partial",
    "e3_spatial_seed42",
    "e3_spatial_seed43",
    "e3_spatial_seed44",
]


def metadata(dataset: dict, nodes_path: Path) -> pd.DataFrame:
    sites = [str(site) for site in dataset["site_no"]]
    nodes = pd.read_csv(nodes_path, dtype={"site_no": str})
    meta = pd.DataFrame({"station": sites}).merge(
        nodes[["site_no", "site_tp_cd", "station_nm"]].rename(
            columns={"site_no": "station"}
        ),
        on="station",
        how="left",
        validate="one_to_one",
    )
    meta["cohort"] = np.select(
        [meta.site_tp_cd.eq("ST"), meta.site_tp_cd.eq("LK")],
        ["ST", "LK"],
        default="other",
    )
    graph = nx.DiGraph()
    graph.add_nodes_from(range(len(sites)))
    graph.add_edges_from(dataset["edge_index"].T.tolist())
    undirected = graph.to_undirected()
    meta["upstream_degree"] = [graph.in_degree(i) for i in range(len(sites))]
    meta["downstream_degree"] = [graph.out_degree(i) for i in range(len(sites))]
    meta["graph_role"] = np.select(
        [
            (meta.upstream_degree == 0) & (meta.downstream_degree == 0),
            meta.upstream_degree == 0,
            meta.downstream_degree == 0,
        ],
        ["isolated", "no_upstream_monitor", "no_downstream_monitor"],
        default="internal_monitor",
    )
    site_index = {site: i for i, site in enumerate(sites)}
    lake_nodes = {site_index[site] for site in meta.loc[meta.cohort == "LK", "station"]}
    distance = (
        dict(nx.multi_source_dijkstra_path_length(undirected, lake_nodes))
        if lake_nodes
        else {}
    )
    meta["lake_distance"] = [distance.get(i, np.nan) for i in range(len(sites))]
    meta["lake_context"] = np.select(
        [meta.cohort.eq("LK"), meta.lake_distance.eq(1), meta.lake_distance.eq(2)],
        ["lake_station", "one_edge_from_lake", "two_edges_from_lake"],
        default="no_lake_context",
    )
    return meta


def load_pair(pred_dir: Path, prefix: str, seed: int, mask: str) -> pd.DataFrame:
    both = pd.read_parquet(
        pred_dir / f"FRZ_{prefix}_both_river_s{seed}__{mask}.parquet"
    )
    down = pd.read_parquet(
        pred_dir / f"FRZ_{prefix}_down_river_s{seed}__{mask}.parquet"
    )
    both = both[both.split == "test"][["station", "month", "y_true", "y_pred"]]
    down = down[down.split == "test"][["station", "month", "y_true", "y_pred"]]
    both = both.rename(columns={"y_true": "y_true_both", "y_pred": "y_pred_both"})
    down = down.rename(columns={"y_true": "y_true_down", "y_pred": "y_pred_down"})
    paired = both.merge(down, on=["station", "month"], validate="one_to_one")
    if not np.allclose(paired.y_true_both, paired.y_true_down, equal_nan=True):
        raise ValueError(f"label mismatch for {prefix} seed={seed} mask={mask}")
    paired["y_true"] = paired.y_true_both
    paired["seed"] = seed
    paired["mask"] = mask
    paired["scenario"] = "temporal" if mask.startswith("e2") else "spatial"
    return paired


def summarize(cells: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    rows = []
    for key, sub in cells.groupby(keys, observed=True, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        both = metrics(sub.y_true.to_numpy(), sub.y_pred_both.to_numpy())
        down = metrics(sub.y_true.to_numpy(), sub.y_pred_down.to_numpy())
        rows.append(
            {
                **dict(zip(keys, key, strict=True)),
                "n_cells": len(sub),
                "n_stations": sub.station.nunique(),
                "both_r2": both["r2"],
                "downstream_r2": down["r2"],
                "delta_r2": down["r2"] - both["r2"],
                "both_log_r2": both["log_r2"],
                "downstream_log_r2": down["log_r2"],
                "delta_log_r2": down["log_r2"] - both["log_r2"],
                "both_mae": both["mae"],
                "downstream_mae": down["mae"],
                "delta_mae": down["mae"] - both["mae"],
            }
        )
    return pd.DataFrame(rows)


def collect(dataset_path: Path, pred_dir: Path, prefix: str, nodes: Path) -> pd.DataFrame:
    dataset = torch.load(dataset_path, weights_only=False)
    meta = metadata(dataset, nodes)
    frames = [
        load_pair(pred_dir, prefix, seed, mask)
        for seed in (42, 43, 44)
        for mask in MASKS
    ]
    cells = pd.concat(frames, ignore_index=True)
    return cells.merge(meta, on="station", how="left", validate="many_to_one")


def compare_common_st(full: pd.DataFrame, st: pd.DataFrame) -> pd.DataFrame:
    keys = ["seed", "mask", "station", "month"]
    full = full[(full.scenario == "temporal") & (full.cohort == "ST")]
    st = st[st.scenario == "temporal"]
    common = full.merge(st, on=keys, suffixes=("_370", "_st"), validate="one_to_one")
    rows = []
    for (seed, mask), sub in common.groupby(["seed", "mask"]):
        row = {"seed": seed, "mask": mask, "n_cells": len(sub), "n_stations": sub.station.nunique()}
        for label in ("370", "st"):
            both = metrics(
                sub[f"y_true_{label}"].to_numpy(),
                sub[f"y_pred_both_{label}"].to_numpy(),
            )
            down = metrics(
                sub[f"y_true_{label}"].to_numpy(),
                sub[f"y_pred_down_{label}"].to_numpy(),
            )
            row[f"delta_r2_{label}"] = down["r2"] - both["r2"]
            row[f"delta_log_r2_{label}"] = down["log_r2"] - both["log_r2"]
            row[f"delta_mae_{label}"] = down["mae"] - both["mae"]
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="experiments/analysis_frozen_cohorts")
    ap.add_argument("--nodes", default="data/processed/graph_nodes.csv")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    nodes = Path(args.nodes)
    full = collect(
        Path("data/processed/mississippi_graph_graphfix_cached370.pt"),
        Path("experiments/predictions_frozen_370"),
        "370",
        nodes,
    )
    st = collect(
        Path("data/processed/mississippi_graph_graphfix_st357.pt"),
        Path("experiments/predictions_frozen_st357"),
        "st",
        nodes,
    )
    full.to_csv(out / "frozen_370_paired_cells.csv", index=False)
    st.to_csv(out / "frozen_st357_paired_cells.csv", index=False)
    for name, keys in {
        "cohort": ["scenario", "seed", "mask", "cohort"],
        "graph_role": ["scenario", "seed", "mask", "graph_role"],
        "lake_context": ["scenario", "seed", "mask", "lake_context"],
    }.items():
        summarize(full, keys).to_csv(out / f"frozen_370_by_{name}.csv", index=False)
    common = compare_common_st(full, st)
    common.to_csv(out / "common_st_temporal.csv", index=False)
    print("common ST temporal comparison")
    print(common.to_string(index=False))
    print("\n370 temporal cohort means")
    cohort = summarize(full[full.scenario == "temporal"], ["seed", "mask", "cohort"])
    print(
        cohort.groupby("cohort")[["delta_r2", "delta_log_r2", "delta_mae"]]
        .mean()
        .to_string()
    )
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
