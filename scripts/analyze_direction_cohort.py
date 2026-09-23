"""Decompose directional message-passing effects by station cohort and graph position.

This is a prediction-only analysis.  It pairs stored H2 bidirectional and
downstream-only predictions on the same station-month test cells, then asks
where the downstream advantage (if any) is concentrated.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import torch

from river_graph.experiments.evaluate import metrics

PAIRS_370 = {
    42: ("H2_ext_river_s42", "H2_downstream_river_s42"),
    43: ("H2_both_s43_river_s43", "H2_downstream_s43_river_s43"),
    44: ("H2_both_s44_river_s44", "H2_downstream_s44_river_s44"),
}
PAIRS_ST = {
    42: ("ST_H2_both_s42_river_s42", "ST_H2_downstream_s42_river_s42"),
    43: ("ST_H2_both_s43_river_s43", "ST_H2_downstream_s43_river_s43"),
    44: ("ST_H2_both_s44_river_s44", "ST_H2_downstream_s44_river_s44"),
}


def load_pair(
    pred_dir: Path, both_name: str, downstream_name: str, mask: str
) -> pd.DataFrame:
    """Load and pair test predictions for one seed and mask."""
    both_path = pred_dir / f"{both_name}__{mask}.parquet"
    downstream_path = pred_dir / f"{downstream_name}__{mask}.parquet"
    if not both_path.exists() or not downstream_path.exists():
        raise FileNotFoundError(
            f"missing pair for {mask}: {both_path}, {downstream_path}"
        )
    both = pd.read_parquet(both_path)
    downstream = pd.read_parquet(downstream_path)
    both = both[both["split"] == "test"][["station", "month", "y_true", "y_pred"]]
    downstream = downstream[downstream["split"] == "test"][
        ["station", "month", "y_true", "y_pred"]
    ]
    both = both.rename(columns={"y_true": "y_true_both", "y_pred": "y_pred_both"})
    downstream = downstream.rename(
        columns={"y_true": "y_true_downstream", "y_pred": "y_pred_downstream"}
    )
    paired = both.merge(
        downstream, on=["station", "month"], how="inner", validate="one_to_one"
    )
    if paired.empty:
        raise ValueError(f"empty paired test grid for {mask}")
    if not np.allclose(paired.y_true_both, paired.y_true_downstream, equal_nan=True):
        raise ValueError(f"labels differ between paired models for {mask}")
    paired["y_true"] = paired["y_true_both"]
    paired["delta_abs_error"] = (paired.y_true - paired.y_pred_both).abs() - (
        paired.y_true - paired.y_pred_downstream
    ).abs()
    paired["delta_squared_error"] = (paired.y_true - paired.y_pred_both) ** 2 - (
        paired.y_true - paired.y_pred_downstream
    ) ** 2
    return paired


def summarize(paired: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for key, sub in paired.groupby(keys, observed=True, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        both = metrics(sub.y_true.to_numpy(), sub.y_pred_both.to_numpy())
        downstream = metrics(sub.y_true.to_numpy(), sub.y_pred_downstream.to_numpy())
        rows.append(
            {
                **dict(zip(keys, key, strict=True)),
                "n_cells": len(sub),
                "n_stations": sub.station.nunique(),
                "both_r2": both["r2"],
                "downstream_r2": downstream["r2"],
                "delta_r2": downstream["r2"] - both["r2"],
                "both_log_r2": both["log_r2"],
                "downstream_log_r2": downstream["log_r2"],
                "delta_log_r2": downstream["log_r2"] - both["log_r2"],
                "both_mae": both["mae"],
                "downstream_mae": downstream["mae"],
                "delta_mae": downstream["mae"] - both["mae"],
                "mean_delta_abs_error": sub.delta_abs_error.mean(),
                "fraction_cell_improved": (sub.delta_abs_error > 0).mean(),
            }
        )
    return pd.DataFrame(rows)


def station_metadata(dataset: dict, nodes_path: Path) -> pd.DataFrame:
    sites = [str(s) for s in dataset["site_no"]]
    n = len(sites)
    graph = nx.DiGraph()
    graph.add_nodes_from(range(n))
    graph.add_edges_from(dataset["edge_index"].T.tolist())
    undirected = graph.to_undirected()

    nodes = pd.read_csv(nodes_path, dtype={"site_no": str})
    nodes["site_no"] = nodes["site_no"].astype(str)
    meta = pd.DataFrame({"station": sites})
    meta = meta.merge(
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
    meta["upstream_degree"] = [graph.in_degree(i) for i in range(n)]
    meta["downstream_degree"] = [graph.out_degree(i) for i in range(n)]
    meta["graph_role"] = np.select(
        [
            (meta.upstream_degree == 0) & (meta.downstream_degree == 0),
            meta.upstream_degree == 0,
            meta.downstream_degree == 0,
        ],
        ["isolated", "no_upstream_monitor", "no_downstream_monitor"],
        default="internal_monitor",
    )
    index = {s: i for i, s in enumerate(sites)}
    lake_nodes = {index[s] for s in meta.loc[meta.cohort == "LK", "station"]}
    meta["has_lake_neighbor"] = [
        bool(set(undirected.neighbors(i)) & lake_nodes) for i in range(n)
    ]
    lake_distance = {}
    if lake_nodes:
        lake_distance = dict(
            nx.multi_source_dijkstra_path_length(undirected, lake_nodes)
        )
    meta["lake_distance"] = [lake_distance.get(i, np.nan) for i in range(n)]
    meta["lake_context"] = np.select(
        [meta.cohort.eq("LK"), meta.has_lake_neighbor, meta.lake_distance.le(2)],
        ["lake_station", "one_edge_from_lake", "two_edges_from_lake"],
        default="no_lake_context",
    )
    return meta


def analyze_dataset(
    dataset_path: Path,
    pred_dir: Path,
    nodes_path: Path,
    pairs: dict[int, tuple[str, str]],
    out_dir: Path,
    label: str,
) -> dict[str, pd.DataFrame]:
    dataset = torch.load(dataset_path, weights_only=False)
    meta = station_metadata(dataset, nodes_path)
    frames = []
    for seed, (both_name, downstream_name) in pairs.items():
        for mask in ("e2a_strict", "e2b_partial"):
            frame = load_pair(pred_dir, both_name, downstream_name, mask)
            frame["seed"] = seed
            frame["mask"] = mask
            frame["dataset_label"] = label
            frames.append(frame)
    paired = pd.concat(frames, ignore_index=True)
    paired = paired.merge(meta, on="station", how="left", validate="many_to_one")
    if paired.cohort.isna().any():
        raise ValueError("some prediction stations are absent from graph node metadata")

    prefix = f"{label}_"
    paired.to_csv(out_dir / f"{prefix}paired_cells.csv", index=False)
    tables = {
        "overall": summarize(paired, ["seed", "mask"]),
        "cohort": summarize(paired, ["seed", "mask", "cohort"]),
        "lake_context": summarize(paired, ["seed", "mask", "lake_context"]),
        "graph_role": summarize(paired, ["seed", "mask", "graph_role"]),
        "station": summarize(
            paired, ["seed", "mask", "station", "cohort", "lake_context"]
        ),
    }
    for name, table in tables.items():
        table.to_csv(out_dir / f"{prefix}by_{name}.csv", index=False)
    return tables


def compare_common_st(out_dir: Path) -> pd.DataFrame:
    """Compare datasets on identical ST station-month test cells."""
    cached = pd.read_csv(out_dir / "cached370_paired_cells.csv", dtype={"station": str})
    st = pd.read_csv(out_dir / "st357_paired_cells.csv", dtype={"station": str})
    keys = ["seed", "mask", "station", "month"]
    cached = cached[cached.cohort == "ST"]
    st = st[st.cohort == "ST"]
    common = cached.merge(
        st,
        on=keys,
        how="inner",
        suffixes=("_cached370", "_st357"),
        validate="one_to_one",
    )
    rows: list[dict[str, object]] = []
    groupings = [
        ["seed", "mask"],
        ["seed", "mask", "lake_context_cached370"],
    ]
    for group_keys in groupings:
        for group, sub in common.groupby(group_keys, observed=True, dropna=False):
            group = group if isinstance(group, tuple) else (group,)
            row: dict[str, object] = {
                **dict(zip(group_keys, group, strict=True)),
                "n_cells": len(sub),
                "n_stations": sub.station.nunique(),
            }
            for label in ("cached370", "st357"):
                y_true = sub[f"y_true_{label}"]
                both_m = metrics(
                    y_true.to_numpy(), sub[f"y_pred_both_{label}"].to_numpy()
                )
                downstream_m = metrics(
                    y_true.to_numpy(), sub[f"y_pred_downstream_{label}"].to_numpy()
                )
                row[f"{label}_delta_r2"] = downstream_m["r2"] - both_m["r2"]
                row[f"{label}_delta_log_r2"] = downstream_m["log_r2"] - both_m["log_r2"]
                row[f"{label}_delta_mae"] = downstream_m["mae"] - both_m["mae"]
            rows.append(row)
    result = pd.DataFrame(rows)
    result.to_csv(out_dir / "common_st_effect_comparison.csv", index=False)
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="experiments/analysis_direction_cohort")
    ap.add_argument("--nodes", default="data/processed/graph_nodes.csv")
    ap.add_argument(
        "--dataset-370",
        default="data/processed/mississippi_graph_graphfix_cached370.pt",
    )
    ap.add_argument("--pred-370", default="experiments/predictions_graphfix_cached370")
    ap.add_argument(
        "--dataset-st", default="data/processed/mississippi_graph_graphfix_st357.pt"
    )
    ap.add_argument("--pred-st", default="experiments/predictions_graphfix_st357")
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for label, dataset, pred, pairs in (
        ("cached370", args.dataset_370, args.pred_370, PAIRS_370),
        ("st357", args.dataset_st, args.pred_st, PAIRS_ST),
    ):
        results[label] = analyze_dataset(
            Path(dataset), Path(pred), Path(args.nodes), pairs, out_dir, label
        )

    # Compact comparison of cohort-specific temporal effects across datasets.
    comparison = results["cached370"]["cohort"].merge(
        results["st357"]["cohort"],
        on=["seed", "mask", "cohort"],
        how="outer",
        suffixes=("_cached370", "_st357"),
    )
    comparison.to_csv(out_dir / "cohort_effect_comparison.csv", index=False)
    common = compare_common_st(out_dir)
    metadata = {
        "purpose": "prediction-only decomposition of downstream-vs-bidirectional effects",
        "datasets": {"cached370": args.dataset_370, "st357": args.dataset_st},
        "masks": ["e2a_strict", "e2b_partial"],
        "positive_delta_definition": "downstream-only minus bidirectional; positive delta_r2/log_r2 or positive mean_delta_abs_error favors downstream-only",
        "note": "ST357 and cached370 use the same temporal mask names; this is a descriptive matched-cohort comparison, not a causal estimate.",
    }
    (out_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print("cached370 cohort summary")
    print(results["cached370"]["cohort"].to_string(index=False))
    print("\nst357 cohort summary")
    print(results["st357"]["cohort"].to_string(index=False))
    print("\ncommon ST station-month comparison")
    print(common.to_string(index=False))
    print(f"\nwrote outputs to {out_dir}")


if __name__ == "__main__":
    main()
