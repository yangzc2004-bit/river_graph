"""Stratified diagnostics for the graphfix_cached370 H2 predictions.

This script reads stored predictions only. It maps each test cell back to its
station, river-network position, environmental regime, covariate availability,
and contemporaneous graph context, then writes compact audit tables.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from river_graph.experiments.evaluate import metrics

REGIME_COLUMNS = [
    "stream_order",
    "log_drainage_area",
    "slope",
    "is_monitoring_headwater",
    "forest_pct",
    "agriculture_pct",
    "urban_pct",
    "wetland_pct",
    "precipitation",
    "log_mean_temperature_shifted",
    "soil_organic_matter",
    "elevation",
    "baseflow_index",
]


def huc2(value: object) -> str:
    digits = "".join(ch for ch in str(value) if ch.isdigit())
    if len(digits) < 7:
        return "unknown"
    width = 8 if len(digits) <= 8 else 10 if len(digits) <= 10 else 12
    return digits.zfill(width)[:2]


def qgroup(values: pd.Series, labels: list[str]) -> pd.Series:
    """Stable equal-frequency groups; duplicate cut points remain explicit."""
    ranked = values.rank(method="average", pct=True)
    bins = np.linspace(0, 1, len(labels) + 1)
    return pd.cut(ranked, bins=bins, labels=labels, include_lowest=True)


def summarize(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    rows = []
    for key, sub in df.groupby(keys, observed=True, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        m = metrics(sub["y_true"].to_numpy(), sub["y_pred"].to_numpy())
        rows.append({
            **dict(zip(keys, key, strict=True)),
            "n_cells": len(sub),
            "n_stations": sub["station"].nunique(),
            **m,
        })
    return pd.DataFrame(rows)


def station_summary(cells: pd.DataFrame) -> pd.DataFrame:
    rows = []
    meta = [
        "scenario", "mask", "station", "huc2", "graph_role",
        "upstream_degree", "downstream_degree", "stream_order",
        "log_drainage_area", "slope", "forest_pct", "agriculture_pct",
        "urban_pct", "wetland_pct", "precipitation", "elevation",
        "baseflow_index", "flow_coverage", "temperature_coverage",
        "network_distance_to_train",
    ]
    for _, sub in cells.groupby(["mask", "station"], sort=False):
        m = metrics(sub["y_true"].to_numpy(), sub["y_pred"].to_numpy())
        row = {c: sub[c].iloc[0] for c in meta}
        row.update(m)
        row["true_mean"] = sub["y_true"].mean()
        row["true_max"] = sub["y_true"].max()
        row["neighbor_context_fraction"] = (sub["neighbor_train_doc_count"] > 0).mean()
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset",
        default="data/processed/mississippi_graph_graphfix_cached370.pt",
    )
    ap.add_argument("--masks-dir", default="experiments/masks_graphfix_cached370")
    ap.add_argument(
        "--pred-dir", default="experiments/predictions_graphfix_cached370"
    )
    ap.add_argument("--out-dir", default="experiments/analysis_graphfix_cached370")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset = torch.load(args.dataset, weights_only=False)
    sites = [str(s) for s in dataset["site_no"]]
    site_index = {s: i for i, s in enumerate(sites)}
    months = [str(m) for m in dataset["months"]]
    month_index = {m: i for i, m in enumerate(months)}
    n, t = dataset["y"].shape

    graph = nx.DiGraph()
    graph.add_nodes_from(range(n))
    graph.add_edges_from(dataset["edge_index"].T.tolist())
    undirected = graph.to_undirected()
    neighbours = {i: set(undirected.neighbors(i)) for i in range(n)}

    regime = dataset["regime"].numpy()
    if regime.shape[1] != len(REGIME_COLUMNS):
        raise ValueError(f"expected {len(REGIME_COLUMNS)} regime columns, got {regime.shape}")
    station_meta = pd.DataFrame(regime, columns=REGIME_COLUMNS)
    station_meta["station"] = sites
    station_meta["upstream_degree"] = [graph.in_degree(i) for i in range(n)]
    station_meta["downstream_degree"] = [graph.out_degree(i) for i in range(n)]
    station_meta["graph_role"] = np.select(
        [
            (station_meta.upstream_degree == 0) & (station_meta.downstream_degree == 0),
            station_meta.upstream_degree == 0,
            station_meta.downstream_degree == 0,
        ],
        ["isolated", "no_upstream_monitor", "no_downstream_monitor"],
        default="internal_monitor",
    )
    station_meta["flow_coverage"] = dataset["x_mask"][:, :, 1].float().mean(1).numpy()
    station_meta["temperature_coverage"] = (
        dataset["x_mask"][:, :, 0].float().mean(1).numpy()
    )
    nodes = pd.read_csv("data/processed/graph_nodes.csv", dtype={"site_no": str})
    station_meta = station_meta.merge(
        nodes[["site_no", "huc_cd"]].rename(columns={"site_no": "station"}),
        on="station",
        how="left",
    )
    station_meta["huc2"] = station_meta["huc_cd"].map(huc2)
    station_meta = station_meta.drop(columns="huc_cd")

    for col in [
        "log_drainage_area", "forest_pct", "agriculture_pct", "urban_pct",
        "wetland_pct", "precipitation", "elevation", "baseflow_index",
    ]:
        station_meta[f"{col}_group"] = qgroup(
            station_meta[col], ["low", "middle", "high"]
        ).astype(str)
    station_meta["stream_order_group"] = pd.cut(
        station_meta["stream_order"],
        bins=[-np.inf, 3, 5, np.inf],
        labels=["order_1_3", "order_4_5", "order_6_plus"],
    ).astype(str)
    frames = []
    pred_files = sorted(Path(args.pred_dir).glob("H2_ext_river_s42__*.parquet"))
    if not pred_files:
        raise FileNotFoundError("no H2_ext prediction parquet files")
    for path in pred_files:
        pred = pd.read_parquet(path)
        test = pred[pred["split"] == "test"].copy()
        if test.empty:
            continue
        mask_name = str(test["mask"].iloc[0])
        scenario = "temporal" if mask_name.startswith("e2") else "spatial"
        with np.load(Path(args.masks_dir) / f"{mask_name}.npz") as z:
            train = np.asarray(z["train"], dtype=np.int64)
        train_set = set(train.tolist())
        train_sites = set((train // t).tolist())
        distance = nx.multi_source_dijkstra_path_length(
            undirected, train_sites, weight=None
        )

        test["scenario"] = scenario
        test["station"] = test["station"].astype(str)
        test["station_index"] = test["station"].map(site_index)
        test["month_index"] = test["month"].astype(str).map(month_index)
        if test[["station_index", "month_index"]].isna().any().any():
            raise ValueError(f"prediction grid does not match dataset: {path}")
        test["station_index"] = test["station_index"].astype(int)
        test["month_index"] = test["month_index"].astype(int)
        test["flow_available"] = [
            bool(dataset["x_mask"][i, j, 1])
            for i, j in zip(test.station_index, test.month_index, strict=True)
        ]
        test["temperature_available"] = [
            bool(dataset["x_mask"][i, j, 0])
            for i, j in zip(test.station_index, test.month_index, strict=True)
        ]
        test["neighbor_train_doc_count"] = [
            sum((k * t + j) in train_set for k in neighbours[i])
            for i, j in zip(test.station_index, test.month_index, strict=True)
        ]
        test["network_distance_to_train"] = test["station_index"].map(distance)
        test = test.merge(station_meta, on="station", how="left", validate="many_to_one")
        frames.append(test)

    cells = pd.concat(frames, ignore_index=True)
    observed = dataset["y"][dataset["y_mask"]].numpy()
    q90, q95, q99 = np.quantile(observed, [0.90, 0.95, 0.99])
    cells["concentration_band"] = pd.cut(
        cells["y_true"],
        bins=[-np.inf, q90, q95, q99, np.inf],
        labels=["typical_le_p90", "high_p90_p95", "very_high_p95_p99", "extreme_gt_p99"],
        include_lowest=True,
    ).astype(str)
    cells["neighbor_context_group"] = pd.cut(
        cells["neighbor_train_doc_count"],
        bins=[-np.inf, 0, 1, np.inf],
        labels=["none", "one", "two_plus"],
    ).astype(str)
    cells["network_distance_group"] = np.select(
        [
            cells["network_distance_to_train"].isna(),
            cells["network_distance_to_train"] == 0,
            cells["network_distance_to_train"] == 1,
        ],
        ["unreachable", "training_station", "one_edge"],
        default="two_plus_edges",
    )
    cells["absolute_error"] = (cells["y_true"] - cells["y_pred"]).abs()

    station = station_summary(cells)
    cells.to_csv(out_dir / "test_cell_errors.csv", index=False)
    station.to_csv(out_dir / "per_station_errors.csv", index=False)

    correlation_variables = [
        "upstream_degree", "downstream_degree", "stream_order",
        "log_drainage_area", "slope", "forest_pct", "agriculture_pct",
        "urban_pct", "wetland_pct", "precipitation", "elevation",
        "baseflow_index", "flow_coverage", "neighbor_context_fraction",
        "true_mean", "true_max",
    ]
    correlation_rows = []
    for scenario, sub in station.groupby("scenario"):
        for error_metric in ("mae", "log_mae"):
            for variable in correlation_variables:
                complete = sub[[variable, error_metric]].dropna()
                if complete[variable].nunique() < 2 or complete[error_metric].nunique() < 2:
                    rho, p_value = np.nan, np.nan
                else:
                    rho, p_value = spearmanr(
                        complete[variable], complete[error_metric]
                    )
                correlation_rows.append({
                    "scenario": scenario,
                    "error_metric": error_metric,
                    "variable": variable,
                    "n_station_masks": len(complete),
                    "spearman_rho": rho,
                    "p_value_descriptive": p_value,
                })
    pd.DataFrame(correlation_rows).to_csv(
        out_dir / "station_error_spearman.csv", index=False
    )

    typical_spatial = cells[
        (cells.scenario == "spatial")
        & (cells.concentration_band == "typical_le_p90")
    ].copy()
    typical_spatial["has_neighbor_context"] = (
        typical_spatial.neighbor_train_doc_count > 0
    )
    paired_rows = []
    for (mask_name, station_id), sub in typical_spatial.groupby(["mask", "station"]):
        with_context = sub[sub.has_neighbor_context].absolute_error
        without_context = sub[~sub.has_neighbor_context].absolute_error
        if len(with_context) >= 3 and len(without_context) >= 3:
            paired_rows.append({
                "mask": mask_name,
                "station": station_id,
                "n_with_context": len(with_context),
                "n_without_context": len(without_context),
                "mae_with_context": with_context.mean(),
                "mae_without_context": without_context.mean(),
                "mae_reduction": without_context.mean() - with_context.mean(),
            })
    paired = pd.DataFrame(paired_rows)
    paired.to_csv(out_dir / "paired_neighbor_context_typical.csv", index=False)
    rng = np.random.default_rng(42)
    reductions = paired["mae_reduction"].to_numpy()
    bootstrap = np.asarray([
        rng.choice(reductions, len(reductions), replace=True).mean()
        for _ in range(10_000)
    ])
    paired_summary = {
        "station_mask_pairs": len(paired),
        "mean_mae_reduction": float(reductions.mean()),
        "median_mae_reduction": float(np.median(reductions)),
        "fraction_improved": float((reductions > 0).mean()),
        "bootstrap_95_ci_mean": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
        "interpretation": (
            "Descriptive paired comparison only; the interval crossing zero "
            "means contemporaneous neighbor availability is not yet causal evidence."
        ),
    }
    (out_dir / "paired_neighbor_context_summary.json").write_text(
        json.dumps(paired_summary, indent=2), encoding="utf-8"
    )

    summaries = {
        "overall": summarize(cells, ["mask"]),
        "graph_role": summarize(cells, ["scenario", "graph_role"]),
        "stream_order": summarize(cells, ["scenario", "stream_order_group"]),
        "drainage_area": summarize(cells, ["scenario", "log_drainage_area_group"]),
        "land_cover_forest": summarize(cells, ["scenario", "forest_pct_group"]),
        "land_cover_agriculture": summarize(
            cells, ["scenario", "agriculture_pct_group"]
        ),
        "land_cover_urban": summarize(cells, ["scenario", "urban_pct_group"]),
        "wetland": summarize(cells, ["scenario", "wetland_pct_group"]),
        "precipitation": summarize(cells, ["scenario", "precipitation_group"]),
        "baseflow": summarize(cells, ["scenario", "baseflow_index_group"]),
        "huc2": summarize(cells, ["scenario", "huc2"]),
        "flow_availability": summarize(cells, ["scenario", "flow_available"]),
        "concentration": summarize(cells, ["scenario", "concentration_band"]),
        "neighbor_context_spatial": summarize(
            cells[cells.scenario == "spatial"], ["neighbor_context_group"]
        ),
        "neighbor_context_typical_spatial": summarize(
            cells[
                (cells.scenario == "spatial")
                & (cells.concentration_band == "typical_le_p90")
            ],
            ["neighbor_context_group"],
        ),
        "graph_role_typical_spatial": summarize(
            cells[
                (cells.scenario == "spatial")
                & (cells.concentration_band == "typical_le_p90")
            ],
            ["graph_role"],
        ),
        "stream_order_typical_spatial": summarize(
            cells[
                (cells.scenario == "spatial")
                & (cells.concentration_band == "typical_le_p90")
            ],
            ["stream_order_group"],
        ),
        "network_distance_spatial": summarize(
            cells[cells.scenario == "spatial"], ["network_distance_group"]
        ),
    }
    for name, table in summaries.items():
        table.to_csv(out_dir / f"by_{name}.csv", index=False)

    metadata = {
        "dataset": args.dataset,
        "prediction_files": [str(p) for p in pred_files],
        "n_test_cells": len(cells),
        "n_test_stations": cells.station.nunique(),
        "concentration_thresholds_mg_l": {
            "p90": float(q90), "p95": float(q95), "p99": float(q99)
        },
        "notes": {
            "graph_role": "roles refer to the monitoring graph, not true hydrologic sources",
            "neighbor_context": "count of adjacent monitoring stations with a train DOC observation in the same month",
            "network_distance": "undirected monitoring-graph distance to a station with at least one training label",
            "correlations": "descriptive station-mask Spearman correlations; repeated stations across masks are not independent replicates",
        },
    }
    (out_dir / "analysis_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"wrote {len(cells):,} test cells and {len(station):,} station-mask rows")
    print(json.dumps(metadata["concentration_thresholds_mg_l"], indent=2))
    for key in ("graph_role", "stream_order", "concentration", "neighbor_context_spatial"):
        print(f"\n[{key}]\n{summaries[key].to_string(index=False)}")


if __name__ == "__main__":
    main()
