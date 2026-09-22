#!/usr/bin/env python3
"""Task 7: field-campaign demo — a new researcher with K local samples.

Given one survey month and K support stations in a pseudo-new watershed,
produce:
- unsampled-station predictions (analytic_blend on climatology / H2 base)
- uncertainty via leave-one-support-out jackknife + distance-to-support
- which stations move the most because of support (influence)
- where to sample next (max jackknife variance, tie-break far from support)
- a screening flag for high-DOC (or high-pH) reaches

Writes tables + a short markdown report + map-like scatter figures.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from river_graph.experiments.evaluate import load_dataset
from river_graph.experiments.kshot import (
    MIN_QUERY,
    make_region_split,
    make_support_query_tasks,
    region_station_index,
)
from river_graph.models.gcn import GCNDocModel
from river_graph.models.support_encoder import analytic_blend, residual_idw


def hop_lookup(dataset: dict):
    g = nx.Graph()
    n = dataset["y"].shape[0]
    g.add_nodes_from(range(n))
    g.add_edges_from(dataset["edge_index"].T.tolist())
    length = dict(nx.all_pairs_shortest_path_length(g))
    return lambda a, b: length.get(a, {}).get(b)


def climatology_base(y, y_mask, train_rows, months) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    y_mask = np.asarray(y_mask)
    mo = np.array([int(m[5:7]) for m in months])
    base = np.zeros_like(y)
    gmean = float(y[y_mask].mean()) if y_mask.any() else 0.0
    train_mask = y_mask[train_rows]
    train_y = y[train_rows]
    for m in range(1, 13):
        cols = np.flatnonzero(mo == m)
        if cols.size == 0:
            continue
        cell_mask = train_mask[:, cols]
        mu = float(train_y[:, cols][cell_mask].mean()) if cell_mask.any() else gmean
        base[:, cols] = mu
    return base


def pick_demo_task(tasks, month: str | None, prefer_k: int = 5):
    """Pick a survey month that can host prefer_k support with useful queries."""
    cands = [t for t in tasks if max(t.support_by_k) >= 3 and len(t.query) >= 1]
    if not cands:
        return None, None
    if month is not None:
        for task in cands:
            if task.month.startswith(month):
                k = min(prefer_k, max(task.support_by_k))
                return task, k
    # prefer months that can take prefer_k support, then more query sites
    def key(t):
        max_k = max(t.support_by_k)
        return (min(max_k, prefer_k), max_k, len(t.query))

    task = max(cands, key=key)
    k = min(prefer_k, max(task.support_by_k))
    return task, k


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/processed/mississippi_graph_graphfix_st357.pt")
    ap.add_argument("--analyte", default="doc", choices=["doc", "ph", "spec_conductance"])
    ap.add_argument("--region", default="11010001")
    ap.add_argument("--month", default=None, help="YYYY-MM survey month; default richest task")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--base", choices=["climatology", "h2"], default="climatology")
    ap.add_argument("--nodes", default="data/processed/graph_nodes.csv")
    ap.add_argument("--out-dir", default="experiments/field_demo")
    ap.add_argument("--max-epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=20)
    args = ap.parse_args()

    if args.analyte == "ph":
        args.dataset = "data/processed/mississippi_graph_ph_st357.pt"
    elif args.analyte == "spec_conductance":
        args.dataset = "data/processed/mississippi_graph_spec_conductance_st357.pt"

    out = Path(args.out_dir) / f"{args.analyte}_{args.region}"
    out.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(args.dataset)
    y_mask = np.asarray(dataset["y_mask"])
    y = np.asarray(dataset["y"], dtype=np.float64)
    y_flat = y.ravel()
    sites = [str(s) for s in dataset["site_no"]]
    months = [str(m) for m in dataset["months"]]
    nodes = pd.read_csv(args.nodes, dtype={"site_no": str})
    hops = hop_lookup(dataset)
    t = y.shape[1]
    n = y.shape[0]

    rows = region_station_index(sites, nodes, args.region)
    region_rows = [int(i) for i in rows]
    split = make_region_split(y_mask, rows, seed=42)
    tasks = make_support_query_tasks(
        y_mask, rows, months, k_list=(0, 1, 3, 5, 10), min_query=MIN_QUERY, seed=42
    )
    task, k_used = pick_demo_task(tasks, args.month, prefer_k=args.k)
    if task is None:
        raise SystemExit("no feasible survey month")
    j = task.month_index
    support = list(task.support_by_k[k_used])
    # truth for scoring: observed region cells this month not in support
    query_observed = [i * t + j for i in region_rows if y_mask[i, j] and (i * t + j) not in set(support)]
    all_region_cells = [i * t + j for i in region_rows]

    train_rows = sorted({int(f) // t for f in split["train"]})
    if args.base == "h2":
        model = GCNDocModel(
            architecture="transport",
            variant="river",
            edge_set="river",
            edge_direction="both",
            seed=args.seed,
            lr=1e-3,
            max_epochs=args.max_epochs,
            patience=args.patience,
        )
        model.fit(dataset, split)
        pred0 = model.predict()
    else:
        pred0 = climatology_base(y, y_mask, train_rows, months)

    def blend_pred(sup: list[int], targets: list[int]) -> dict[int, float]:
        return analytic_blend(pred0, y, sup, targets, hops, t)

    # full-region prediction map (includes unobserved sites)
    pred_map = blend_pred(support, all_region_cells)
    # jackknife over support
    jack = {q: [] for q in all_region_cells}
    if len(support) >= 2:
        for drop in range(len(support)):
            sup_loo = [s for i, s in enumerate(support) if i != drop]
            p = blend_pred(sup_loo, all_region_cells)
            for q in all_region_cells:
                jack[q].append(p[q])
    else:
        for q in all_region_cells:
            jack[q].append(pred_map[q])

    # influence: |blend with support - climatology/K0|
    base_only = {q: float(pred0[q // t, q % t]) for q in all_region_cells}

    records = []
    for q in all_region_cells:
        row, col = q // t, q % t
        site = sites[row]
        js = jack[q]
        jmean = float(np.mean(js))
        jstd = float(np.std(js)) if len(js) > 1 else 0.0
        hs = [hops(s // t, row) for s in support]
        hs = [h for h in hs if h is not None]
        min_h = min(hs) if hs else np.nan
        y_true = float(y_flat[q]) if y_mask[row, col] else np.nan
        is_support = q in set(support)
        records.append(
            {
                "site_no": site,
                "month": task.month,
                "station_index": row,
                "is_support": is_support,
                "observed": bool(y_mask[row, col]),
                "y_true": y_true,
                "y_pred": pred_map[q],
                "y_base": base_only[q],
                "influence": pred_map[q] - base_only[q],
                "jackknife_mean": jmean,
                "jackknife_std": jstd,
                "min_support_hops": min_h,
            }
        )
    df = pd.DataFrame(records)

    # next-sample score: high uncertainty, far from support, not already sampled
    unsampled = df[(~df.is_support) & (~df.observed | True)].copy()
    unsampled = unsampled[~unsampled.is_support]
    # normalize
    def nz(s):
        s = s.astype(float)
        rng = s.max() - s.min()
        return (s - s.min()) / rng if rng > 0 else s * 0.0

    unsampled["next_score"] = (
        0.5 * nz(unsampled.jackknife_std)
        + 0.3 * nz(unsampled.min_support_hops.fillna(unsampled.min_support_hops.max()))
        + 0.2 * nz(unsampled.y_pred)
    )
    ranked = unsampled.sort_values("next_score", ascending=False)

    # ecological screening: high predicted DOC (or extreme pH)
    if args.analyte == "doc":
        thr = float(np.nanquantile(df.y_pred, 0.8))
        df["risk_flag"] = np.where(df.y_pred >= thr, "high_DOC_risk", "ok")
    elif args.analyte == "ph":
        thr_hi = float(np.nanquantile(df.y_pred, 0.9))
        thr_lo = float(np.nanquantile(df.y_pred, 0.1))
        df["risk_flag"] = np.where(
            df.y_pred >= thr_hi, "high_pH", np.where(df.y_pred <= thr_lo, "low_pH", "ok")
        )
    else:
        thr = float(np.nanquantile(df.y_pred, 0.8))
        df["risk_flag"] = np.where(df.y_pred >= thr, "high_EC", "ok")

    df.to_csv(out / "station_predictions.csv", index=False)
    ranked.to_csv(out / "next_samples.csv", index=False)

    # scoring on held-out observed queries
    qdf = df[df.observed & ~df.is_support]
    mae = float(np.mean(np.abs(qdf.y_true - qdf.y_pred))) if len(qdf) else np.nan
    mae0 = float(np.mean(np.abs(qdf.y_true - qdf.y_base))) if len(qdf) else np.nan

    # figures
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    ax = axes[0]
    sc = ax.scatter(df.station_index, df.y_pred, c=df.jackknife_std, cmap="magma_r", s=40)
    sup_rows = df[df.is_support].station_index.to_numpy()
    ax.scatter(sup_rows, df[df.is_support].y_true, marker="x", c="k", s=60, label="support")
    ax.set_xlabel("station index")
    ax.set_ylabel(f"predicted {args.analyte}")
    ax.set_title(f"{args.region} {task.month} K={k_used}  (color=uncertainty)")
    fig.colorbar(sc, ax=ax, fraction=0.046)
    ax.legend()
    ax = axes[1]
    if len(qdf):
        ax.scatter(qdf.y_true, qdf.y_pred, s=30, alpha=0.7)
        lim = max(qdf.y_true.max(), qdf.y_pred.max()) * 1.05
        lo = min(qdf.y_true.min(), qdf.y_pred.min()) * 0.95
        ax.plot([lo, lim], [lo, lim], "k--", lw=0.8)
        ax.set_xlim(lo, lim)
        ax.set_ylim(lo, lim)
    ax.set_xlabel("true")
    ax.set_ylabel("predicted")
    ax.set_title(f"observed queries MAE={mae:.3f} (base {mae0:.3f})")
    fig.savefig(out / "field_demo.png", dpi=150)
    plt.close(fig)

    report = f"""# Field demo — {args.analyte} / {args.region}

- Survey month: **{task.month}**
- Support (K={k_used}): {', '.join(sites[s // t] for s in support)}
- Base: {args.base}
- Observed queries scored: n={len(qdf)}, MAE={mae:.3f} (K=0 base MAE={mae0:.3f})

## Where to sample next (top 5)

| rank | site | pred | jackknife_std | min_hops | score |
|------|------|------|---------------|----------|-------|
"""
    for i, row in enumerate(ranked.head(5).itertuples(index=False), 1):
        report += (
            f"| {i} | {row.site_no} | {row.y_pred:.3f} | {row.jackknife_std:.3f} "
            f"| {row.min_support_hops} | {row.next_score:.3f} |\n"
        )
    report += f"""
## Screening flags

{df.risk_flag.value_counts().to_string()}

## Interpretation

- **Most support influence** (largest |Δ vs base|): see `influence` column in `station_predictions.csv`.
- **Still uncertain**: high `jackknife_std` and/or large `min_support_hops`.
- **Ecological call**: treat `risk_flag != ok` as priority reaches for follow-up.

Generated by `scripts/run_field_demo.py` at {time.strftime('%Y-%m-%dT%H:%M:%S')}.
"""
    (out / "REPORT.md").write_text(report, encoding="utf-8")
    (out / "protocol.json").write_text(
        json.dumps(
            {
                "analyte": args.analyte,
                "region": args.region,
                "month": task.month,
                "k": k_used,
                "base": args.base,
                "seed": args.seed,
                "n_support": len(support),
                "n_region_stations": len(region_rows),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(report)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
