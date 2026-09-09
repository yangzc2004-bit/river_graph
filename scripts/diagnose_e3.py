"""E3 seed42 diagnosis: what makes this spatial split hard?

Two questions (design_m4.md, H1.5 postmortem):
  Q1: is the H1 divergence driven by a few stations?
  Q2: or by a type of reach/region absent from training?

Outputs experiments/analysis/e3_seed42_diagnosis.csv (per held-out station)
+ e3_seed42_map.png + a console distribution comparison train vs held-out.

Usage: python scripts/diagnose_e3.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from river_graph.experiments.evaluate import load_dataset, load_mask
from river_graph.models.gcn import GCNDocModel

OUT = Path("experiments/analysis")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ds = load_dataset()
    split = load_mask("e3_spatial_seed42")
    nodes = pd.read_csv("data/processed/graph_nodes.csv",
                        dtype={"site_no": str, "huc_cd": str})
    edges = pd.read_csv("data/processed/graph_edges.csv", dtype=str)

    # retrain H1 on this mask to obtain predictions
    model = GCNDocModel(variant="river", architecture="directed")
    pred = model.fit_predict(ds, split)  # (N, T) mg/L

    y = ds["y"].numpy()
    sites = ds["site_no"]
    t = y.shape[1]
    held = set(split["held_out_sites"].tolist())
    held_rows = sorted(i for i, s in enumerate(sites) if s in held)
    print(f"held-out stations: {len(held_rows)}")

    # --- split analysis: does seed42 hold out a different ecology? ---
    def station_doc_stats(rows, cell_filter=None):
        vals = []
        for i in rows:
            cells = y[i][ds["y_mask"].numpy()[i]]
            if cell_filter is not None:
                cells = y[i][cell_filter[i]]
            vals.append(np.nanmean(cells) if len(cells) else np.nan)
        return np.array(vals)

    obs = ds["y_mask"].numpy()
    train_cells = np.zeros((len(sites), t), dtype=bool)
    train_cells.ravel()[split["train"]] = True
    train_rows = [i for i in range(len(sites)) if i not in held_rows]

    held_hist = station_doc_stats(held_rows)          # all-time mean of held-out
    train_hist = station_doc_stats(train_rows)         # all-time mean of training
    print("\nDOC all-time station means (mg/L):")
    print(f"  train stations: mean={np.nanmean(train_hist):.2f} "
          f"p90={np.nanpercentile(train_hist, 90):.2f} max={np.nanmax(train_hist):.2f}")
    print(f"  held-out      : mean={np.nanmean(held_hist):.2f} "
          f"p90={np.nanpercentile(held_hist, 90):.2f} max={np.nanmax(held_hist):.2f}")

    nd = nodes.set_index("site_no")
    print("\nheld-out stations by HUC2 region:")
    print(nd.loc[sorted(held), "huc_cd"].str[:2].value_counts().sort_index().to_string())

    # --- error attribution per held-out station ---
    in_deg = edges["target"].value_counts()
    out_deg = edges["source"].value_counts()
    rows = []
    for i in held_rows:
        s = sites[i]
        test_months = np.zeros(t, dtype=bool)
        test_months[split["test"] % t] = True  # month slots used in test
        test_cells = split["test"][split["test"] // t == i]
        tj = test_cells % t
        true_v = y[i, tj]
        pred_v = pred[i, tj]
        rows.append({
            "station_id": s,
            "n_test_cells": len(tj),
            "true_doc_mean": float(np.mean(true_v)),
            "pred_doc_mean": float(np.mean(pred_v)),
            "mae": float(np.mean(np.abs(true_v - pred_v))),
            "bias": float(np.mean(pred_v - true_v)),
            "lat": float(nd.loc[s, "dec_lat_va"]),
            "lon": float(nd.loc[s, "dec_long_va"]),
            "huc2": str(nd.loc[s, "huc_cd"])[:2],
            "upstream_degree": int(in_deg.get(s, 0)),
            "downstream_degree": int(out_deg.get(s, 0)),
            "hist_doc_mean": float(np.nanmean(y[i][obs[i]])),
        })
    diag = pd.DataFrame(rows).sort_values("mae", ascending=False)
    diag.to_csv(OUT / "e3_seed42_diagnosis.csv", index=False)

    total_mae = diag["mae"].mean()
    print(f"\noverall held-out MAE: {total_mae:.3f}")
    print("\ntop 10 worst stations:")
    print(diag.head(10).to_string(index=False))
    top5_share = diag.head(5)["mae"].sum() / diag["mae"].sum()
    print(f"\nQ1: top-5 worst stations contribute {top5_share:.0%} of total error")
    print("\nQ2: MAE by HUC2 region:")
    print(diag.groupby("huc2")["mae"].agg(["count", "mean"]).round(3).to_string())

    # --- map ---
    fig, ax = plt.subplots(figsize=(12, 9))
    tr_ll = nd.loc[[sites[i] for i in train_rows]]
    ax.scatter(tr_ll["dec_long_va"].astype(float), tr_ll["dec_lat_va"].astype(float),
               s=5, c="lightgray", label="train stations")
    sc = ax.scatter(diag["lon"], diag["lat"], c=diag["mae"], cmap="Reds",
                    s=25, vmin=0, label="held-out (color = MAE)")
    fig.colorbar(sc, ax=ax, label="MAE (mg/L)")
    ax.set_title("E3 seed42 held-out stations, colored by H1 error")
    fig.savefig(OUT / "e3_seed42_map.png", dpi=150, bbox_inches="tight")
    print(f"\nsaved {OUT / 'e3_seed42_diagnosis.csv'} and e3_seed42_map.png")


if __name__ == "__main__":
    main()
