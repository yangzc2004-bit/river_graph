"""Headwater recovery analysis: does ecological context fix headwater stations?

Compares H2 (no env, v03) vs H2E-Encoder (env, v04) on the E3 seed42
held-out stations diagnosed as hardest (upstream_degree=0, high DOC,
HUC2 10/11 headwaters). Retrains both models on that mask (deterministic,
same protocol) and writes experiments/analysis/headwater_recovery.csv.

Usage: python scripts/interpret_headwaters.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

from river_graph.experiments.evaluate import load_dataset, load_mask
from river_graph.models.gcn import GCNDocModel

OUT = Path("experiments/analysis")


def station_mae(pred, y, test_cells, i, t):
    sel = test_cells[test_cells // t == i] % t
    if len(sel) == 0:
        return np.nan
    return float(np.mean(np.abs(pred[i, sel] - y[i, sel])))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    split = load_mask("e3_spatial_seed42")
    diag = pd.read_csv(OUT / "e3_seed42_diagnosis.csv", dtype={"station_id": str})

    d3 = load_dataset("data/processed/mississippi_graph_v03.pt")  # no env
    d4 = load_dataset("data/processed/mississippi_graph_v04.pt")  # with env
    sites = d3["site_no"]
    t = d3["y"].shape[1]
    test_cells = split["test"]

    pred_h2 = GCNDocModel(variant="river", architecture="transport").fit_predict(d3, split)
    print("H2 (no env) done", flush=True)
    pred_h2e = GCNDocModel(variant="river", architecture="transport_enc",
                           env_encoder=True).fit_predict(d4, split)
    print("H2E-Encoder done", flush=True)

    y = d3["y"].numpy()
    rows = []
    for i, s in enumerate(sites):
        rows.append({
            "station_id": s,
            "mae_h2": station_mae(pred_h2, y, test_cells, i, t),
            "mae_h2e": station_mae(pred_h2e, y, test_cells, i, t),
        })
    res = pd.DataFrame(rows).merge(diag, on="station_id", how="inner")
    res["improvement"] = res["mae_h2"] - res["mae_h2e"]
    res = res.sort_values("improvement", ascending=False)
    res.to_csv(OUT / "headwater_recovery.csv", index=False)

    hw = res[res["upstream_degree"] == 0]
    print(f"\nheld-out stations: {len(res)}; headwater (upstream_degree=0): {len(hw)}")
    print(f"all held-out:  MAE H2 {res['mae_h2'].mean():.3f} -> H2E {res['mae_h2e'].mean():.3f}")
    print(f"headwater:     MAE H2 {hw['mae_h2'].mean():.3f} -> H2E {hw['mae_h2e'].mean():.3f}")
    print("\ntop 8 recovered stations (with ecological context):")
    cols = ["station_id", "mae_h2", "mae_h2e", "improvement", "hist_doc_mean", "huc2"]
    print(res.head(8)[cols].to_string(index=False))
    print(f"\nsaved {OUT / 'headwater_recovery.csv'}")


if __name__ == "__main__":
    main()
