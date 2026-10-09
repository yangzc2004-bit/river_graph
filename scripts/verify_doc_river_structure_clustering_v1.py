"""Independent count, response, geometry and figure checks for the DOC study."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, silhouette_score

from river_graph.experiments.provenance import sha256_file

ROOT = Path("experiments/phase4_transfer/doc_river_structure_clustering_v1")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    root, a = args.root, args.root/"analysis"
    meta = json.loads((root/"sources.json").read_text())
    for path, digest in meta["source_hashes"].items():
        assert sha256_file(Path(path)) == digest, path
    s = pd.read_csv(a/"station_classification.csv", dtype={"station": str})
    panel = pd.read_csv(a/"station_doc_response.csv", dtype={"station": str})
    assert len(s) == 357 and not s.station.duplicated().any()
    assert len(panel) == 340 and not panel.station.duplicated().any()
    assert panel.eligible.sum() == 333
    assert s.groupby("fine_class").cluster.nunique().eq(1).all()
    counts = pd.read_csv(a/"class_physical_summary.csv")
    assert counts.n_stations.sum() == 357
    cfg = json.loads((root/"classification.json").read_text())
    f = pd.read_csv(a/"physical_features.csv")[cfg["features"]].to_numpy()
    filled = np.where(np.isfinite(f), f, np.array(cfg["imputation_medians"]))
    x = (filled-np.array(cfg["scaler_mean"]))/np.array(cfg["scaler_scale"])
    for cols in cfg["blocks"].values():
        x[:, [cfg["features"].index(n) for n in cols]] /= np.sqrt(len(cols))
    labels = AgglomerativeClustering(n_clusters=cfg["selected_k"], linkage="ward").fit_predict(x)
    assert adjusted_rand_score(labels, s.cluster) == 1
    candidates = pd.read_csv(a/"cluster_candidates.csv")
    for row in candidates.itertuples():
        labs = s[f"candidate_k{row.k}"]
        assert np.isclose(row.silhouette, silhouette_score(x, labs))
    winner = candidates[candidates.minimum_class >= 20].sort_values(["silhouette", "k"], ascending=[False, True]).iloc[0].k
    assert winner == cfg["selected_k"]
    dataset_path = next(p for p in meta["source_hashes"] if p.endswith("st357.pt"))
    dataset = torch.load(dataset_path, map_location="cpu", weights_only=False)
    cells = np.load(a/"source_cells.npy")
    source_masks = [p for p in meta["source_hashes"] if p.endswith(".npz")]
    independently_allowed = np.unique(np.concatenate([np.load(p)["train"] for p in source_masks]))
    np.testing.assert_array_equal(cells, independently_allowed)
    y = np.asarray(dataset["y"])
    assert np.isclose(np.quantile(y.ravel()[cells], .9), meta["q90_source_threshold_mg_L"])
    # Recompute all station medians and observation counts without the analysis helper.
    lookup = {str(name): i for i, name in enumerate(dataset["site_no"])}
    for row in panel.itertuples():
        idx = lookup[row.station]
        times = cells[cells//y.shape[1] == idx] % y.shape[1]
        assert len(times) == row.n_doc
        assert np.isclose(np.median(y[idx, times]), row.doc_median)
        assert np.isclose(np.mean(y[idx, times] >= meta["q90_source_threshold_mg_L"]), row.q90_fraction)
    stats = pd.read_csv(a/"class_doc_response.csv")
    for row in stats.itertuples():
        values = panel.loc[panel.eligible & panel.cluster.eq(row.cluster), row.metric].dropna()
        assert len(values) == row.n_stations
        assert np.isclose(values.median(), row.median)
    pred = pd.read_csv(a/"huc4_blocked_predictions.csv", dtype={"station": str, "huc4": str})
    assert not pred.duplicated(["station", "model"]).any()
    assert pred.groupby("huc4").fold.nunique().eq(1).all()
    cv = pd.read_csv(a/"huc4_blocked_comparison.csv")
    for row in cv.itertuples():
        sub = pred[pred.fold.eq(row.fold) & pred.model.eq(row.model)]
        assert np.isclose(np.mean(abs(sub.y_log1p-sub.y_pred_log1p)), row.mae_log1p)
    diagnostics = pd.read_csv(a/"regression_diagnostics.csv")
    assert (diagnostics["rank"] == diagnostics.effective_parameters).all()
    assert np.isfinite(pd.read_csv(a/"adjusted_associations.csv")[["estimate", "ci_low", "ci_high"]]).all().all()
    figure_meta = json.loads((root/"figure_sources.json").read_text())
    for path, digest in {**figure_meta["input_hashes"], **figure_meta["figure_hashes"]}.items():
        assert sha256_file(Path(path)) == digest, path
    result = {"status": "passed", "n_structure_stations": len(s), "n_response_stations": len(panel),
              "n_eligible": int(panel.eligible.sum()), "unique_source_cells": len(cells),
              "selected_k": cfg["selected_k"], "classification_exact_replay": True,
              "all_station_medians_recomputed": True, "blocked_error_recomputed": True,
              "figure_hashes_current": True, "no_training": True}
    (root/"verification.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
