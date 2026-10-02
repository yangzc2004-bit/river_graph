"""Source-selection transfer with source-fitted geomorphology descriptors.

This is an independent low-cost experiment.  It compares the existing
label-free station descriptor against the same descriptor augmented with
genuinely new physical fields (gauge elevation, reach length and reach area).
All imputation/scaling is fitted on source stations within each split.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from river_graph.experiments.evaluate import metrics
from river_graph.experiments.h3_masks import e3_internal_split
from river_graph.experiments.transfer import DATASETS
from river_graph.models.kgml_local_transport import (
    FIT_ROLES,
    TEST_ROLES,
    build_rf_features,
    target_values,
)

T = 654
MASK = Path("experiments/masks_stcore_v1/e3_spatial_seed42.npz")
NODE_CSV = Path("data/processed/graph_nodes_graphfix_st357.csv")
REACH_CSV = Path("data/processed/reach_attributes.csv")


def load_split(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {k: np.asarray(z[k], dtype=np.int64) for k in z.files if k in {"train", "val", "test", "context"}}


def _raw_descriptor(data: dict) -> np.ndarray:
    x = data["x"].numpy().astype(np.float64)
    xm = data["x_mask"].numpy().astype(np.float64)
    n, _, c = x.shape
    blocks: list[np.ndarray] = []
    for j in range(c):
        a, v = x[:, :, j], xm[:, :, j]
        den = np.maximum(v.sum(1), 1.0)
        mean = (a * v).sum(1) / den
        sd = np.sqrt(((a - mean[:, None]) ** 2 * v).sum(1) / den)
        qs = [np.asarray([np.quantile(a[i, v[i] > 0], q) if np.any(v[i] > 0) else 0.0 for i in range(n)]) for q in (0.1, 0.5, 0.9)]
        blocks.extend([mean, sd, *qs, v.mean(1)])
    blocks.extend([data["static"].numpy()[:, j].astype(np.float64) for j in range(data["static"].shape[1])])
    blocks.extend([data["regime"].numpy()[:, j].astype(np.float64) for j in range(data["regime"].shape[1])])
    edge = data["edge_index"].numpy()
    attr = data["edge_attr"].numpy().astype(np.float64)
    blocks.extend([np.bincount(edge[0], minlength=n), np.bincount(edge[1], minlength=n)])
    for j in range(attr.shape[1]):
        blocks.append(np.bincount(edge[1], weights=attr[:, j], minlength=n))
    return np.nan_to_num(np.stack(blocks, axis=1), nan=0.0, posinf=0.0, neginf=0.0)


def geomorph(data: dict) -> tuple[np.ndarray, dict]:
    nodes = pd.read_csv(NODE_CSV, dtype={"site_no": str})
    reach = pd.read_csv(REACH_CSV, dtype={"site_no": str})
    site_values = data["site_no"]
    if hasattr(site_values, "numpy"):
        site_values = site_values.numpy()
    site = pd.Series(np.asarray(site_values).astype(str))
    n = len(site)
    nd = nodes.drop_duplicates("site_no").set_index("site_no")
    rd = reach.drop_duplicates("site_no").set_index("site_no")
    alt = pd.to_numeric(site.map(nd["alt_va"]), errors="coerce").to_numpy(float)
    length = pd.to_numeric(site.map(rd["lengthkm"]), errors="coerce").to_numpy(float)
    area = pd.to_numeric(site.map(rd["areasqkm"]), errors="coerce").to_numpy(float)
    out = np.column_stack([alt, np.isnan(alt).astype(float), np.log1p(np.maximum(length, 0)), np.log1p(np.maximum(area, 0))])
    audit = {
        "columns": ["alt_va_station_elevation", "alt_va_missing", "log1p_reach_lengthkm", "log1p_reach_areasqkm"],
        "source_files": [str(NODE_CSV), str(REACH_CSV)],
        "missing_counts": {"alt_va": int(np.isnan(alt).sum()), "lengthkm": int(np.isnan(length).sum()), "areasqkm": int(np.isnan(area).sum())},
        "excluded_existing_regime": ["streamorde", "totdasqkm", "slope", "elevws", "bfiws", "landcover", "climate", "soil/organic matter"],
        "excluded_measure": "NLDI within-reach hydrolocation; not stream order or geomorphology",
    }
    assert len(out) == n
    return out, audit


def fit_transform(raw: np.ndarray, source_stations: np.ndarray) -> tuple[np.ndarray, dict]:
    imp = SimpleImputer(strategy="median").fit(raw[source_stations])
    imputed = imp.transform(raw)
    scaler = StandardScaler().fit(imputed[source_stations])
    return scaler.transform(imputed).astype(np.float32), {"source_count": len(source_stations), "imputer_statistics": imp.statistics_.tolist(), "scale_mean": scaler.mean_.tolist(), "scale_scale": scaler.scale_.tolist()}


def nearest(desc: np.ndarray, source: np.ndarray, q: int, k: int) -> np.ndarray:
    d = np.square(desc[source] - desc[q]).sum(1)
    return source[np.argsort(d)[: min(k, len(source))]]


def predict(data: dict, split: dict, fit_cells: np.ndarray, query_cells: np.ndarray, fit_x: np.ndarray, query_x: np.ndarray, desc: np.ndarray, k: int, seed: int, trees: int, leaf: int, jobs: int) -> np.ndarray:
    _, t = data["y"].shape
    source = np.unique(fit_cells // t)
    target = np.unique(query_cells // t)
    by_station = {int(s): fit_cells[fit_cells // t == s] for s in source}
    yz = target_values(data, "log1p").reshape(-1)
    pred = np.empty(len(query_cells), dtype=float)
    pos = {int(c): i for i, c in enumerate(query_cells)}
    for st in target:
        qcells = query_cells[query_cells // t == st]
        ns = nearest(desc, source, int(st), int(k))
        train = np.concatenate([by_station[int(s)] for s in ns])
        model = ExtraTreesRegressor(n_estimators=trees, min_samples_leaf=leaf, max_features=1.0, random_state=seed, n_jobs=jobs)
        model.fit(fit_x[train], yz[train])
        pp = np.expm1(model.predict(query_x[qcells]))
        for cell, value in zip(qcells, pp, strict=True):
            pred[pos[int(cell)]] = value
    return pred


def evaluate_one(data, split, raw_desc, geom, feature_set, k, seed, role, trees, leaf, jobs):
    source = np.unique(split["train"] // T)
    d = raw_desc if feature_set == "base" else np.column_stack([raw_desc, geom])
    desc, prep = fit_transform(d, source)
    fit_x = build_rf_features(data, split, FIT_ROLES, target_transform="log1p", include_network=True)
    # Internal validation must hide val labels; the frozen outer test view may
    # use train+val+context, matching the established evaluation protocol.
    qroles = FIT_ROLES if role == "internal_val" else TEST_ROLES
    q_x = build_rf_features(data, split, qroles, target_transform="log1p", include_network=True)
    qcells = split["test"] if role == "outer_e3_test" else split["val"]
    pred = predict(data, split, split["train"], qcells, fit_x, q_x, desc, k, seed, trees, leaf, jobs)
    y = data["y"].numpy().reshape(-1)
    row = {"feature_set": feature_set, "seed": seed, "k": k, "role": role, **metrics(y[qcells], pred)}
    return row, pred, prep


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=Path("experiments/phase4_transfer/spatial_adaptation/source_geomorph_v1"))
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    ap.add_argument("--ks", nargs="+", type=int, default=[20, 40, 80, 160])
    ap.add_argument("--trees", type=int, default=120)
    ap.add_argument("--leaf", type=int, default=4)
    ap.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args()
    data = torch.load(DATASETS["doc"], map_location="cpu", weights_only=False)
    outer = load_split(MASK)
    internal, _, _ = e3_internal_split(data["y_mask"].numpy(), outer["test"])
    raw = _raw_descriptor(data)
    gm, audit = geomorph(data)
    out = args.out_dir; out.mkdir(parents=True, exist_ok=True); (out / "predictions").mkdir(exist_ok=True)
    (out / "feature_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    rows = []; prep_records = {}
    for fs in ("base", "base_geomorph"):
        for seed in args.seeds:
            for k in args.ks:
                row, _, prep = evaluate_one(data, internal, raw, gm, fs, k, seed, "internal_val", args.trees, args.leaf, args.jobs)
                rows.append(row); prep_records[f"{fs}:{seed}:{k}:internal"] = prep
    val = pd.DataFrame(rows)
    sel = val.groupby(["feature_set", "k"], as_index=False).mae.mean().sort_values(["feature_set", "mae"]).groupby("feature_set", as_index=False).first().rename(columns={"k": "selected_k", "mae": "validation_mae"})
    outer_rows=[]
    for _, r in sel.iterrows():
        fs, k = str(r.feature_set), int(r.selected_k)
        for seed in args.seeds:
            row, pred, prep = evaluate_one(data, outer, raw, gm, fs, k, seed, "outer_e3_test", args.trees, args.leaf, args.jobs)
            outer_rows.append(row); prep_records[f"{fs}:{seed}:{k}:outer"] = prep
            np.save(out / "predictions" / f"{fs}_seed{seed}_k{k}_outer.npy", pred)
    val.to_csv(out / "internal_validation.csv", index=False); pd.DataFrame(outer_rows).to_csv(out / "outer_test.csv", index=False); sel.to_csv(out / "selected_config.csv", index=False)
    spec = {"version":"source_geomorph_v1", "seeds":args.seeds, "ks":args.ks, "trees":args.trees, "leaf":args.leaf, "outer_mask":str(MASK), "selection":"internal station-heldout E3 only; outer only selected k", "preprocessing":"median imputation and standardization fitted on source stations per split", "target":"DOC K0 spatial transfer"}
    (out / "spec.json").write_text(json.dumps(spec, indent=2) + "\n")
    (out / "preprocessing_summary.json").write_text(json.dumps(prep_records, indent=2) + "\n")
    print(val.groupby(["feature_set","k"]).mae.agg(["mean","std"]).to_string()); print(sel.to_string(index=False)); print(pd.DataFrame(outer_rows).to_string(index=False))


if __name__ == "__main__": main()
