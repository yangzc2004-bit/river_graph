"""Geometry-only clustering of entire upstream river footprints."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

BLOCKS = {
    "shape": ("log_basin_aspect", "log_network_axis_ratio", "basin_compactness"),
    "branching": ("log_drainage_density", "log_junction_frequency", "mainstem_share", "hierarchy_order"),
    "organization": ("side_imbalance", "tributary_alignment", "confluence_position", "tributary_balance"),
}


def physical_features(frame):
    """Absence of tributaries/junctions is structural zero, not missing geometry."""
    x = frame.copy()
    for name in ("basin_aspect", "network_axis_ratio", "drainage_density", "junction_frequency"):
        raw = x[name].to_numpy(float)
        x["log_" + name] = np.log1p(raw) if name == "junction_frequency" else np.log(raw)
    absent_trib = x.mainstem_share.ge(1-1e-8)
    x.loc[absent_trib, ["side_imbalance", "tributary_alignment"]] = 0
    x.loc[x.n_mainstem_junctions.eq(0), "confluence_position"] = 0
    return x[[name for block in BLOCKS.values() for name in block]]


def prepare(frame):
    x = physical_features(frame)
    eligible = frame.n_reaches.ge(5) & np.isfinite(x.to_numpy()).all(axis=1)
    x = x.loc[eligible]
    selected, removed, weights = [], [], []
    for block, names in BLOCKS.items():
        kept = []
        for name in names:
            if x[name].std() < 1e-9:
                removed.append({"feature": name, "reason": "constant"})
            elif any(abs(x[name].corr(x[other], method="spearman")) > .90 for other in kept):
                removed.append({"feature": name, "reason": "within-block Spearman > 0.90"})
            else:
                kept.append(name)
        if not kept:
            raise ValueError(f"No informative features in {block}")
        selected.extend(kept)
        weights.extend([1/np.sqrt(len(kept))]*len(kept))
    z = StandardScaler().fit_transform(x[selected]) * weights
    return eligible, x[selected], z, removed, np.asarray(weights)


def classify(frame, *, draws=100, seed=42):
    eligible, x, z, removed, weights = prepare(frame)
    if len(x) < 30:
        raise ValueError(f"Only {len(x)} eligible complete networks; need 30")
    candidates, cuts = [], {}
    for k in range(3, min(8, len(x)-1)+1):
        labels = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(z)
        counts = np.bincount(labels)
        candidates.append({"k": k, "silhouette": silhouette_score(z, labels),
                           "minimum_class": int(counts.min()), "preferred_size": bool(counts.min() >= 10)})
        cuts[k] = labels
    candidates = pd.DataFrame(candidates)
    preferred = candidates.loc[candidates.preferred_size]
    if preferred.empty:
        raise ValueError("No 3–8-class solution has minimum class size 10; report continuous form")
    k = int(preferred.sort_values(["silhouette", "k"], ascending=[False, True]).iloc[0].k)
    # Stable display order: elongated footprints first, then decreasing mainstem
    # share. This changes class numbers only, not memberships or selection.
    labels = cuts[k]
    ordering = sorted(range(k), key=lambda c: (-frame.loc[x.index[labels == c], "basin_aspect"].median(),
                                               -frame.loc[x.index[labels == c], "mainstem_share"].median()))
    remap = {old: new+1 for new, old in enumerate(ordering)}
    labels = np.asarray([remap[v] for v in labels])
    rng = np.random.default_rng(seed)
    stability = []
    for draw in range(draws):
        index = np.sort(rng.choice(len(x), size=max(25, int(.8*len(x))), replace=False))
        # Preprocessing refit on each subset. Apply the declared full-sample
        # selected feature set; retained covariance directions may still change.
        subz = StandardScaler().fit_transform(x.iloc[index]) * weights
        for n, full in cuts.items():
            sub = AgglomerativeClustering(n_clusters=n, linkage="ward").fit_predict(subz)
            stability.append({"draw": draw, "k": n, "ari": adjusted_rand_score(full[index], sub)})
    classes = frame.copy()
    classes["cluster"] = pd.Series(pd.NA, index=frame.index, dtype="Int64")
    classes.loc[x.index, "cluster"] = labels
    classes["classification_status"] = np.where(eligible, "classified",
        np.where(frame.n_reaches.lt(5), "fewer_than_five_reaches", "nonfinite_shape_measurement"))
    candidate_assignments = frame.loc[x.index, ["station", "comid"]].copy()
    for n, cut in cuts.items():
        candidate_assignments[f"k{n}"] = cut+1
    candidate_assignments[f"k{k}"] = labels
    centroid_rows, representatives = [], []
    for c in range(1, k+1):
        membership = labels == c
        centre = z[membership].mean(axis=0)
        centroid_rows.append({"cluster": c, **dict(zip(x.columns, centre, strict=True))})
        distances = np.linalg.norm(z[membership]-centre, axis=1)
        members = classes.loc[x.index[membership]].copy()
        members["centroid_distance"] = distances
        for rank, (_, row) in enumerate(members.sort_values(["centroid_distance", "station"]).head(3).iterrows(), 1):
            representatives.append({"cluster": c, "rank": rank, "station": row.station,
                                    "comid": int(row.comid), "centroid_distance": row.centroid_distance})
    return {"classes": classes, "features": x, "candidates": candidates,
            "stability": pd.DataFrame(stability), "centroids": pd.DataFrame(centroid_rows),
            "candidate_assignments": candidate_assignments,
            "representatives": pd.DataFrame(representatives), "selected_k": k,
            "removed": removed, "selected_features": list(x.columns)}
