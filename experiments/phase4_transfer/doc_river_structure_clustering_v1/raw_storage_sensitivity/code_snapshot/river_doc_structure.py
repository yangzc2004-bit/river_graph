"""Station-centred river classification and source-only DOC association tools."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.impute import SimpleImputer
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

BLOCKS = {
    "scale": ("stream_order", "log_area", "log_slope"),
    "branching": ("junction_density_20", "junction_density_5", "junction_density_50",
                  "log_length_20", "log_length_5", "log_length_50"),
    "mixing": ("major_fraction_20", "tributary_balance_5"),
    "storage": ("storage_fraction_20km", "storage_fraction_5km", "storage_fraction_50km"),
}


def huc_prefix(value, digits=4):
    """NWIS codes lose leading zeroes; HUC8/HUC12 have even digit counts."""
    text = str(value).split(".")[0]
    if not text.isdigit() or len(text) not in (7, 8, 11, 12):
        raise ValueError(f"unexpected HUC code: {value}")
    if len(text) % 2:
        text = "0"+text
    return text[:digits]


def physical_features(frame):
    """Only physical topology enters this function; labels/DOC are ignored."""
    out = pd.DataFrame(index=frame.index)
    out["stream_order"] = frame.stream_order
    out["log_area"] = np.log1p(frame.drainage_area_km2.clip(lower=0))
    out["log_slope"] = np.log10(frame.slope.clip(lower=1e-6))
    for radius in (5, 20, 50):
        length = frame[f"upstream_length_{radius}km"]
        out[f"junction_density_{radius}"] = np.log1p(
            100*frame[f"upstream_junction_count_{radius}km"].div(length.where(length > 0)))
        out[f"log_length_{radius}"] = np.log1p(length)
        out[f"storage_fraction_{radius}km"] = frame[f"storage_fraction_{radius}km"]
    denominator = frame.upstream_junction_count_20km
    out["major_fraction_20"] = frame.upstream_major_count_20km.div(denominator.where(denominator > 0)).where(denominator > 0, 0)
    out["tributary_balance_5"] = frame.largest_minor_area_share_5km
    return out.replace([np.inf, -np.inf], np.nan)


def retained_features(frame):
    selected, block_map = [], {}
    for block, names in BLOCKS.items():
        keep = []
        for name in names:
            if frame[name].nunique() <= 1:
                continue
            if any(abs(frame[name].corr(frame[other], method="spearman")) > .90 for other in keep):
                continue
            keep.append(name)
        if not keep:
            raise ValueError(f"physical feature block has no variation: {block}")
        selected.extend(keep)
        block_map[block] = keep
    return selected, block_map


def structure_matrix(frame, names, blocks):
    imputer, scaler = SimpleImputer(strategy="median"), StandardScaler()
    x = scaler.fit_transform(imputer.fit_transform(frame[names]))
    for cols in blocks.values():
        idx = [names.index(name) for name in cols]
        x[:, idx] /= np.sqrt(len(cols))
    return x, imputer, scaler


def classify_rivers(features, *, stability_draws=100, seed=42):
    names, blocks = retained_features(features)
    x, imputer, scaler = structure_matrix(features, names, blocks)
    candidates, all_labels, stability = [], {}, []
    for k in (3, 4, 5, 6):
        labels = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(x)
        all_labels[k] = labels
        candidates.append({"k": k, "silhouette": silhouette_score(x, labels),
                           "minimum_class": int(np.bincount(labels).min())})
        rng = np.random.default_rng(seed)
        for draw in range(stability_draws):
            sub = np.sort(rng.choice(len(x), int(.8*len(x)), replace=False))
            subnames, subblocks = retained_features(features.iloc[sub])
            sx, _, _ = structure_matrix(features.iloc[sub], subnames, subblocks)
            sl = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(sx)
            stability.append({"k": k, "draw": draw, "ari": adjusted_rand_score(labels[sub], sl)})
    candidates = pd.DataFrame(candidates)
    qualified = candidates[candidates.minimum_class >= 20]
    if qualified.empty:
        qualified = candidates[candidates.minimum_class == candidates.minimum_class.max()]
    winner = int(qualified.sort_values(["silhouette", "k"], ascending=[False, True]).iloc[0].k)
    # IDs increase in river size; relabelling has no DOC dependency.
    chosen = all_labels[winner]
    order = sorted(np.unique(chosen), key=lambda c: features.loc[chosen == c, "log_area"].median())
    remap = {old: i+1 for i, old in enumerate(order)}
    chosen = np.asarray([remap[v] for v in chosen])
    return chosen, candidates, pd.DataFrame(stability), all_labels, names, blocks, x, imputer, scaler


def source_station_response(dataset, cells):
    """Unique permitted source cells only; no repeated partition observations."""
    y, x = np.asarray(dataset["y"]), np.asarray(dataset["x"])
    ym, xm = np.asarray(dataset["y_mask"], bool), np.asarray(dataset["x_mask"], bool)
    cells = np.asarray(cells, dtype=int)
    if cells.ndim != 1 or len(cells) != len(np.unique(cells)) or (cells < 0).any() or (cells >= y.size).any():
        raise ValueError("unique in-range source cells required")
    if not ym.ravel()[cells].all():
        raise ValueError("source cells must have observed DOC")
    truth = y.ravel()[cells].astype(float)
    if not np.isfinite(truth).all() or (truth < 0).any():
        raise ValueError("source DOC must be finite and nonnegative")
    threshold = float(np.quantile(truth, .9))
    dates = pd.DatetimeIndex(dataset["months"])
    flow = list(dataset["feature_channels"]).index("discharge")
    temp = list(dataset["feature_channels"]).index("temperature")
    rows, seasonal = [], []
    for node in np.unique(cells//y.shape[1]):
        times = cells[cells//y.shape[1] == node] % y.shape[1]
        values = y[node, times].astype(float)
        months, years = dates.month.to_numpy()[times], dates.year.to_numpy()[times]
        phase = 2*np.pi*months/12
        span = years.max()-years.min()
        year = (years-years.mean())/10
        design = np.column_stack([np.ones(len(times)), np.sin(phase), np.cos(phase), year])
        amp, peak = np.nan, np.nan
        eligible = len(values) >= 12 and len(np.unique(months)) >= 6
        if eligible and np.linalg.matrix_rank(design) == 4:
            coef = np.linalg.lstsq(design, np.log1p(values), rcond=None)[0]
            amp = 2*float(np.hypot(coef[1], coef[2]))  # peak-to-trough in log1p
            curve = coef[1]*np.sin(2*np.pi*np.arange(1, 13)/12)+coef[2]*np.cos(2*np.pi*np.arange(1, 13)/12)
            peak = int(np.argmax(curve)+1)
        valid_flow = xm[node, times, flow] & np.isfinite(x[node, times, flow]) & (x[node, times, flow] >= 0)
        valid_temp = xm[node, times, temp] & np.isfinite(x[node, times, temp])
        slope = np.nan
        if valid_flow.sum() >= 24:
            q = np.log1p(x[node, times[valid_flow], flow])
            cq = np.column_stack([design[valid_flow], q])
            if np.linalg.matrix_rank(cq) == 5:
                slope = float(np.linalg.lstsq(cq, np.log1p(values[valid_flow]), rcond=None)[0][-1])
        station = str(dataset["site_no"][node])
        rows.append({"station": station, "n_doc": len(values), "n_calendar_months": len(np.unique(months)),
                     "eligible": eligible, "doc_median": np.median(values), "doc_mean": values.mean(),
                     "doc_cv": values.std()/values.mean() if values.mean() > 0 else np.nan,
                     "doc_iqr": np.subtract(*np.quantile(values, [.75, .25])),
                     "q90_fraction": np.mean(values >= threshold), "harmonic_amplitude": amp, "peak_month": peak,
                     "cq_slope": slope, "n_doc_flow": int(valid_flow.sum()), "median_year": np.median(years),
                     "record_span": span, "median_flow": np.median(x[node, times[valid_flow], flow]) if valid_flow.any() else np.nan,
                     "median_temperature": np.median(x[node, times[valid_temp], temp]) if valid_temp.any() else np.nan,
                     "flow_available": valid_flow.mean(), "temperature_available": valid_temp.mean()})
        for month in np.unique(months):
            seasonal.append({"station": station, "month": month, "doc_median": np.median(values[months == month]),
                             "n_cells": int((months == month).sum())})
    return pd.DataFrame(rows), pd.DataFrame(seasonal), threshold


def bootstrap_ols(x, y, *, draws=2000, seed=42):
    """Whole-station pairs bootstrap; each design row represents one station."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("finite regression arrays required")
    point = np.linalg.lstsq(x, y, rcond=None)[0]
    rng = np.random.default_rng(seed)
    estimates = []
    for start in range(0, draws, 100):
        w = rng.multinomial(len(y), np.full(len(y), 1/len(y)), size=min(100, draws-start))
        gram = np.einsum("bi,ij,ik->bjk", w, x, x, optimize=True)
        rhs = np.einsum("bi,ij,i->bj", w, x, y, optimize=True)
        estimates.append(np.einsum("bij,bj->bi", np.linalg.pinv(gram), rhs))
    boot = np.concatenate(estimates)
    return point, boot, {"n_stations": len(y), "n_parameters": x.shape[1],
                         "rank": int(np.linalg.matrix_rank(x)), "condition": float(np.linalg.cond(x))}
