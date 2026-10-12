"""Metrics and label-free adapters for the first DOC fusion benchmark.

This is a development comparison on previously examined geographical tasks.
Seeds repeat fits; they are never counted as independent rivers.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge

REGIONS = ("1013", "1019", "0708", "1030", "1101")
SEEDS = (42, 43, 44)
MODELS = ("source_seasonal_median", "ridge", "random_forest", "boosted_trees",
          "local_no_messages", "fixed_fusion")
MAX_LOG_DOC = 15.
# Three declared recipes per family; no target-dependent search expansion.
CANDIDATES = {
    "ridge": {
        "log_alpha_0_1": {"target": "log1p", "alpha": .1},
        "log_alpha_10": {"target": "log1p", "alpha": 10.},
        "native_alpha_10": {"target": "native", "alpha": 10.},
    },
    "random_forest": {
        "log_leaf3": {"target": "log1p", "min_samples_leaf": 3},
        "log_leaf10": {"target": "log1p", "min_samples_leaf": 10},
        "native_leaf3": {"target": "native", "min_samples_leaf": 3},
    },
    "boosted_trees": {
        "native_l1": {"target": "native", "loss": "absolute_error", "max_leaf_nodes": 15},
        "log_l2": {"target": "log1p", "loss": "squared_error", "max_leaf_nodes": 15},
        "log_l2_wide": {"target": "log1p", "loss": "squared_error", "max_leaf_nodes": 31},
    },
}


def tabular_features(inputs, cells, months):
    """Same local information as the neural recipe, preserving the full history."""
    cells = np.asarray(cells, dtype=np.int64)
    node, month = cells // months, cells % months
    environment = inputs["environment"].numpy()[node]
    windows = inputs["windows"].numpy()[month, node].reshape(len(cells), -1)
    season = inputs["season"].numpy()[month]
    features = np.column_stack([environment, windows, season])
    if not np.isfinite(features).all():
        raise ValueError("nonfinite permitted covariates")
    return features


def make_estimator(family, recipe, seed, threads):
    settings = {key: value for key, value in recipe.items() if key != "target"}
    if family == "ridge":
        return Ridge(**settings)
    if family == "random_forest":
        return RandomForestRegressor(n_estimators=300, max_features=1., criterion="squared_error",
                                     random_state=seed, n_jobs=threads, **settings)
    if family == "boosted_trees":
        return HistGradientBoostingRegressor(max_iter=200, learning_rate=.05,
            min_samples_leaf=20, l2_regularization=1., early_stopping=False,
            random_state=seed, **settings)
    raise ValueError("undeclared estimator family")


def concentration_prediction(raw, transform):
    raw = np.asarray(raw, dtype=np.float64)
    if transform not in ("native", "log1p"):
        raise ValueError("undeclared target transform")
    if not np.isfinite(raw).all():
        raise ValueError("nonfinite estimator output")
    native = np.expm1(np.minimum(raw, MAX_LOG_DOC)) if transform == "log1p" else raw
    upper = float(np.expm1(MAX_LOG_DOC))
    return np.clip(native, 0., upper), {
        "negative_raw_native": int((native < 0).sum()),
        "upper_clipped": int((raw > MAX_LOG_DOC).sum()) if transform == "log1p"
                         else int((native > upper).sum()),
    }


def seasonal_medians(values, month_numbers):
    values, month_numbers = np.asarray(values), np.asarray(month_numbers)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("finite source DOC is required")
    fallback = float(np.median(values))
    return {str(month): float(np.median(values[month_numbers == month]))
            if np.any(month_numbers == month) else fallback for month in range(1, 13)}


def select_recipe(scores):
    """Native validation MAE, then log MAE, then lexical recipe ID on exact ties."""
    if not scores:
        raise ValueError("empty candidate scores")
    if any(not rows or not all(np.isfinite([row["mae"], row["log1p_mae"]]).all()
                              for row in rows) for rows in scores.values()):
        raise ValueError("finite candidate validation scores required")
    return min(scores, key=lambda name: (
        np.mean([row["mae"] for row in scores[name]]),
        np.mean([row["log1p_mae"] for row in scores[name]]), name))


def metrics(frame):
    y, p = frame.y_true.to_numpy(dtype=float), frame.y_pred.to_numpy(dtype=float)
    if not len(y) or not np.isfinite(y).all() or not np.isfinite(p).all() or np.any(y < 0) or np.any(p < 0):
        raise ValueError("complete finite nonnegative observed-query predictions required")
    error = p - y
    denominator = float(np.square(y - y.mean()).sum())
    tail = frame.high_doc.to_numpy(dtype=bool)
    errors = pd.DataFrame({"station": frame.station.to_numpy(), "absolute": np.abs(error)})
    return {"mae": float(np.abs(error).mean()),
        "log1p_mae": float(np.abs(np.log1p(p) - np.log1p(y)).mean()),
        "rmse": float(np.sqrt(np.square(error).mean())),
        "r2": float(1 - np.square(error).sum() / denominator) if denominator > 0 else None,
        "bias": float(error.mean()),
        "station_macro_mae": float(errors.groupby("station").absolute.mean().mean()),
        "source_q90_mae": float(np.abs(error[tail]).mean()) if tail.any() else None,
        "query_count": len(y), "tail_query_count": int(tail.sum()), "coverage": 1.}


def structure_metrics(frame, edges, months):
    """Only compare structures for which paired hidden real observations exist.

    These diagnostics do not establish accuracy on naturally unobserved cells.
    Seasonal means share exactly the same observed dates in truth and prediction.
    """
    seasonal, correlations, amplitudes, temporal = [], [], [], []
    for _, part in frame.groupby("station", sort=False):
        part = part.sort_values("cell")
        y, p = part.y_true.to_numpy(float), part.y_pred.to_numpy(float)
        if len(part) >= 12 and y.std() > 0:
            amplitudes.append(float(p.std() / y.std()))
            if p.std() > 0:
                correlations.append(float(np.corrcoef(y, p)[0, 1]))
        if len(part) >= 24:
            bins = part.assign(calendar_month=part.month.str[5:7]).groupby("calendar_month")[["y_true", "y_pred"]].mean()
            if len(bins) >= 6:
                y_cycle = bins.y_true.to_numpy()
                p_cycle = bins.y_pred.to_numpy()
                seasonal.append(float(np.abs((p_cycle - p_cycle.mean()) -
                                             (y_cycle - y_cycle.mean())).mean()))
        consecutive = np.diff(part.cell.to_numpy() % months) == 1
        temporal.extend(np.abs(np.diff(p)[consecutive] - np.diff(y)[consecutive]).tolist())
    lookup = frame.set_index("cell")
    spatial = []
    # Only direct edges with both endpoints in the held-out query at the same month.
    for a, b in np.unique(np.asarray(edges, dtype=np.int64), axis=1).T:
        if a == b:
            continue
        left = frame[frame.cell // months == a]
        right_cells = b * months + left.cell.to_numpy() % months
        exists = np.isin(right_cells, lookup.index.to_numpy())
        if exists.any():
            l = left.iloc[np.flatnonzero(exists)]
            r = lookup.loc[right_cells[exists]]
            spatial.extend(np.abs((r.y_pred.to_numpy() - l.y_pred.to_numpy()) -
                                  (r.y_true.to_numpy() - l.y_true.to_numpy())).tolist())
    mean = lambda values: float(np.mean(values)) if values else None
    return {"seasonal_centered_mae": mean(seasonal), "seasonal_stations": len(seasonal),
        "station_temporal_correlation": mean(correlations), "correlation_stations": len(correlations),
        "station_amplitude_ratio": mean(amplitudes), "amplitude_stations": len(amplitudes),
        "consecutive_month_delta_mae": mean(temporal), "consecutive_month_pairs": len(temporal),
        "same_month_edge_contrast_mae": mean(spatial), "same_month_edge_pairs": len(spatial)}


def paired_basin_interval(frame, comparator, endpoint, *, repetitions=5000):
    """Average seed errors first, then resample whole basin clusters."""
    if endpoint not in ("mae", "log1p_mae"):
        raise ValueError("only declared joint-primary absolute-error endpoints supported")
    parts = []
    for model in (comparator, "fixed_fusion"):
        part = frame[frame.model_name == model].copy()
        y, p = part.y_true.to_numpy(float), part.y_pred.to_numpy(float)
        part["error"] = np.abs(p - y) if endpoint == "mae" else np.abs(np.log1p(p) - np.log1p(y))
        cells = part.groupby(["target_huc4", "cell"], sort=True).error.mean()
        parts.append(cells.groupby(level=0).agg(["sum", "count"]))
    if not parts[0].index.equals(parts[1].index) or not np.array_equal(parts[0]["count"], parts[1]["count"]):
        raise ValueError("unpaired basin queries")
    base, ours = parts[0]["sum"].to_numpy(), parts[1]["sum"].to_numpy()
    counts = parts[0]["count"].to_numpy()
    rng = np.random.default_rng(20261011)
    draws = rng.integers(0, len(base), (repetitions, len(base)))
    differences = (ours[draws].sum(1) - base[draws].sum(1)) / counts[draws].sum(1)
    gains = np.divide(100 * (base[draws].sum(1) - ours[draws].sum(1)), base[draws].sum(1),
                      out=np.full(repetitions, np.nan), where=base[draws].sum(1) > 0)
    return {"comparator": comparator, "endpoint": endpoint, "basin_clusters": len(base),
        "difference_fusion_minus_comparator": float((ours.sum() - base.sum()) / counts.sum()),
        "difference_interval": np.quantile(differences, [.025, .975]).tolist(),
        "gain_percent": float(100 * (base.sum() - ours.sum()) / base.sum()) if base.sum() else None,
        "gain_percent_interval": np.nanquantile(gains, [.025, .975]).tolist() if np.isfinite(gains).any() else None,
        "regions_with_lower_error": int((ours < base).sum()),
        "exploratory": True, "multiplicity_adjusted": False}
