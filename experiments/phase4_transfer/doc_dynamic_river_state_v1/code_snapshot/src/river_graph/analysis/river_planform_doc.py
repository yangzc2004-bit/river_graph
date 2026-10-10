"""Whole-network morphology associations, keeping station and region units."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from river_graph.analysis.river_doc_structure import bootstrap_ols

SHAPE = ("log_basin_aspect", "log_network_axis_ratio", "log_drainage_density",
         "mainstem_share", "mainstem_sinuosity")
HYDRO = ("log_flow", "median_temperature", "flow_available", "temperature_available")
CONTROLS = ("log_polygon_area", "wetland", "forest", "agriculture", "urban",
            "precipitation", "climate_temperature", *HYDRO, "median_year", "record_span",
            "n_calendar_months", "log_n_doc", "latitude", "longitude")
OUTCOMES = {"doc_median": "log1p", "doc_cv": "log1p", "harmonic_amplitude": "log1p",
            "q90_fraction": "raw", "cq_slope": "raw"}


def shape_features(frame):
    out = frame[["mainstem_share", "mainstem_sinuosity"]].copy()
    for name in ("basin_aspect", "network_axis_ratio", "drainage_density"):
        if not (frame[name] > 0).all():
            raise ValueError(f"positive morphology required: {name}")
        out["log_"+name] = np.log(frame[name])
    return out[list(SHAPE)]


def group_summary(panel, draws=5000):
    rows = []
    for group, sub in panel[panel.eligible].groupby("cluster"):
        for metric in (*OUTCOMES, "doc_iqr"):
            s = sub.dropna(subset=[metric])
            if s.empty:
                continue
            a = s[metric].to_numpy(float)
            rng = np.random.default_rng(42)
            boot = np.median(a[rng.integers(len(a), size=(draws, len(a)))], axis=1)
            rows.append({"cluster": int(group), "metric": metric, "n_stations": len(a),
                         "n_huc4": s.huc4.nunique(), "median": np.median(a),
                         "q25": np.quantile(a, .25), "q75": np.quantile(a, .75),
                         "ci_low": np.quantile(boot, .025), "ci_high": np.quantile(boot, .975),
                         "few_stations": len(a) < 10})
    return pd.DataFrame(rows)


def adjusted_associations(panel, draws=5000):
    results, diagnostics = [], []
    for metric, transform in OUTCOMES.items():
        s = panel[panel.eligible & panel[metric].notna()].copy()
        y = s[metric].to_numpy(float)
        if transform == "log1p":
            y = np.log1p(y)
        for mode, hydro in (("class", True), ("class", False), ("continuous", True)):
            control = [c for c in CONTROLS if hydro or c not in HYDRO]
            numeric = control + (list(SHAPE) if mode == "continuous" else [])
            imputer = SimpleImputer(strategy="median", add_indicator=True)
            raw = imputer.fit_transform(s[numeric])
            names = list(imputer.get_feature_names_out(numeric))
            variable = raw.std(axis=0) > 1e-9
            x = StandardScaler().fit_transform(raw[:, variable])
            labels = ["intercept", *[n for n, keep in zip(names, variable, strict=True) if keep]]
            region = pd.get_dummies(s.huc2, prefix="huc2", drop_first=True, dtype=float)
            x = np.column_stack([np.ones(len(s)), x, region.to_numpy()])
            labels.extend(region.columns)
            if mode == "class":
                cat = pd.get_dummies(s.cluster.astype(int), prefix="class", drop_first=True, dtype=float)
                x = np.column_stack([x, cat.to_numpy()])
                labels.extend(cat.columns)
            point, boot, diag = bootstrap_ols(x, y, draws=draws)
            _, regional, _ = bootstrap_ols(x, y, draws=draws, groups=s.huc4)
            model = mode+("_hydro_adjusted" if hydro else "_without_local_hydro")
            diagnostics.append({"metric": metric, "model": model, "transform": transform,
                                **diag, "n_huc4": s.huc4.nunique(),
                                "dropped_terms": [labels[i] for i in diag["dropped_columns"]]})
            for i, label in enumerate(labels):
                if i in diag["dropped_columns"] or not (label.startswith("class_") or label in SHAPE):
                    continue
                lo, hi = np.quantile(boot[:, i], [.025, .975])
                rlo, rhi = np.quantile(regional[:, i], [.025, .975])
                results.append({"metric": metric, "model": model, "term": label,
                                "estimate": point[i], "ci_low": lo, "ci_high": hi,
                                "huc4_ci_low": rlo, "huc4_ci_high": rhi, "transform": transform,
                                "n_stations": len(s), "n_huc4": s.huc4.nunique()})
    return pd.DataFrame(results), pd.DataFrame(diagnostics)


def blocked_predictions(panel):
    s = panel[panel.eligible].copy().reset_index(drop=True)
    s["river_class"] = s.cluster.astype(int).astype(str)
    y = np.log1p(s.doc_median.to_numpy())
    rows = []
    for fold, (train, val) in enumerate(GroupKFold(n_splits=5).split(s, groups=s.huc4)):
        for arm in ("environment_area", "environment_area_classes", "environment_area_shape"):
            numeric = [*CONTROLS, *(SHAPE if arm.endswith("shape") else [])]
            cat = ["huc2", *(["river_class"] if arm.endswith("classes") else [])]
            prep = ColumnTransformer([
                ("num", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()), numeric),
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat)])
            model = make_pipeline(prep, Ridge(alpha=10))
            model.fit(s.iloc[train], y[train])
            prediction = model.predict(s.iloc[val])
            for i, p in zip(val, prediction, strict=True):
                rows.append({"station": s.station.iloc[i], "huc4": s.huc4.iloc[i], "fold": fold,
                             "model": arm, "y_log1p": y[i], "y_pred_log1p": p,
                             "y_true": s.doc_median.iloc[i], "y_pred": max(0, np.expm1(p))})
    return pd.DataFrame(rows)


def blocked_gain(predictions, draws=5000):
    """Equal-fold paired prediction loss, with joint station/region resampling."""
    rows = []
    for space in ("native", "log1p"):
        truth, pred = ("y_true", "y_pred") if space == "native" else ("y_log1p", "y_pred_log1p")
        frame = predictions.copy()
        frame["loss"] = abs(frame[truth]-frame[pred])
        table = frame.pivot(index=["station", "huc4", "fold"], columns="model", values="loss").reset_index()
        base = table.environment_area.to_numpy()
        for arm in ("environment_area_classes", "environment_area_shape"):
            candidate = table[arm].to_numpy()
            point_base = table.groupby("fold").environment_area.mean().mean()
            point_candidate = table.groupby("fold")[arm].mean().mean()
            for unit in ("station", "huc4"):
                _, gi = np.unique(table[unit], return_inverse=True)
                n = gi.max()+1
                rng = np.random.default_rng(42)
                gains = []
                folds = [table.fold.to_numpy() == f for f in sorted(table.fold.unique())]
                attempts = 0
                while len(gains) < draws:
                    attempts += 1
                    if attempts > draws*100:
                        raise ValueError("cannot resample all folds")
                    w = rng.multinomial(n, np.full(n, 1/n))[gi]
                    if any(w[f].sum() == 0 for f in folds):
                        continue
                    b = np.mean([np.average(base[f], weights=w[f]) for f in folds])
                    c = np.mean([np.average(candidate[f], weights=w[f]) for f in folds])
                    gains.append(100*(b-c)/b)
                lo, hi = np.quantile(gains, [.025, .975])
                rows.append({"model": arm, "space": space, "resampling_unit": unit,
                             "reference_mae": point_base, "candidate_mae": point_candidate,
                             "relative_gain_pct": 100*(point_base-point_candidate)/point_base,
                             "gain_ci_low_pct": lo, "gain_ci_high_pct": hi,
                             "n_stations": len(table), "n_huc4": table.huc4.nunique(),
                             "positive_folds": sum(candidate[f].mean() < base[f].mean() for f in folds)})
    return pd.DataFrame(rows)
