"""Same-calendar DOC comparisons between fixed real river-network forms."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GroupKFold

from river_graph.analysis.river_morphology_effect import BRANCHING, FOOTPRINT, PATHS

ENVIRONMENT = ("log_polygon_area", "wetland", "forest", "agriculture", "urban", "precipitation",
               "climate_temperature", "latitude", "longitude", "month_sin", "month_cos", "calendar_year")
HYDRO = ("log_discharge", "temperature")
ARMS = {"environment": (), "hydro": HYDRO, "form": (*HYDRO, "form_2", "form_3"),
        "branching": (*HYDRO, *BRANCHING), "paths": (*HYDRO, *PATHS),
        "all_morphology": (*HYDRO, *FOOTPRINT, *BRANCHING, *PATHS)}
COMPARISONS = (("hydro", "environment"), ("form", "hydro"), ("branching", "hydro"),
               ("paths", "hydro"), ("all_morphology", "hydro"))
THRESHOLD = 10.
RESPONSE_NAMES = ("doc_mean", "doc_median", "doc_sd", "doc_cv", "doc_iqr", "log_doc_mean", "log_doc_sd",
                  "high_doc_fraction", "environment_residual_mean", "environment_residual_sd",
                  "hydro_residual_mean", "hydro_residual_sd", "environment_exceedance", "hydro_exceedance")
POPULATIONS = ("common_doc", "complete_hydro", "at_least_24_months")


def station_weights(frame):
    return 1/frame.groupby("station").month_index.transform("size").to_numpy(float)


def raw_features(frame, arm, region_levels):
    columns = [*ENVIRONMENT, *ARMS[arm]]
    x = frame[columns].to_numpy(float)
    if region_levels:
        x = np.column_stack([x, *[(frame.huc2 == level).to_numpy(float) for level in region_levels]])
        columns += ["huc2_"+level for level in region_levels]
    return x, columns


def fit_background(frame, arm, *, probability=False):
    levels = sorted(str(h) for h in frame.huc2.unique())[1:]
    x, columns = raw_features(frame, arm, levels)
    median = np.array([np.median(c[np.isfinite(c)]) if np.isfinite(c).any() else 0. for c in x.T])
    missing = ~np.isfinite(x)
    x = np.column_stack([np.where(missing, median, x), missing.astype(float)])
    weight = station_weights(frame)
    mean = np.average(x, axis=0, weights=weight)
    sd = np.maximum(np.sqrt(np.average((x-mean)**2, axis=0, weights=weight)), 1e-8)
    truth = frame.y_true.to_numpy(float)
    if not np.isfinite(truth).all() or (truth < 0).any():
        raise ValueError("finite nonnegative fitting DOC required")
    target = (truth >= THRESHOLD).astype(float) if probability else np.log1p(truth)
    scaled = (x-mean)/sd
    if probability:
        if len(np.unique(target)) == 1:
            coefficient, intercept = np.zeros(x.shape[1]), 0.
            constant = float(target[0])
        else:
            model = LogisticRegression(C=1., max_iter=1000, random_state=42).fit(scaled, target, sample_weight=weight)
            coefficient, intercept = model.coef_[0], float(model.intercept_[0])
            constant = None
    else:
        model = Ridge(alpha=10.).fit(scaled, target, sample_weight=weight)
        coefficient, intercept, constant = model.coef_, float(model.intercept_), None
    return {"arm": arm, "probability": probability, "region_levels": levels, "median": median.tolist(),
            "mean": mean.tolist(), "sd": sd.tolist(), "coef": coefficient.tolist(), "intercept": intercept,
            "constant_probability": constant, "columns": columns+[c+"_missing" for c in columns]}


def background_prediction(frame, state):
    x, _ = raw_features(frame, state["arm"], state["region_levels"])
    missing = ~np.isfinite(x)
    x = np.column_stack([np.where(missing, state["median"], x), missing.astype(float)])
    linear = ((x-np.asarray(state["mean"]))/np.asarray(state["sd"]))@np.asarray(state["coef"])+state["intercept"]
    if state["probability"]:
        return np.full(len(frame), state["constant_probability"]) if state["constant_probability"] is not None else expit(linear)
    return np.maximum(0., linear)


def crossfit_backgrounds(frame):
    if frame.duplicated(["station", "month_index"]).any():
        raise ValueError("one observation per station/calendar month required")
    stations = frame[["station", "huc4"]].drop_duplicates().sort_values("station").reset_index(drop=True)
    if stations.station.duplicated().any() or stations.huc4.nunique() < 5:
        raise ValueError("unique station region and five source regions required")
    fold_map = {}
    for fold, (_, test) in enumerate(GroupKFold(5).split(stations, groups=stations.huc4)):
        fold_map.update(dict.fromkeys(stations.station.iloc[test], fold))
    folds = frame.station.map(fold_map)
    predicted, states = [], []
    for fold in range(5):
        train, query = frame[folds.ne(fold)], frame[folds.eq(fold)]
        assert set(train.huc4).isdisjoint(query.huc4)
        for arm in ARMS:
            state = fit_background(train, arm)
            estimate = background_prediction(query, state)
            state.update(fold=fold, training_huc4=sorted(train.huc4.unique()), held_huc4=sorted(query.huc4.unique()))
            states.append(state)
            f = query[["station", "month_index", "huc4", "cluster", "y_true"]].copy()
            f["fold"], f["arm"], f["log_pred"] = fold, arm, estimate
            f["y_pred"] = np.expm1(estimate)
            f["high_probability"] = np.nan
            if arm in ("environment", "hydro"):
                probability_state = fit_background(train, arm, probability=True)
                f["high_probability"] = background_prediction(query, probability_state)
                probability_state.update(fold=fold, training_huc4=sorted(train.huc4.unique()), held_huc4=sorted(query.huc4.unique()))
                states.append(probability_state)
            predicted.append(f)
    output = pd.concat(predicted, ignore_index=True)
    if not np.isfinite(output[["y_pred", "log_pred"]]).all().all():
        raise ValueError("finite held-region predictions required")
    return output, states


def paired_records(frame, pairs, predictions):
    look = frame.set_index(["station", "month_index"])
    rows, ledger = [], []
    for r in pairs.itertuples():
        a = frame[frame.station.eq(r.station_a)]
        b = frame[frame.station.eq(r.station_b)]
        shared = np.intersect1d(a.month_index, b.month_index)
        pair_id = f"{r.class_a}_{r.class_b}_{r.station_a}_{r.station_b}"
        months = a.set_index("month_index").loc[shared, "calendar_month"]
        if len(shared):
            x, y = look.loc[(r.station_a, shared), :], look.loc[(r.station_b, shared), :]
            x, y = x.reset_index(), y.reset_index()
            complete = (x.discharge_valid & y.discharge_valid & x.temperature_valid & y.temperature_valid).to_numpy(bool)
        else:
            complete = np.zeros(0, bool)
        selected = len(shared) >= 12 and months.nunique() >= 6
        hydro_selected = complete.sum() >= 12 and months.iloc[np.flatnonzero(complete)].nunique() >= 6
        ledger.append({"pair_id": pair_id, "station_a": r.station_a, "station_b": r.station_b,
                       "class_a": r.class_a, "class_b": r.class_b, "huc4": r.huc4,
                       "n_common_months": len(shared), "n_months_of_year": months.nunique(),
                       "n_complete_hydro_months": int(complete.sum()), "included": selected,
                       "complete_hydro_included": hydro_selected,
                       "longer_record_included": len(shared) >= 24 and months.nunique() >= 6})
        if not selected:
            continue
        f = pd.DataFrame({"pair_id": pair_id, "station_a": r.station_a, "station_b": r.station_b,
                          "class_a": r.class_a, "class_b": r.class_b, "huc4": r.huc4,
                          "month_index": shared, "date": x.date.to_numpy(), "complete_hydro": complete})
        for suffix, member in (("a", x), ("b", y)):
            for name in ("y_true", "log_discharge", "temperature", "discharge_valid", "temperature_valid"):
                f[f"{name}_{suffix}"] = member[name].to_numpy()
            for arm in ("environment", "hydro"):
                p = predictions[predictions.arm.eq(arm)].set_index(["station", "month_index"])
                v = p.loc[(getattr(r, "station_"+suffix), shared), :]
                f[f"{arm}_log_prediction_{suffix}"] = v.log_pred.to_numpy()
                f[f"{arm}_high_probability_{suffix}"] = v.high_probability.to_numpy()
        rows.append(f)
    return pd.concat(rows, ignore_index=True), pd.DataFrame(ledger)


def member_responses(frame, suffix):
    y = frame[f"y_true_{suffix}"].to_numpy(float)
    log = np.log1p(y)
    mean, sd = y.mean(), y.std(ddof=1)
    high = (y >= THRESHOLD).astype(float)
    out = {"doc_mean": mean, "doc_median": np.median(y), "doc_sd": sd,
           "doc_cv": sd/mean if mean > 0 else np.nan, "doc_iqr": np.quantile(y, .75)-np.quantile(y, .25),
           "log_doc_mean": log.mean(), "log_doc_sd": log.std(ddof=1), "high_doc_fraction": high.mean()}
    for arm in ("environment", "hydro"):
        residual = log-frame[f"{arm}_log_prediction_{suffix}"].to_numpy()
        out[f"{arm}_residual_mean"] = residual.mean()
        out[f"{arm}_residual_sd"] = residual.std(ddof=1)
        out[f"{arm}_exceedance"] = np.mean(high-frame[f"{arm}_high_probability_{suffix}"])
    return out, int(high.sum())


def response_evidence(records):
    rows = []
    for pair, f in records.groupby("pair_id"):
        r = f.iloc[0]
        for population in POPULATIONS:
            s = f[f.complete_hydro] if population == "complete_hydro" else f
            if population == "at_least_24_months" and len(s) < 24:
                continue
            if len(s) < 12 or pd.DatetimeIndex(s.date).month.nunique() < 6:
                continue
            ra, high_a = member_responses(s, "a")
            rb, high_b = member_responses(s, "b")
            for metric in RESPONSE_NAMES:
                rows.append({"pair_id": pair, "station_a": r.station_a, "station_b": r.station_b,
                             "class_a": r.class_a, "class_b": r.class_b, "huc4": r.huc4,
                             "population": population, "metric": metric, "value_a": ra[metric], "value_b": rb[metric],
                             "difference_b_minus_a": rb[metric]-ra[metric], "n_months": len(s),
                             "n_high_a": high_a, "n_high_b": high_b, "n_months_of_year": pd.DatetimeIndex(s.date).month.nunique()})
    return pd.DataFrame(rows)


def response_summary(evidence, draws=5000):
    rows, sensitivity = [], []
    for keys, f in evidence.groupby(["class_a", "class_b", "population", "metric"]):
        f = f[np.isfinite(f.difference_b_minus_a)]
        if f.empty:
            continue
        difference = f.difference_b_minus_a.to_numpy()
        _, gi = np.unique(f.huc4, return_inverse=True)
        n = int(gi.max()+1)
        w = np.random.default_rng(42).multinomial(n, np.full(n, 1/n), size=draws)[:, gi]
        boot = w@difference/w.sum(axis=1)
        lo, hi = np.quantile(boot, [.025, .975]) if n > 1 else (np.nan, np.nan)
        context = dict(zip(("class_a", "class_b", "population", "metric"), keys, strict=True))
        rows.append({**context, "difference_b_minus_a": difference.mean(), "ci_low": lo, "ci_high": hi,
                     "mean_a": f.value_a.mean(), "mean_b": f.value_b.mean(), "n_pairs": len(f), "n_huc4": n,
                     "interval_status": "estimable" if n > 1 else "not_estimable_single_HUC4",
                     "positive_pairs": int((difference > 0).sum())})
        if n > 1:
            for excluded in sorted(f.huc4.unique()):
                s = f[f.huc4.ne(excluded)]
                sensitivity.append({**context, "excluded_huc4": excluded, "n_pairs": len(s),
                                    "difference_b_minus_a": s.difference_b_minus_a.mean()})
    return pd.DataFrame(rows), pd.DataFrame(sensitivity)


def information_scores(predictions, draws=5000):
    f = predictions.copy()
    f["mae"] = abs(f.y_pred-f.y_true)
    f["log_mae"] = abs(f.log_pred-np.log1p(f.y_true))
    station = f.groupby(["station", "huc4", "fold", "cluster", "arm"], as_index=False)[["mae", "log_mae"]].mean()
    scores, gains = [], []
    for metric in ("mae", "log_mae"):
        table = station.pivot(index=["station", "huc4", "fold"], columns="arm", values=metric).reset_index()
        _, gi = np.unique(table.huc4, return_inverse=True)
        n = int(gi.max()+1)
        w = np.random.default_rng(42).multinomial(n, np.full(n, 1/n), size=draws)[:, gi]
        for arm in ARMS:
            boot = w@table[arm].to_numpy()/w.sum(axis=1)
            lo, hi = np.quantile(boot, [.025, .975])
            scores.append({"arm": arm, "metric": metric, "mean": table[arm].mean(), "ci_low": lo,
                           "ci_high": hi, "n_stations": len(table), "n_huc4": n})
        for candidate, reference in COMPARISONS:
            c, b = table[candidate].to_numpy(), table[reference].to_numpy()
            reduction = w@(b-c)/w.sum(axis=1)
            pct = 100*(1-(w@c)/(w@b))
            lo, hi = np.quantile(reduction, [.025, .975])
            plo, phi = np.quantile(pct, [.025, .975])
            gains.append({"candidate": candidate, "reference": reference, "metric": metric,
                          "error_reduction": np.mean(b-c), "ci_low": lo, "ci_high": hi,
                          "relative_reduction_pct": 100*(1-c.mean()/b.mean()), "gain_ci_low_pct": plo,
                          "gain_ci_high_pct": phi, "n_stations": len(table), "n_huc4": n,
                          "positive_folds": int((table.groupby("fold")[candidate].mean() < table.groupby("fold")[reference].mean()).sum())})
    return station, pd.DataFrame(scores), pd.DataFrame(gains)
