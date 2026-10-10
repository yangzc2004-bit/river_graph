"""Cross-validated information in the arrival of observed tributary DOC."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

from river_graph.analysis.river_mechanisms import correlation

OPERATORS = ("background", "same_month", "uniform_history", "mean_delay", "branch_arrival")
FRACTIONS = (0., .10, .25, .50, 1.)
BACKGROUND = ("month_sin", "month_cos", "year_fraction", "log_basin_area", "log_discharge", "temperature")


def concentration_proxy(frame, operator, fraction, max_path):
    """Known source values, same source budget, causal current/previous bins."""
    if operator not in OPERATORS or not np.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError("known operator and lag fraction in [0,1] required")
    if max_path <= 0 or not np.isfinite(max_path):
        raise ValueError("positive common geometric path scale required")
    if operator == "background":
        return None
    a, b, ap, bp, w = (frame[c].to_numpy(float) for c in (
        "doc_a_now", "doc_b_now", "doc_a_previous", "doc_b_previous", "weight_a"))
    if not np.isfinite(np.column_stack([a, b, ap, bp, w])).all() or min(a.min(), b.min(), ap.min(), bp.min()) < 0:
        raise ValueError("finite nonnegative known source DOC required")
    if ((w <= 0) | (w >= 1)).any():
        raise ValueError("positive branch flow shares required")
    now, previous = w*a+(1-w)*b, w*ap+(1-w)*bp
    if operator == "same_month":
        return now
    if operator == "uniform_history":
        return (1-fraction)*now+fraction*previous
    la, lb = (frame[c].to_numpy(float) for c in ("path_a_km", "path_b_km"))
    if not np.isfinite(np.column_stack([la, lb])).all() or min(la.min(), lb.min()) < 0 or max(la.max(), lb.max()) > max_path+1e-9:
        raise ValueError("causal paths must fit inside the common geometric scale")
    da, db = fraction*la/max_path, fraction*lb/max_path
    if operator == "mean_delay":
        mean = w*da+(1-w)*db
        return (1-mean)*now+mean*previous
    return w*((1-da)*a+da*ap)+(1-w)*((1-db)*b+db*bp)


def receiver_weights(frame):
    counts = frame.groupby("pair_id").month_index.transform("size").to_numpy(float)
    pairs = frame.groupby("target").pair_id.transform("nunique").to_numpy(float)
    return 1/counts/pairs


def feature_matrix(frame, proxy):
    columns = list(BACKGROUND)
    x = frame[columns].to_numpy(float)
    if proxy is not None:
        x = np.column_stack([x, np.log1p(proxy)])
        columns.append("log_upstream_proxy")
    return x, columns


def fit_calibrator(frame, proxy):
    x, columns = feature_matrix(frame, proxy)
    median = np.array([np.median(c[np.isfinite(c)]) if np.isfinite(c).any() else 0. for c in x.T])
    missing = ~np.isfinite(x)
    x = np.column_stack([np.where(missing, median, x), missing.astype(float)])
    weights = receiver_weights(frame)
    mean = np.average(x, axis=0, weights=weights)
    sd = np.sqrt(np.average((x-mean)**2, axis=0, weights=weights))
    sd = np.maximum(sd, 1e-8)
    truth = frame.y_true.to_numpy(float)
    if not np.isfinite(truth).all() or (truth < 0).any():
        raise ValueError("finite nonnegative fitting DOC required")
    model = Ridge(alpha=1.).fit((x-mean)/sd, np.log1p(truth), sample_weight=weights)
    return {"median": median.tolist(), "mean": mean.tolist(), "sd": sd.tolist(),
            "coef": model.coef_.tolist(), "intercept": float(model.intercept_),
            "feature_names": columns+[c+"_missing" for c in columns]}


def calibrated_prediction(frame, proxy, state):
    x, _ = feature_matrix(frame, proxy)
    missing = ~np.isfinite(x)
    x = np.column_stack([np.where(missing, state["median"], x), missing.astype(float)])
    z = (x-np.asarray(state["mean"]))/np.asarray(state["sd"])
    log_prediction = np.maximum(0., z@np.asarray(state["coef"])+state["intercept"])
    return np.expm1(log_prediction)


def receiver_log_loss(frame, prediction):
    errors = frame[["target", "pair_id"]].copy()
    errors["error"] = abs(np.log1p(frame.y_true.to_numpy())-np.log1p(prediction))
    return errors.groupby(["target", "pair_id"]).error.mean().groupby("target").mean().mean()


def choose_fraction(train, operator, max_path):
    candidates = (0.,) if operator in ("background", "same_month") else FRACTIONS
    groups = train.component.to_numpy()
    n_groups = len(np.unique(groups))
    if n_groups < 3:
        raise ValueError("at least three training monitoring systems required")
    split = list(GroupKFold(n_splits=min(3, n_groups)).split(train, groups=groups))
    trials = []
    for fraction in candidates:
        inner = []
        for fit_indices, val_indices in split:
            fit, val = train.iloc[fit_indices], train.iloc[val_indices]
            state = fit_calibrator(fit, concentration_proxy(fit, operator, fraction, max_path))
            pred = calibrated_prediction(val, concentration_proxy(val, operator, fraction, max_path), state)
            v = val[["target", "pair_id", "y_true"]].copy()
            v["prediction"] = pred
            inner.append(v)
        pooled = pd.concat(inner, ignore_index=True)
        # Dates never cross into another row; each inner validation row appears once.
        loss = receiver_log_loss(pooled, pooled.prediction.to_numpy())
        trials.append({"operator": operator, "fraction": fraction, "inner_log_mae": loss})
    chosen = min(trials, key=lambda r: (r["inner_log_mae"], r["fraction"]))["fraction"]
    return chosen, trials


def nested_predictions(frame, max_path):
    """Leave one complete shared monitoring system out; nested source selection."""
    if frame.duplicated(["pair_id", "month_index"]).any():
        raise ValueError("one record per connection/calendar month required")
    predictions, trial_rows, states = [], [], []
    for held in sorted(frame.component.unique()):
        train, test = frame[frame.component.ne(held)], frame[frame.component.eq(held)]
        train_values = train.drop_duplicates(["target", "month_index"]).y_true
        q90 = float(train_values.quantile(.9))
        for operator in OPERATORS:
            fraction, trials = choose_fraction(train, operator, max_path)
            for trial in trials:
                trial_rows.append({"held_component": int(held), **trial,
                                   "selected": trial["fraction"] == fraction,
                                   "n_training_receivers": train.target.nunique()})
            state = fit_calibrator(train, concentration_proxy(train, operator, fraction, max_path))
            pred = calibrated_prediction(test, concentration_proxy(test, operator, fraction, max_path), state)
            if not np.isfinite(pred).all():
                raise ValueError("nonfinite held-out prediction")
            output = test.copy()
            output["y_pred"] = pred
            output["operator"] = operator
            output["fraction"] = fraction
            output["q90_threshold"] = q90
            output["high_doc"] = output.y_true >= q90
            output["visibility_role"] = "held_receiver_score_known_upstream_input"
            predictions.append(output)
            states.append({"held_component": int(held), "operator": operator, "fraction": fraction,
                           "training_components": sorted(int(g) for g in train.component.unique()),
                           "q90_threshold": q90, **state})
    return pd.concat(predictions, ignore_index=True), pd.DataFrame(trial_rows), states


def connection_metrics(predictions):
    rows = []
    for (pair, operator), f in predictions.groupby(["pair_id", "operator"]):
        r = f.iloc[0]
        error = f.y_pred.to_numpy()-f.y_true.to_numpy()
        tail = f.high_doc.to_numpy(bool)
        rows.append({"pair_id": pair, "operator": operator, "target": r.target, "huc4": r.huc4,
                     "component": r.component, "cluster": r.cluster, "fraction": r.fraction,
                     "mae": np.mean(abs(error)), "log_mae": np.mean(abs(np.log1p(f.y_pred)-np.log1p(f.y_true))),
                     "mse": np.mean(error**2), "rmse": np.sqrt(np.mean(error**2)), "bias": error.mean(),
                     "q90_mae": np.mean(abs(error[tail])) if tail.any() else np.nan,
                     "n_tail": int(tail.sum()), "tail_unstable": tail.sum() < 20,
                     "signal_rho": correlation(f.y_pred, f.y_true), "n_months": len(f),
                     "path_difference_scaled": r.path_difference_scaled,
                     "branch_balance": 4*r.weight_a*(1-r.weight_a),
                     "source_drainage_coverage": r.source_drainage_coverage})
    return pd.DataFrame(rows)
