"""Observed within-river flow responses for fixed morphology classes."""
from __future__ import annotations

import numpy as np
import pandas as pd

from river_graph.analysis.river_morphology_effect import BRANCHING, FOOTPRINT, PATHS

HIGH_DOC = 10.
CONTROLS = ("log_polygon_area", "wetland", "forest", "agriculture", "urban", "precipitation",
            "climate_temperature", "latitude", "longitude", "log_flow_contrast")
BLOCKS = {"form": ("form_2", "form_3"), "footprint": FOOTPRINT, "branching": BRANCHING, "paths": PATHS}
METRICS = ("raw_doc_contrast", "raw_log_contrast", "high_frequency_contrast", "adjusted_log_contrast",
           "adjusted_doc_contrast", "log_flow_contrast")


def flow_references(flow):
    """A hydro-only calendar panel; DOC columns cannot affect thresholds."""
    if flow.duplicated(["station", "month_index"]).any():
        raise ValueError("unique station/month flow reference required")
    rows = []
    for station, s in flow.groupby("station", sort=True):
        q = s.log_discharge.to_numpy(float)
        q = q[np.isfinite(q) & (q > 0)]
        lo, hi = np.quantile(q, [1/3, 2/3]) if len(q) else (np.nan, np.nan)
        rows.append({"station": station, "n_reference_months": len(q), "low_threshold": lo,
                     "high_threshold": hi, "reference_valid": len(q) >= 24 and hi-lo > 1e-8,
                     "first_reference_month": s.month_index.min(), "last_reference_month": s.month_index.max()})
    return pd.DataFrame(rows)


def assign_states(frame, references):
    f = frame.merge(references, on="station", how="left", validate="many_to_one")
    low = f.log_discharge.le(f.low_threshold)
    high = f.log_discharge.ge(f.high_threshold)
    valid = f.reference_valid.fillna(False) & np.isfinite(f.log_discharge) & f.log_discharge.gt(0)
    f["flow_state"] = np.where(valid, np.where(low, "low", np.where(high, "high", "middle")), "unavailable")
    return f


def response_eligibility(frame):
    return {"n_months": len(frame), "n_low": int(frame.flow_state.eq("low").sum()),
            "n_high": int(frame.flow_state.eq("high").sum()),
            "n_calendar_months": frame.calendar_month.nunique(),
            "n_years": np.floor(frame.calendar_year).nunique(),
            "eligible": len(frame) >= 24 and frame.flow_state.eq("low").sum() >= 6
            and frame.flow_state.eq("high").sum() >= 6 and frame.calendar_month.nunique() >= 6
            and np.floor(frame.calendar_year).nunique() >= 2}


def measured_response(frame, *, temperature=False):
    """High-minus-low effects on a fixed observed station calendar."""
    if not response_eligibility(frame)["eligible"]:
        raise ValueError("insufficient observed low/high response record")
    if temperature and not np.isfinite(frame.temperature).all():
        raise ValueError("temperature sensitivity requires observed temperature")
    low, high = frame[frame.flow_state.eq("low")], frame[frame.flow_state.eq("high")]
    out = {"raw_doc_contrast": high.y_true.mean()-low.y_true.mean(),
           "raw_log_contrast": np.log1p(high.y_true).mean()-np.log1p(low.y_true).mean(),
           "high_frequency_contrast": high.y_true.ge(HIGH_DOC).mean()-low.y_true.ge(HIGH_DOC).mean(),
           "log_flow_contrast": high.log_discharge.mean()-low.log_discharge.mean()}
    columns = ["intercept", "middle_state", "high_state", "month_sin", "month_cos", "year_centered"]
    x = np.column_stack([np.ones(len(frame)), frame.flow_state.eq("middle"), frame.flow_state.eq("high"),
                         frame.month_sin, frame.month_cos, frame.calendar_year-frame.calendar_year.mean()])
    if temperature:
        columns.append("temperature_centered")
        x = np.column_stack([x, frame.temperature-frame.temperature.mean()])
    rank = np.linalg.matrix_rank(x)
    out.update(fit_rank=int(rank), fit_columns=len(columns), residual_df=len(frame)-int(rank),
               adjusted_identified=bool(rank == len(columns)), adjusted_log_contrast=np.nan,
               adjusted_doc_contrast=np.nan, fit_coefficients={})
    for name, y in (("log", np.log1p(frame.y_true)), ("doc", frame.y_true)):
        coef = np.linalg.lstsq(x, y, rcond=None)[0]
        out["fit_coefficients"][name] = dict(zip(columns, coef.tolist(), strict=True))
        if rank == len(columns):
            out[f"adjusted_{name}_contrast"] = float(coef[2])
    states = []
    for state in ("low", "middle", "high"):
        s = frame[frame.flow_state.eq(state)]
        states.append({"flow_state": state, "n_months": len(s), "n_high_doc": int(s.y_true.ge(HIGH_DOC).sum()),
                       "doc_mean": s.y_true.mean(), "log_doc_mean": np.log1p(s.y_true).mean(),
                       "doc_sd": s.y_true.std(ddof=1), "high_frequency": s.y_true.ge(HIGH_DOC).mean(),
                       "mean_log_discharge": s.log_discharge.mean(), "mean_temperature": s.temperature.mean(),
                       "month_sin_mean": s.month_sin.mean(), "month_cos_mean": s.month_cos.mean(),
                       "calendar_year_mean": s.calendar_year.mean()})
    return out, states


def station_responses(frame):
    ledger, fits, states, coefficients = [], [], [], []
    for station, g in frame.groupby("station", sort=True):
        r = g.iloc[0]
        context = {"station": station, "comid": int(r.comid), "cluster": int(r.cluster), "huc4": r.huc4}
        for population in ("observed_flow", "observed_temperature"):
            s = g[g.flow_state.ne("unavailable")]
            if population == "observed_temperature":
                s = s[s.temperature_valid]
            status = response_eligibility(s)
            ledger.append({**context, "population": population, **status})
            if not status["eligible"]:
                continue
            response, summaries = measured_response(s, temperature=population == "observed_temperature")
            coef = response.pop("fit_coefficients")
            fits.append({**context, "population": population, **status, **response})
            states.extend({**context, "population": population, **summary} for summary in summaries)
            coefficients.append({**context, "population": population, "coefficients": coef})
    return pd.DataFrame(fits), pd.DataFrame(states), pd.DataFrame(ledger), coefficients


def shared_pair_records(frame, pairs):
    columns = ["station", "month_index", "date", "calendar_month", "calendar_year", "month_sin", "month_cos",
               "y_true", "log_discharge", "temperature", "temperature_valid", "flow_state"]
    records = []
    for r in pairs.itertuples():
        a = frame[frame.station.eq(r.station_a)][columns]
        b = frame[frame.station.eq(r.station_b)][columns]
        s = a.merge(b, on=["month_index", "date", "calendar_month", "calendar_year", "month_sin", "month_cos"],
                    suffixes=("_a", "_b"), validate="one_to_one")
        s = s[s.flow_state_a.ne("unavailable") & s.flow_state_b.ne("unavailable")].copy()
        s["pair_id"] = f"{r.class_a}_{r.class_b}_{r.station_a}_{r.station_b}"
        s["class_a"], s["class_b"], s["huc4"] = r.class_a, r.class_b, r.huc4
        records.append(s)
    return pd.concat(records, ignore_index=True)


def pair_responses(records, pairs):
    evidence, ledger, states, coefficients, joint = [], [], [], [], []
    for r in pairs.itertuples():
        pair = f"{r.class_a}_{r.class_b}_{r.station_a}_{r.station_b}"
        context = {"pair_id": pair, "station_a": r.station_a, "station_b": r.station_b,
                   "class_a": r.class_a, "class_b": r.class_b, "huc4": r.huc4}
        g = records[records.pair_id.eq(pair)]
        for population in ("observed_flow", "observed_temperature"):
            s = g if population == "observed_flow" else g[g.temperature_valid_a & g.temperature_valid_b]
            members, status = {}, {}
            for suffix in ("a", "b"):
                member = s.rename(columns={name+"_"+suffix: name for name in
                    ("y_true", "log_discharge", "temperature", "flow_state")})
                status[suffix] = response_eligibility(member)
                members[suffix] = member
            eligible = status["a"]["eligible"] and status["b"]["eligible"]
            ledger.append({**context, "population": population, "eligible": eligible,
                           **{k+"_"+suffix: v for suffix in ("a", "b") for k, v in status[suffix].items()}})
            if not eligible:
                continue
            measured = {}
            for suffix in ("a", "b"):
                response, summaries = measured_response(members[suffix], temperature=population == "observed_temperature")
                coef = response.pop("fit_coefficients")
                measured[suffix] = response
                coefficients.append({**context, "population": population, "member": suffix, "coefficients": coef,
                                     "rank": response["fit_rank"], "n_columns": response["fit_columns"]})
                states.extend({**context, "population": population, "member": suffix, **st} for st in summaries)
            for metric in METRICS:
                va, vb = measured["a"][metric], measured["b"][metric]
                evidence.append({**context, "population": population, "metric": metric, "value_a": va, "value_b": vb,
                                 "difference_b_minus_a": vb-va, "n_common_months": len(s)})
            for state in ("low", "high"):
                together = s[s.flow_state_a.eq(state) & s.flow_state_b.eq(state)]
                for row in together.itertuples():
                    joint.append({**context, "population": population, "flow_state": state, "month_index": row.month_index,
                                  "date": row.date, "calendar_month": row.calendar_month,
                                  "doc_difference": row.y_true_b-row.y_true_a,
                                  "log_difference": np.log1p(row.y_true_b)-np.log1p(row.y_true_a),
                                  "high_difference": float(row.y_true_b >= HIGH_DOC)-float(row.y_true_a >= HIGH_DOC)})
    return (pd.DataFrame(evidence), pd.DataFrame(ledger), pd.DataFrame(states), coefficients, pd.DataFrame(joint))


def block_interval(frame, values, draws=5000):
    values = np.asarray(values, float)
    _, group = np.unique(frame.huc4, return_inverse=True)
    n = group.max()+1
    weights = np.random.default_rng(42).multinomial(n, np.full(n, 1/n), size=draws)[:, group]
    boot = weights@values/weights.sum(axis=1)
    lo, hi = np.quantile(boot, [.025, .975]) if n > 1 else (np.nan, np.nan)
    return float(lo), float(hi), int(n)


def summarize_pairs(evidence, draws=5000):
    rows, influence = [], []
    for key, g in evidence.groupby(["class_a", "class_b", "population", "metric"]):
        s = g[np.isfinite(g.difference_b_minus_a)]
        if s.empty:
            continue
        lo, hi, regions = block_interval(s, s.difference_b_minus_a, draws)
        context = dict(zip(("class_a", "class_b", "population", "metric"), key, strict=True))
        rows.append({**context, "mean_a": s.value_a.mean(), "mean_b": s.value_b.mean(),
                     "difference_b_minus_a": s.difference_b_minus_a.mean(), "ci_low": lo, "ci_high": hi,
                     "n_pairs": len(s), "n_huc4": regions, "positive_pairs": int(s.difference_b_minus_a.gt(0).sum())})
        if regions > 1:
            for region in s.huc4.unique():
                retained = s[s.huc4.ne(region)]
                influence.append({**context, "omitted_huc4": region, "difference_b_minus_a": retained.difference_b_minus_a.mean()})
    return pd.DataFrame(rows), pd.DataFrame(influence)


def summarize_stations(fits, draws=5000):
    rows = []
    for (population, cluster), g in fits.groupby(["population", "cluster"]):
        for metric in METRICS:
            s = g[np.isfinite(g[metric])]
            if s.empty:
                continue
            lo, hi, regions = block_interval(s, s[metric], draws)
            rows.append({"population": population, "cluster": cluster, "metric": metric, "mean": s[metric].mean(),
                         "median": s[metric].median(), "ci_low": lo, "ci_high": hi, "n_stations": len(s),
                         "n_huc4": regions, "positive_stations": int(s[metric].gt(0).sum())})
    return pd.DataFrame(rows)


def joint_evidence(records):
    rows, ledger = [], []
    for (pair, population), s in records.groupby(["pair_id", "population"]):
        r = s.iloc[0]
        low, high = s[s.flow_state.eq("low")], s[s.flow_state.eq("high")]
        context = {k: r[k] for k in ("pair_id", "station_a", "station_b", "class_a", "class_b", "huc4")}
        ledger.append({**context, "population": population, "n_joint_low": len(low), "n_joint_high": len(high),
                       "eligible": len(low) >= 4 and len(high) >= 4,
                       "low_month_sin_mean": np.sin(2*np.pi*low.calendar_month/12).mean(),
                       "high_month_sin_mean": np.sin(2*np.pi*high.calendar_month/12).mean(),
                       "low_month_cos_mean": np.cos(2*np.pi*low.calendar_month/12).mean(),
                       "high_month_cos_mean": np.cos(2*np.pi*high.calendar_month/12).mean()})
        if len(low) < 4 or len(high) < 4:
            continue
        for column in ("doc_difference", "log_difference", "high_difference"):
            rows.append({**context, "population": population, "metric": "joint_"+column,
                         "value_a": low[column].mean(), "value_b": high[column].mean(),
                         "difference_b_minus_a": high[column].mean()-low[column].mean()})
    return pd.DataFrame(rows), pd.DataFrame(ledger)


def estimable_coefficients(x, y, weight):
    weighted = x*np.sqrt(weight[:, None])
    coef = np.linalg.lstsq(weighted, y*np.sqrt(weight), rcond=None)[0]
    _, singular, vt = np.linalg.svd(weighted, full_matrices=False)
    rank = int(np.sum(singular > singular.max()*max(weighted.shape)*np.finfo(float).eps))
    identified = np.sum(vt[:rank]**2, axis=0) > 1-1e-8
    return np.where(identified, coef, np.nan), rank


def structure_associations(fits, panel, draws=5000):
    rows, saved = [], []
    for population, s in fits.groupby("population"):
        s = s[np.isfinite(s.adjusted_log_contrast)].merge(panel, on="station", validate="one_to_one", suffixes=("", "_panel"))
        for block, features in BLOCKS.items():
            names = [*CONTROLS, *features]
            valid = np.isfinite(s[names].to_numpy(float)).all(axis=1)
            f = s[valid].copy()
            values = f[names].to_numpy(float)
            mean, sd = values.mean(axis=0), values.std(axis=0)
            keep = sd > 1e-8
            x = (values[:, keep]-mean[keep])/sd[keep]
            columns = [name for name, k in zip(names, keep, strict=True) if k]
            # Form labels retain a class1 reference and a B-minus-A unit scale.
            for name in features:
                if name.startswith("form_") and name in columns:
                    x[:, columns.index(name)] = f[name]
            levels = sorted(f.huc2.unique())[1:]
            x = np.column_stack([np.ones(len(f)), x, *[(f.huc2 == level).to_numpy(float) for level in levels]])
            columns = ["intercept", *columns, *["huc2_"+level for level in levels]]
            y = f.adjusted_log_contrast.to_numpy(float)
            coef, rank = estimable_coefficients(x, y, np.ones(len(f)))
            _, group = np.unique(f.huc4, return_inverse=True)
            n = group.max()+1
            weights = np.random.default_rng(42).multinomial(n, np.full(n, 1/n), size=draws)[:, group]
            boot = np.array([estimable_coefficients(x, y, weight)[0] for weight in weights])
            saved.append({"population": population, "block": block, "columns": columns, "coef": coef.tolist(),
                          "numeric_columns": names, "numeric_mean": mean.tolist(), "numeric_sd": sd.tolist(),
                          "rank": rank, "n_stations": len(f), "n_huc4": int(n)})
            for name in features:
                if name not in columns:
                    rows.append({"population": population, "block": block, "feature": name, "coefficient": np.nan,
                                 "ci_low": np.nan, "ci_high": np.nan, "n_estimable_draws": 0, "n_stations": len(f), "n_huc4": n})
                    continue
                j = columns.index(name)
                usable = boot[np.isfinite(boot[:, j]), j]
                lo, hi = np.quantile(usable, [.025, .975]) if n > 1 and len(usable) > 1 else (np.nan, np.nan)
                rows.append({"population": population, "block": block, "feature": name, "coefficient": coef[j],
                             "ci_low": lo, "ci_high": hi, "n_estimable_draws": len(usable), "n_stations": len(f), "n_huc4": n})
    return pd.DataFrame(rows), saved
