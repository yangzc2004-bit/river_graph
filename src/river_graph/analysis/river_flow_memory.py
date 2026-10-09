"""Measured monthly flow memory with chronological and geographic checks."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from river_graph.analysis.river_hydrologic_activation import AMOUNT_CONTROLS
from river_graph.analysis.river_morphology_effect import BRANCHING, FOOTPRINT, PATHS

ARMS = {"context": (), "footprint": FOOTPRINT, "branching": BRANCHING,
        "paths": PATHS, "all_form": (*FOOTPRINT, *BRANCHING, *PATHS)}
RESPONSES = ("current_response", "previous_response", "combined_response")


def flow_month_pairs(dataset, cells, metadata):
    """Read target DOC only at allowed cells; previous flow uses its true month."""
    from river_graph.analysis.river_hydrologic_activation import monthly_source_panel

    out = monthly_source_panel(dataset, cells, metadata)
    months = pd.DatetimeIndex(dataset["months"])
    if not months.to_period("M").equals(pd.period_range(months[0], months[-1], freq="M")):
        raise ValueError("continuous unique calendar months required")
    x = np.asarray(dataset["x"], float)
    mask = np.asarray(dataset["x_mask"], bool)
    qi = list(dataset["feature_channels"]).index("discharge")
    ni = pd.Index(np.asarray(dataset["site_no"], str)).get_indexer(out.station)
    ti = months.get_indexer(out.month)
    if (ni < 0).any() or (ti < 0).any():
        raise ValueError("station/month not aligned to dataset")
    prev = np.maximum(ti-1, 0)
    out["previous_discharge_cfs"] = x[ni, prev, qi]
    out["previous_flow_observed"] = (ti > 0) & mask[ni, prev, qi]
    out.loc[~out.previous_flow_observed, "previous_discharge_cfs"] = np.nan
    out["pair_usable"] = (out.flow_observed & np.isfinite(out.discharge_cfs) & out.discharge_cfs.gt(0)
                          & out.previous_flow_observed & np.isfinite(out.previous_discharge_cfs)
                          & out.previous_discharge_cfs.gt(0) & out.landscape_included)
    return out


def design_matrices(train, query, *, temperature=False):
    """All references derive from train; changing query DOC never changes design."""
    month = train.month.dt.month.to_numpy()
    year = train.month.dt.year.to_numpy()+(month-1)/12
    year_ref = year.mean()
    qmean = np.log(train[["discharge_cfs", "previous_discharge_cfs"]].to_numpy(float)).mean(0)
    temp_mean = train.temperature_c.mean() if temperature else 0.

    def build(s):
        m = s.month.dt.month.to_numpy()
        yr = s.month.dt.year.to_numpy()+(m-1)/12
        nuisance = [np.ones(len(s)), np.sin(2*np.pi*m/12), np.cos(2*np.pi*m/12), (yr-year_ref)/10]
        if temperature:
            nuisance.append(s.temperature_c.to_numpy(float)-temp_mean)
        z = np.column_stack(nuisance)
        q = np.log(s[["discharge_cfs", "previous_discharge_cfs"]].to_numpy(float))-qmean
        if not np.isfinite(z).all() or not np.isfinite(q).all():
            raise ValueError("finite positive paired flow and nuisance required")
        return z, q

    return (*build(train), *build(query))


def fit_response(s, *, temperature=False, minimum_months=24):
    months, years = s.month.dt.month.nunique(), s.month.dt.year.nunique()
    record = {"n_months": len(s), "n_calendar_months": months, "n_years": years}
    if len(s) < minimum_months or months < 6 or years < 2:
        return record | {"status": "insufficient_paired_calendar_coverage"}, None
    z, q, _, _ = design_matrices(s, s, temperature=temperature)
    y = np.log1p(s.doc.to_numpy(float))
    v = np.column_stack([y, q])
    r = v-z@np.linalg.lstsq(z, v, rcond=None)[0]
    scale = r[:, 1:].std(0)
    if np.linalg.matrix_rank(z) < z.shape[1] or (scale < .05).any():
        return record | {"status": "insufficient_flow_variation_after_nuisance"}, None
    standardized = r[:, 1:]/scale
    condition = float(np.linalg.cond(standardized))
    if np.linalg.matrix_rank(standardized) < 2 or condition > 100:
        return record | {"status": "current_previous_flow_not_identifiable", "condition": condition}, None
    coef = np.linalg.lstsq(r[:, 1:], r[:, 0], rcond=None)[0]
    result = record | {"status": "included", "current_response": coef[0], "previous_response": coef[1],
                       "combined_response": coef.sum(), "condition": condition,
                       "current_previous_residual_rho": np.corrcoef(r[:, 1:].T)[0, 1],
                       "median_flow": s.discharge_cfs.median(), "median_year": s.month.dt.year.median()}
    residual = s[["station", "huc4", "month"]].copy()
    residual["doc_residual"], residual["current_flow_residual"], residual["previous_flow_residual"] = r.T
    return result, residual


def chronological_prediction(s, *, temperature=False):
    """Fixed half-year split, identified on train; no query response in inputs."""
    years = np.sort(s.month.dt.year.unique())
    record = {"train_months": 0, "query_months": 0}
    if len(years) < 4:
        return record | {"status": "fewer_than_four_years"}, pd.DataFrame()
    boundary = int(years[len(years)//2])
    train, query = s[s.month.dt.year < boundary], s[s.month.dt.year >= boundary]
    record.update(train_months=len(train), query_months=len(query), split_year=boundary)
    fit, _ = fit_response(train, temperature=temperature)
    if fit["status"] != "included" or len(query) < 12 or query.month.dt.month.nunique() < 6:
        return record | {"status": "insufficient_identified_train_or_query"}, pd.DataFrame()
    z, q, tz, tq = design_matrices(train, query, temperature=temperature)
    y, truth = np.log1p(train.doc.to_numpy(float)), np.log1p(query.doc.to_numpy(float))
    late_fit, _ = fit_response(query, temperature=temperature, minimum_months=12)
    record.update(status="included", train_current_response=fit["current_response"],
                  train_previous_response=fit["previous_response"], late_status=late_fit["status"],
                  late_current_response=late_fit.get("current_response", np.nan),
                  late_previous_response=late_fit.get("previous_response", np.nan))
    predictions = []
    for arm, count in (("season_trend", 0), ("current_flow", 1), ("flow_memory", 2)):
        design = np.column_stack([z, q[:, :count]])
        test_design = np.column_stack([tz, tq[:, :count]])
        coef = np.linalg.lstsq(design, y, rcond=None)[0]
        pred = np.maximum(0, test_design@coef)
        f = query[["station", "huc4", "month", "doc"]].copy()
        f["arm"], f["log_true"], f["log_pred"] = arm, truth, pred
        f["prediction"] = np.expm1(pred)
        if not np.isfinite(f.prediction).all():
            raise ValueError("nonfinite chronological prediction")
        predictions.append(f)
    return record, pd.concat(predictions, ignore_index=True)


def descriptor_predictions(stations):
    """Keep each HUC4 entirely outside training; score measured response only."""
    if stations.station.duplicated().any() or stations.huc4.nunique() < 5:
        raise ValueError("unique stations and >=5 regional groups required")
    rows = []
    for fold, (train, query) in enumerate(GroupKFold(5).split(stations, groups=stations.huc4)):
        a, b = stations.iloc[train], stations.iloc[query]
        for arm, extra in ARMS.items():
            numeric = [*AMOUNT_CONTROLS, *extra]
            prep = ColumnTransformer([
                ("numeric", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()), numeric),
                ("region", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["huc2"])])
            model = make_pipeline(prep, Ridge(alpha=10)).fit(a, a[list(RESPONSES)])
            prediction = model.predict(b)
            for i, r in enumerate(b.itertuples()):
                for j, response in enumerate(RESPONSES):
                    rows.append({"station": r.station, "huc4": r.huc4, "fold": fold, "arm": arm,
                                 "response": response, "true": getattr(r, response), "prediction": prediction[i, j]})
    return pd.DataFrame(rows)


def paired_gains(errors, comparisons, *, draws=5000, equal_folds=False):
    """Station-first, paired whole-region bootstrap; fixed held-out predictions."""
    keys = ["station", "huc4", *(["fold"] if equal_folds else [])]
    wide = errors.pivot(index=keys, columns="arm", values="error").reset_index()
    regions, ri = np.unique(wide.huc4, return_inverse=True)
    n = len(regions)
    if n < 2:
        raise ValueError(">=2 resampling regions required")
    folds = [wide.fold.eq(v).to_numpy() for v in sorted(wide.fold.unique())] if equal_folds else [np.ones(len(wide), bool)]
    rng = np.random.default_rng(42)
    batches = []
    while sum(len(b) for b in batches) < draws:
        weights = rng.multinomial(n, np.full(n, 1/n), size=draws)[:, ri]
        valid = np.all(np.column_stack([weights[:, f].sum(1) > 0 for f in folds]), axis=1)
        batches.append(weights[valid])
    weights = np.concatenate(batches)[:draws]
    rows = []
    for candidate, reference in comparisons:
        a, b = wide[candidate].to_numpy(), wide[reference].to_numpy()
        ma, mb = np.mean([a[f].mean() for f in folds]), np.mean([b[f].mean() for f in folds])
        ba = np.mean([weights[:, f]@a[f]/weights[:, f].sum(1) for f in folds], axis=0)
        bb = np.mean([weights[:, f]@b[f]/weights[:, f].sum(1) for f in folds], axis=0)
        boot = 100*(1-ba/bb)
        lo, hi = np.quantile(boot, [.025, .975])
        rows.append({"candidate": candidate, "reference": reference, "candidate_mae": ma, "reference_mae": mb,
                     "gain_pct": 100*(1-ma/mb), "ci_low_pct": lo, "ci_high_pct": hi,
                     "positive_stations": int((a < b).sum()), "n_stations": len(wide), "n_huc4": n,
                     "positive_folds": int(sum(a[f].mean() < b[f].mean() for f in folds)) if equal_folds else np.nan})
    return pd.DataFrame(rows)


def response_moderation(residuals, stations, *, draws=5000):
    """Jointly estimate structure modifiers of current and previous flow response."""
    from river_graph.analysis.river_hydrologic_activation import station_basis

    stations = stations.sort_values("station").reset_index(drop=True)
    numeric = [*AMOUNT_CONTROLS, *ARMS["all_form"]]
    basis, names = station_basis(stations, numeric)
    idx = pd.Index(stations.station).get_indexer(residuals.station)
    if (idx < 0).any():
        raise ValueError("missing station in response moderation")
    flow = residuals[["current_flow_residual", "previous_flow_residual"]].to_numpy(float)
    design = np.column_stack([flow[:, j, None]*basis[idx] for j in (0, 1)])
    labels = [f"{time}:{name}" for time in ("current", "previous") for name in names]
    y = residuals.doc_residual.to_numpy(float)
    w = 1/residuals.groupby("station").station.transform("size").to_numpy()
    scale = np.sqrt(np.maximum(np.sum(w[:, None]*design**2, axis=0), 1e-30))
    x = design/scale
    keep, dropped = [], []
    gram = x.T@(w[:, None]*x)
    for j in range(x.shape[1]):
        selected = [*keep, j]
        if np.linalg.matrix_rank(gram[np.ix_(selected, selected)], tol=1e-9) > len(keep):
            keep.append(j)
        else:
            dropped.append(j)
    x = x[:, keep]
    group, gi = np.unique(residuals.huc4, return_inverse=True)
    grams, rhs = [], []
    for j in range(len(group)):
        take = gi == j
        grams.append(x[take].T@(w[take, None]*x[take]))
        rhs.append(x[take].T@(w[take]*y[take]))
    grams, rhs = np.asarray(grams), np.asarray(rhs)
    point = np.full(len(labels), np.nan)
    point[keep] = (np.linalg.pinv(grams.sum(0), hermitian=True)@rhs.sum(0))/scale[keep]
    bootstrap = np.full((draws, len(labels)), np.nan)
    rng = np.random.default_rng(42)
    for start in range(0, draws, 100):
        copies = rng.multinomial(len(group), np.full(len(group), 1/len(group)), size=min(100, draws-start))
        bg = np.einsum("bi,ijk->bjk", copies, grams)
        br = copies@rhs
        bootstrap[start:start+len(copies), keep] = np.einsum("bij,bj->bi", np.linalg.pinv(bg, hermitian=True), br)/scale[keep]
    lo, hi = np.full(len(labels), np.nan), np.full(len(labels), np.nan)
    lo[keep], hi[keep] = np.quantile(bootstrap[:, keep], [.025, .975], axis=0)
    table = pd.DataFrame({"term": labels, "coefficient": point, "ci_low": lo, "ci_high": hi,
                          "identified": [j not in dropped for j in range(len(labels))]})
    return table, {"n_stations": len(stations), "n_huc4": len(group), "n_months": len(residuals),
                   "identified_columns": len(keep), "total_columns": len(labels),
                   "condition_scaled_design": float(np.linalg.cond(x)), "dropped_terms": [labels[j] for j in dropped]}
