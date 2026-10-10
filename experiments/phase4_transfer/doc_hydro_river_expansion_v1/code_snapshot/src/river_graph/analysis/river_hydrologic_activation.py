"""Within-station monthly C-Q responses and landscape moderation."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

AMOUNT_CONTROLS = (
    "log_polygon_area", "wetland", "forest", "agriculture", "urban",
    "precipitation", "climate_temperature", "latitude", "longitude",
    "median_year", "log_flow",
)


def monthly_source_panel(dataset, cells, station_panel):
    """Read DOC only at explicitly allowed cells; keep hydro absence explicit."""
    y = np.asarray(dataset["y"])
    cells = np.asarray(cells, dtype=np.int64)
    if cells.ndim != 1 or len(np.unique(cells)) != len(cells):
        raise ValueError("unique one-dimensional cells required")
    if (cells < 0).any() or (cells >= y.size).any():
        raise ValueError("source cell out of range")
    if not np.asarray(dataset["y_mask"], bool).ravel()[cells].all():
        raise ValueError("all source DOC must be observed")
    doc = y.ravel()[cells].astype(float)
    if not np.isfinite(doc).all() or (doc < 0).any():
        raise ValueError("finite nonnegative source DOC required")
    n, t = cells//y.shape[1], cells % y.shape[1]
    channels = list(dataset["feature_channels"])
    x, mask = np.asarray(dataset["x"]), np.asarray(dataset["x_mask"], bool)
    q, temp = channels.index("discharge"), channels.index("temperature")
    dates = pd.DatetimeIndex(dataset["months"])[t]
    out = pd.DataFrame({"cell": cells, "station": np.asarray(dataset["site_no"], str)[n],
                        "month": dates, "doc": doc, "discharge_cfs": x[n, t, q].astype(float),
                        "temperature_c": x[n, t, temp].astype(float),
                        "flow_observed": mask[n, t, q], "temperature_observed": mask[n, t, temp]})
    out["flow_usable"] = out.flow_observed & np.isfinite(out.discharge_cfs) & out.discharge_cfs.ge(0)
    out["temperature_usable"] = out.temperature_observed & np.isfinite(out.temperature_c)
    out["flow_reason"] = np.select(
        [~out.flow_observed, ~np.isfinite(out.discharge_cfs), out.discharge_cfs.lt(0)],
        ["unobserved", "nonfinite", "negative_or_reverse_flow"], default="usable")
    for name, observed in (("discharge_cfs", "flow_observed"), ("temperature_c", "temperature_observed")):
        out.loc[~out[observed], name] = np.nan
    if station_panel.station.duplicated().any():
        raise ValueError("unique station metadata required")
    out = out.merge(station_panel, on="station", how="left", validate="many_to_one")
    if out.duplicated(["station", "month"]).any():
        raise ValueError("station-month grain is not unique")
    out["landscape_included"] = out.included.eq(True)
    # Nullable metadata must survive Parquet without NaN/None reinterpretation.
    for name in ("eligible", "included"):
        if name in out:
            out[name] = out[name].astype("boolean")
    for name in out:
        if pd.api.types.is_object_dtype(out[name]) or pd.api.types.is_string_dtype(out[name]):
            out[name] = out[name].astype("string")
    return out.sort_values(["station", "month"]).reset_index(drop=True)


def station_flow_responses(panel, *, temperature=False, minimum_months=24):
    """Remove station-specific season/trend before estimating linear/quadratic C-Q."""
    residuals, summaries = [], []
    for station, all_rows in panel.groupby("station", sort=True):
        s = all_rows[all_rows.flow_usable & all_rows.landscape_included].copy()
        if temperature:
            s = s[s.temperature_usable].copy()
        months = s.month.dt.month.to_numpy()
        years = s.month.dt.year.to_numpy()
        reason = "included"
        q = np.log1p(s.discharge_cfs.to_numpy(float))
        if not all_rows.landscape_included.any():
            reason = "landscape_or_mapping_excluded"
        elif len(s) < minimum_months:
            reason = "fewer_than_24_usable_months"
        elif len(np.unique(months)) < 6 or len(np.unique(years)) < 2:
            reason = "insufficient_calendar_coverage"
        elif q.std() < .1:
            reason = "insufficient_log_flow_variation"
        record = {"station": station, "n_source_doc": len(all_rows), "n_doc_flow": len(s),
                  "calendar_months": len(np.unique(months)), "years": len(np.unique(years)),
                  "response_status": reason}
        if reason != "included":
            summaries.append(record)
            continue
        fractional_year = years+(months-1)/12
        nuisance = [np.ones(len(s)), np.sin(2*np.pi*months/12), np.cos(2*np.pi*months/12),
                    (fractional_year-fractional_year.mean())/10]
        if temperature:
            v = s.temperature_c.to_numpy(float)
            nuisance.append(v-v.mean())
        nuisance = np.column_stack(nuisance)
        median = np.median(q)
        centered = q-median
        variables = np.column_stack([np.log1p(s.doc.to_numpy(float)), centered, centered**2])
        residual = variables-nuisance @ np.linalg.lstsq(nuisance, variables, rcond=None)[0]
        design = residual[:, 1:]
        if design[:, 0].std() < 1e-6 or np.linalg.matrix_rank(design, tol=1e-8) < 2:
            summaries.append(record | {"response_status": "flow_not_identifiable_after_season_trend"})
            continue
        coef = np.linalg.lstsq(design, residual[:, 0], rcond=None)[0]
        q25, q75 = np.quantile(q, [.25, .75])
        contrast = coef[0]*(q75-q25)+coef[1]*((q75-median)**2-(q25-median)**2)
        record.update(cq_linear=coef[0], cq_quadratic=coef[1], interquartile_response=contrast,
                      logq_median=median, logq_q25=q25, logq_q75=q75, logq_sd=q.std(),
                      nuisance_rank=int(np.linalg.matrix_rank(nuisance)),
                      response_condition=float(np.linalg.cond(design)),
                      flow_residual_sd=design[:, 0].std(),
                      median_flow=s.discharge_cfs.median(), median_year=np.median(years),
                      station_fit_mae=np.mean(abs(residual[:, 0]-design @ coef)))
        s["log_doc_residual"], s["linear_flow_residual"], s["quadratic_flow_residual"] = residual.T
        s["logq_centered"] = centered
        residuals.append(s)
        summaries.append(record)
    return (pd.concat(residuals, ignore_index=True) if residuals else pd.DataFrame(),
            pd.DataFrame(summaries))


def station_basis(stations, numeric, *, classes=False):
    """One covariate row per station, not per observation."""
    imputer = SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)
    raw = imputer.fit_transform(stations[list(numeric)])
    names = list(imputer.get_feature_names_out(numeric))
    variable = raw.std(axis=0) > 1e-10
    names = [name for name, keep in zip(names, variable, strict=True) if keep]
    z = StandardScaler().fit_transform(raw[:, variable])
    region = pd.get_dummies(stations.huc2, prefix="huc2", drop_first=True, dtype=float)
    parts = [np.ones((len(stations), 1)), z, region.to_numpy()]
    labels = ["flow_intercept", *names, *region.columns]
    if classes:
        for group in (2, 3):
            parts.append(stations.cluster.eq(group).to_numpy(float)[:, None])
            labels.append(f"class_{group}_vs_1")
    return np.column_stack(parts), labels


def clustered_interaction_fit(residuals, stations, numeric, *, classes=False, draws=5000, seed=42):
    """Equal-station-weight within-response fit, paired whole-HUC4 bootstrap."""
    if residuals.empty or stations.station.duplicated().any():
        raise ValueError("nonempty residuals and unique station covariates required")
    stations = stations.sort_values("station").reset_index(drop=True)
    basis, labels = station_basis(stations, numeric, classes=classes)
    idx = pd.Index(stations.station).get_indexer(residuals.station)
    if (idx < 0).any():
        raise ValueError("residual station missing from covariates")
    design = np.column_stack([residuals.linear_flow_residual.to_numpy()[:, None]*basis[idx],
                              residuals.quadratic_flow_residual.to_numpy()])
    labels.append("common_flow_curvature")
    y = residuals.log_doc_residual.to_numpy(float)
    counts = residuals.groupby("station").station.transform("size").to_numpy()
    w = 1/counts
    if not np.isfinite(design).all() or not np.isfinite(y).all():
        raise ValueError("finite residual design required")
    gram = design.T @ (w[:, None]*design)
    scale = np.sqrt(np.maximum(np.diag(gram), 1e-30))
    correlation = gram/scale[:, None]/scale[None, :]
    keep, dropped = [], []
    for i in range(design.shape[1]):
        trial = [*keep, i]
        if np.linalg.matrix_rank(correlation[np.ix_(trial, trial)], tol=1e-9) > len(keep):
            keep.append(i)
        else:
            dropped.append(i)
    x = design[:, keep]/scale[keep]
    region, region_index = np.unique(stations.huc4.to_numpy()[idx], return_inverse=True)
    grams, rhs = [], []
    for group in range(len(region)):
        take = region_index == group
        grams.append(x[take].T @ (w[take, None]*x[take]))
        rhs.append(x[take].T @ (w[take]*y[take]))
    grams, rhs = np.asarray(grams), np.asarray(rhs)
    point = np.zeros(design.shape[1])
    point[keep] = (np.linalg.pinv(grams.sum(0), hermitian=True) @ rhs.sum(0))/scale[keep]
    rng = np.random.default_rng(seed)
    bootstrap = np.zeros((draws, design.shape[1]))
    for start in range(0, draws, 100):
        size = min(100, draws-start)
        copies = rng.multinomial(len(region), np.full(len(region), 1/len(region)), size=size)
        bg = np.einsum("bi,ijk->bjk", copies, grams)
        br = copies @ rhs
        estimates = np.einsum("bij,bj->bi", np.linalg.pinv(bg, hermitian=True), br)
        bootstrap[start:start+size, keep] = estimates/scale[keep]
    lo, hi = np.quantile(bootstrap, [.025, .975], axis=0)
    table = pd.DataFrame({"term": labels, "coefficient": point, "ci_low": lo, "ci_high": hi,
                          "identified": [i not in dropped for i in range(len(labels))]})
    table["n_stations"], table["n_huc4"], table["n_months"] = len(stations), len(region), len(y)
    diagnostic = {"n_columns": len(labels), "identified_columns": len(keep),
                  "dropped_terms": [labels[i] for i in dropped],
                  "condition_scaled_design": float(np.linalg.cond(x)),
                  "station_weight_range": [float(w.min()), float(w.max())]}
    return table, bootstrap, basis, diagnostic
