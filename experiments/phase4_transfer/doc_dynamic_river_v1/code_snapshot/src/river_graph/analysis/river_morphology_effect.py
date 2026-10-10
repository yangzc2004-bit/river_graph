"""Morphology-centred DOC comparisons and identical-input channel routing."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

MATCH_COVARIATES = ("log_polygon_area", "wetland", "forest", "agriculture", "urban",
                    "precipitation", "climate_temperature", "latitude", "longitude", "median_year")
FOOTPRINT = ("log_basin_aspect", "log_network_axis_ratio")
BRANCHING = ("log_drainage_density", "mainstem_share")
PATHS = ("mainstem_sinuosity", "log_route_mean_scaled", "route_distance_cv")
# Freeze the preceding study's environmental baseline here so the new study
# does not require an unrelated, previously untracked analysis helper to import.
ENVIRONMENT_CONTROLS = (
    "log_polygon_area", "wetland", "forest", "agriculture", "urban", "precipitation",
    "climate_temperature", "log_flow", "median_temperature", "flow_available",
    "temperature_available", "median_year", "record_span", "n_calendar_months",
    "log_n_doc", "latitude", "longitude",
)


def morphology_pairs(panel, first, second, *, nested=None):
    """Maximum-cardinality minimum-covariate-distance matching within HUC4.

    Outcomes and morphology measurements do not enter pair selection. ``nested``
    is a square boolean matrix aligned to panel rows, constructed from COMIDs.
    """
    if panel.station.duplicated().any():
        raise ValueError("unique stations required")
    raw = panel[list(MATCH_COVARIATES)].to_numpy(float)
    if not np.isfinite(raw).all():
        raise ValueError("complete finite matching covariates required")
    scale = raw.std(axis=0)
    standardized = (raw-raw.mean(axis=0))/np.where(scale > 0, scale, 1.)
    a, b = np.flatnonzero(panel.cluster.eq(first)), np.flatnonzero(panel.cluster.eq(second))
    columns = ["station_a", "station_b", "class_a", "class_b", "huc4", "covariate_rms"]
    if not len(a) or not len(b):
        return pd.DataFrame(columns=columns), pd.DataFrame()
    A, B = panel.iloc[a], panel.iloc[b]
    cost = np.sqrt(np.mean((standardized[a, None]-standardized[None, b])**2, axis=2))
    def difference(name):
        return abs(A[name].to_numpy()[:, None]-B[name].to_numpy()[None, :])
    valid = (A.huc4.to_numpy()[:, None] == B.huc4.to_numpy()[None, :]) & (cost <= 1)
    # The stored covariate is log1p(area); apply the physical ratio to area,
    # rather than inadvertently applying it to area+1 for small catchments.
    area_a = np.expm1(A.log_polygon_area.to_numpy())
    area_b = np.expm1(B.log_polygon_area.to_numpy())
    if (area_a <= 0).any() or (area_b <= 0).any():
        raise ValueError("positive measured basin areas required")
    valid &= abs(np.log(area_a)[:, None]-np.log(area_b)[None, :]) <= np.log(2)
    valid &= difference("wetland") <= 5
    valid &= difference("forest") <= 20
    valid &= difference("climate_temperature") <= 3
    valid &= difference("median_year") <= 10
    valid &= abs(np.log(A.precipitation.to_numpy())[:, None]-np.log(B.precipitation.to_numpy())[None, :]) <= np.log(1.25)
    if nested is not None:
        nested = np.asarray(nested, bool)
        if nested.shape != (len(panel), len(panel)):
            raise ValueError("nested matrix must align with panel")
        valid &= ~nested[np.ix_(a, b)]
    # 1e6 exceeds the sum of any feasible costs, so valid-pair cardinality is
    # prioritized. Within that cardinality assignment minimizes covariate cost.
    row, col = linear_sum_assignment(np.where(valid, cost, 1e6))
    keep = valid[row, col]
    row, col = row[keep], col[keep]
    pairs = pd.DataFrame({"station_a": A.iloc[row].station.to_numpy(),
                          "station_b": B.iloc[col].station.to_numpy(),
                          "class_a": first, "class_b": second,
                          "huc4": A.iloc[row].huc4.to_numpy(), "covariate_rms": cost[row, col]})
    balance = []
    for j, name in enumerate(MATCH_COVARIATES):
        before = (raw[b, j].mean()-raw[a, j].mean())/scale[j] if scale[j] > 0 else 0.
        after = np.mean(standardized[b[col], j]-standardized[a[row], j]) if len(row) else np.nan
        balance.append({"class_a": first, "class_b": second, "covariate": name,
                        "unmatched_standardized_difference": before,
                        "matched_standardized_difference": after, "n_pairs": len(row)})
    return pairs, pd.DataFrame(balance)


def paired_response_summary(pairs, panel, outcomes, *, draws=5000):
    """Equal-pair mean differences with whole-HUC4 resampling; B minus A."""
    lookup = panel.set_index("station")
    rows, evidence = [], []
    for (a, b), matched in pairs.groupby(["class_a", "class_b"]):
        for name, transform in outcomes.items():
            xa = lookup.loc[matched.station_a, name].to_numpy(float)
            xb = lookup.loc[matched.station_b, name].to_numpy(float)
            valid = np.isfinite(xa) & np.isfinite(xb)
            p = matched[valid].copy()
            if not valid.any():
                continue
            xa, xb = xa[valid], xb[valid]
            delta = np.log1p(xb)-np.log1p(xa) if transform == "log1p" else xb-xa
            _, gi = np.unique(p.huc4, return_inverse=True)
            n = gi.max()+1
            weights = np.random.default_rng(42).multinomial(n, np.full(n, 1/n), size=draws)[:, gi]
            boot = weights@delta/weights.sum(axis=1)
            lo, hi = np.quantile(boot, [.025, .975]) if n >= 2 else (np.nan, np.nan)
            rows.append({"class_a": a, "class_b": b, "metric": name, "transform": transform,
                         "difference_b_minus_a": delta.mean(), "ci_low": lo, "ci_high": hi,
                         "n_pairs": len(delta), "n_huc4": n, "positive_pair_fraction": np.mean(delta > 0),
                         "few_blocks": n < 10,
                         "interval_status": "estimable" if n >= 2 else "not_estimable_single_HUC4"})
            p["metric"], p["transform"] = name, transform
            p["value_a"], p["value_b"], p["difference_b_minus_a"] = xa, xb, delta
            evidence.append(p)
    return pd.DataFrame(rows), pd.concat(evidence, ignore_index=True) if evidence else pd.DataFrame()


def geometry_routing(area, distance, basin_area, *, bins=20, grid=400):
    """Vegetation-independent distances and a conservative routing scenario.

    Each local catchment supplies the SAME concentration-anomaly signal with
    constant flow proportional to its area. The maximum route delay is one.
    Quantized route delays have resolution 1/grid; this is not a field time.
    """
    area, distance = np.asarray(area, float), np.asarray(distance, float)
    if area.ndim != 1 or area.shape != distance.shape or not len(area):
        raise ValueError("aligned nonempty one-dimensional areas/distances required")
    if not np.isfinite(area).all() or (area < 0).any() or not np.isfinite(basin_area) or basin_area <= 0:
        raise ValueError("finite nonnegative catchment and positive basin areas required")
    valid = (area > 0) & np.isfinite(distance) & (distance >= 0)
    if not valid.any() or area.sum() <= 0:
        raise ValueError("positive reachable catchment area required")
    a, d = area[valid], distance[valid]
    weights = a/a.sum()
    mean = np.sum(weights*d)
    sd = np.sqrt(np.sum(weights*(d-mean)**2))
    maximum = d.max()
    normalized = d/maximum if maximum > 0 else np.zeros(len(d))
    mass, edges = np.histogram(normalized, bins=np.linspace(0, 1, bins+1), weights=weights)
    kernel = np.bincount(np.rint(normalized*grid).astype(int), weights=weights, minlength=grid+1)
    forcing_time = np.linspace(-.5, .5, grid+1)
    forcing = np.exp(-.5*(forcing_time/.05)**2)
    output = np.convolve(forcing, kernel)
    time = -.5+np.arange(len(output))/grid
    centroid = np.sum(time*output)/output.sum()
    spread = np.sqrt(np.sum((time-centroid)**2*output)/output.sum())
    result = {"route_area_coverage": a.sum()/area.sum(), "route_mean_km": mean,
              "route_sd_km": sd, "route_max_km": maximum,
              "route_mean_scaled": mean/np.sqrt(basin_area),
              "log_route_mean_scaled": np.log1p(mean/np.sqrt(basin_area)),
              "route_distance_cv": sd/mean if mean > 0 else 0.,
              "route_peak_bin_mass": mass.max(), "routing_pulse_peak": output.max(),
              "routing_pulse_spread": spread, "routing_pulse_centroid": centroid,
              "routing_pulse_mass_ratio": output.sum()/forcing.sum(), "n_reachable_reaches": valid.sum()}
    profile = pd.DataFrame({"distance_fraction": (edges[:-1]+edges[1:])/2, "area_mass": mass})
    pulse = pd.DataFrame({"normalized_time": time, "routed_anomaly": output,
                          "no_delay_anomaly": np.pad(forcing, (0, grid))})
    return result, profile, pulse


def group_descriptor_summary(panel, metrics, *, draws=5000):
    rows = []
    for group, s in panel.groupby("cluster"):
        for name in metrics:
            sub = s.dropna(subset=[name])
            if sub.empty:
                continue
            values = sub[name].to_numpy(float)
            _, gi = np.unique(sub.huc4, return_inverse=True)
            n = gi.max()+1
            weight = np.random.default_rng(42).multinomial(n, np.full(n, 1/n), size=draws)[:, gi]
            boot = weight@values/weight.sum(axis=1)
            lo, hi = np.quantile(boot, [.025, .975])
            rows.append({"cluster": group, "metric": name, "mean": values.mean(),
                         "median": np.median(values), "ci_low": lo, "ci_high": hi,
                         "n_stations": len(values), "n_huc4": n})
    return pd.DataFrame(rows)
