"""Independent river-form replication at the river/site, not sample, grain."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from river_graph.analysis.river_morphology_effect import BRANCHING, FOOTPRINT, PATHS
from river_graph.analysis.river_planform_typology import BLOCKS, physical_features

CONTEXT = ("log_basin_area", "wetland", "forest", "agriculture", "urban",
           "log_precipitation", "climate_temperature", "latitude", "longitude", "log_n_months")
ARMS = {"context": (), "footprint": FOOTPRINT, "branching": BRANCHING,
        "paths": PATHS, "all_form": (*FOOTPRINT, *BRANCHING, *PATHS)}
OUTCOMES = ("doc_level_mgl", "doc_relative_iqr")


def calendar_endpoints(occasions, *, start="2018-01-01", end="2025-12-31",
                       minimum_months=36, minimum_years=6):
    """Equal monthly medians; level gives each calendar month equal weight."""
    f = occasions.copy()
    if f.duplicated(["siteID", "sample_base_id"]).any():
        raise ValueError("replicate-aggregated sample occasions must be unique")
    f["date"] = pd.to_datetime(f.collectDate)
    f = f[f.date.ge(pd.Timestamp(start)) & f.date.lt(pd.Timestamp(end) + pd.Timedelta(days=1))]
    f = f[f.waterbody_scope.eq("stream_or_river") & f.primary_surface_location & f.all_finite_nonnegative]
    f["month"] = f.date.dt.strftime("%Y-%m")
    monthly = f.groupby(["siteID", "month"], sort=True).doc_mean_mgl.median().rename("doc_mgl").reset_index()
    monthly["year"] = monthly.month.str[:4].astype(int)
    monthly["calendar_month"] = monthly.month.str[-2:].astype(int)
    rows = []
    for site, g in monthly.groupby("siteID", sort=True):
        season = g.groupby("calendar_month").doc_mgl.median()
        values = g.doc_mgl.to_numpy(float)
        median = np.median(values)
        eligible = len(g) >= minimum_months and g.year.nunique() >= minimum_years and len(season) == 12
        q25, q75 = np.quantile(values, [.25, .75])
        rows.append({"site": site, "n_months": len(g), "n_years": g.year.nunique(),
                     "n_calendar_months": len(season), "calendar_eligible": eligible,
                     "doc_level_mgl": float(season.median()),
                     "doc_relative_iqr": float((q75 - q25)/median) if median > 0 else np.nan,
                     "doc_cv": float(values.std(ddof=1)/values.mean()) if values.mean() > 0 else np.nan,
                     "doc_seasonal_amplitude": float((season.max()-season.min())/season.median()) if season.median() > 0 else np.nan,
                     "doc_month_median_mgl": median, "doc_month_q90_mgl": float(np.quantile(values, .9)),
                     "first_month": g.month.min(), "last_month": g.month.max()})
    return pd.DataFrame(rows), monthly


def upstream_overlap_groups(memberships):
    """Transitive shared upstream COMIDs form one holdout/resampling unit."""
    sites = sorted(memberships)
    parent = {s: s for s in sites}

    def root(s):
        while parent[s] != s:
            s = parent[s]
        return s

    pairs = []
    for i, a in enumerate(sites):
        for b in sites[i+1:]:
            common = memberships[a] & memberships[b]
            if common:
                parent[root(b)] = root(a)
            pairs.append({"site_a": a, "site_b": b, "shared_reaches": len(common),
                          "nested": memberships[a] <= memberships[b] or memberships[b] <= memberships[a]})
    result = {s: min(t for t in sites if root(t) == root(s)) for s in sites}
    return result, pd.DataFrame(pairs)


def transfer_form_classes(forms, old_features, old_centroids):
    """Frozen geometry-only nearest-centroid assignment, not new Ward fitting."""
    features = physical_features(forms)
    columns = list(old_features.columns)
    weights = {}
    for names in BLOCKS.values():
        kept = [n for n in names if n in columns]
        weights.update({n: 1/np.sqrt(len(kept)) for n in kept})
    scale = StandardScaler().fit(old_features)
    z = scale.transform(features[columns]) * np.asarray([weights[n] for n in columns])
    distances = np.linalg.norm(z[:, None, :] - old_centroids[columns].to_numpy()[None, :, :], axis=2)
    chosen = distances.argmin(axis=1)
    result = forms.copy()
    result["cluster"] = old_centroids.cluster.to_numpy(int)[chosen]
    result["centroid_distance"] = distances[np.arange(len(result)), chosen]
    # Useful scale/extrapolation diagnostics, never DOC-selected exclusions.
    train_z = scale.transform(old_features)
    result["outside_old_feature_range_count"] = ((features[columns].to_numpy() < old_features.min().to_numpy()) |
                                                 (features[columns].to_numpy() > old_features.max().to_numpy())).sum(axis=1)
    result["maximum_absolute_old_z"] = np.abs(z / np.asarray([weights[n] for n in columns])).max(axis=1)
    result["old_maximum_absolute_z"] = float(np.abs(train_z).max())
    for name in (*FOOTPRINT, "log_drainage_density"):
        result[name] = features[name].to_numpy()
    return result


def heldout_predictions(panel, *, group_column="network_group", alpha=10):
    """Complete independent networks held out; no tuning with held-out DOC."""
    groups = panel[group_column].to_numpy()
    if len(np.unique(groups)) < 5 or len(panel) < 10:
        raise ValueError("fewer than ten sites/five independent groups: no context-adjusted predictive replication")
    required = [*CONTEXT, *ARMS["all_form"], *OUTCOMES]
    if not np.isfinite(panel[required].to_numpy(float)).all():
        raise ValueError("complete finite context, form and outcome values required")
    rows = []
    for train, test in LeaveOneGroupOut().split(panel, groups=groups):
        for outcome in OUTCOMES:
            y = np.log1p(panel[outcome].to_numpy(float))
            for model, block in ARMS.items():
                columns = [*CONTEXT, *block]
                fit = make_pipeline(StandardScaler(), Ridge(alpha=alpha)).fit(panel.iloc[train][columns], y[train])
                pred = np.maximum(0, np.expm1(fit.predict(panel.iloc[test][columns])))
                for index, value in zip(test, pred, strict=True):
                    rows.append({"site": panel.site.iloc[index], "network_group": panel.network_group.iloc[index],
                                 "holdout_group": groups[index], "outcome": outcome, "model": model,
                                 "y_true": panel[outcome].iloc[index], "y_pred": value,
                                 "n_training_sites": len(train)})
    return pd.DataFrame(rows)


def gain_summary(predictions, *, draws=5000, seed=42):
    """Paired site-MAE gains; resample complete upstream-overlap groups."""
    if draws < 100:
        raise ValueError("at least 100 bootstrap draws required")
    rows, influences = [], []
    for outcome, part in predictions.groupby("outcome", sort=True):
        f = part.assign(error=abs(part.y_true-part.y_pred)).pivot(
            index=["site", "holdout_group"], columns="model", values="error").reset_index()
        if not np.isfinite(f[list(ARMS)].to_numpy()).all():
            raise ValueError("paired model coverage incomplete")
        groups, index = np.unique(f.holdout_group, return_inverse=True)
        w = np.random.default_rng(seed).multinomial(len(groups), np.full(len(groups), 1/len(groups)), size=draws)[:, index]
        reference = f.context.to_numpy()
        rb = w @ reference / w.sum(axis=1)
        for model in list(ARMS)[1:]:
            candidate = f[model].to_numpy()
            cb = w @ candidate / w.sum(axis=1)
            gain = 100*(rb-cb)/rb
            lo, hi = np.quantile(gain, [.025, .975])
            direction = []
            for group in groups:
                select = f.holdout_group.eq(group).to_numpy()
                direction.append(float((reference[select]-candidate[select]).mean()) > 0)
                keep = ~select
                influences.append({"outcome": outcome, "candidate": model, "removed_group": group,
                                   "gain_pct_remaining": 100*(reference[keep].mean()-candidate[keep].mean())/reference[keep].mean()})
            rows.append({"outcome": outcome, "candidate": model, "reference": "context",
                         "candidate_mae": candidate.mean(), "reference_mae": reference.mean(),
                         "gain_pct": 100*(reference.mean()-candidate.mean())/reference.mean(),
                         "ci_low_pct": lo, "ci_high_pct": hi,
                         "positive_groups": sum(direction), "n_groups": len(groups), "n_sites": len(f),
                         "bootstrap_unit": "declared holdout group; fixed out-of-group predictions"})
    return pd.DataFrame(rows), pd.DataFrame(influences)


def matched_shape_pairs(panel):
    """Existing elongated(1)/broad(3) classes; match on context, not DOC."""
    columns = ["log_basin_area", "wetland", "forest", "agriculture", "urban",
               "log_precipitation", "climate_temperature", "latitude", "longitude"]
    a, b = panel[panel.cluster.eq(1)], panel[panel.cluster.eq(3)]
    output = ["site_a", "site_b", "context_rms", "group_a", "group_b"]
    if a.empty or b.empty:
        return pd.DataFrame(columns=output)
    scaler = StandardScaler().fit(panel[columns])
    za, zb = scaler.transform(a[columns]), scaler.transform(b[columns])
    cost = np.sqrt(np.mean((za[:, None]-zb[None, :])**2, axis=2))
    valid = cost <= 1

    def diff(name):
        return abs(a[name].to_numpy()[:, None] - b[name].to_numpy()[None, :])

    valid &= diff("log_basin_area") <= np.log(2)
    valid &= diff("log_precipitation") <= np.log(1.25)
    valid &= diff("wetland") <= 5
    valid &= diff("forest") <= 20
    valid &= diff("climate_temperature") <= 3
    valid &= a.network_group.to_numpy()[:, None] != b.network_group.to_numpy()[None, :]
    i, j = linear_sum_assignment(np.where(valid, cost, 1e6))
    keep = valid[i, j]
    i, j = i[keep], j[keep]
    return pd.DataFrame({"site_a": a.iloc[i].site.to_numpy(), "site_b": b.iloc[j].site.to_numpy(),
                         "context_rms": cost[i, j], "group_a": a.iloc[i].network_group.to_numpy(),
                         "group_b": b.iloc[j].network_group.to_numpy()})
