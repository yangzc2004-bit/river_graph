"""Measured water/DOC accounting and local confluence geometry."""

from __future__ import annotations

import numpy as np
import pandas as pd


def mixing_ledger(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per archived downstream transect position, with paired inputs.

    DOC mg/L times discharge L/s is mg C/s. Source and downstream flow are
    measured separately; retain their mismatch rather than forcing closure.
    """
    d = frame.copy()
    if d.duplicated(["season", "location", "transect.location"]).any():
        raise ValueError("Repeated downstream position within a campaign")
    qm, qt, qd = (d[name].to_numpy(float) for name in
                  ("Q.Ls.main", "Q.Ls.trib", "Q.Ls"))
    cm, ct, cd = (d[name].to_numpy(float) for name in
                  ("DOC.mgL.mean.main", "DOC.mgL.mean.trib", "DOC.mgL.mean"))
    em, et, ed = (d[name].to_numpy(float) for name in ("SpC.main", "SpC.trib", "SpC"))
    if not np.isfinite(np.array([qm, qt, qd, cm, ct, cd, em, et, ed])).all():
        raise ValueError("Missing or nonfinite required observation")
    if (np.array([qm, qt, qd]) <= 0).any() or (np.array([cm, ct, cd]) < 0).any():
        raise ValueError("Invalid flow or DOC observation")
    qsum, incoming = qm + qt, qm * cm + qt * ct
    fraction = qm / qsum
    d["incoming_q_ls"] = qsum
    d["water_closure_pct"] = 100 * (qd / qsum - 1)
    d["main_water_fraction"] = fraction
    d["tributary_water_fraction"] = qt / qsum
    d["incoming_doc_mgs"] = incoming
    d["downstream_doc_mgs"] = qd * cd
    d["doc_flux_discrepancy_mgs"] = qd * cd - incoming
    d["doc_flux_discrepancy_pct"] = 100 * (qd * cd / incoming - 1)
    d["mix_doc_mgl"] = incoming / qsum
    d["mix_doc_measured_q_mgl"] = incoming / qd
    d["doc_departure_pct"] = 100 * (cd / d.mix_doc_mgl - 1)
    d["doc_departure_measured_q_pct"] = 100 * (cd / d.mix_doc_measured_q_mgl - 1)
    d["source_doc_contrast_mgl"] = ct - cm
    d["mix_spc_uscm"] = fraction * em + (1 - fraction) * et
    d["spc_departure_pct"] = 100 * (ed / d.mix_spc_uscm - 1)
    gap = em - et
    tracer = np.full(len(d), np.nan)
    np.divide(ed - et, gap, out=tracer, where=np.abs(gap) > 0)
    d["tracer_main_fraction"] = tracer
    d["tracer_endpoint_gap_uscm"] = np.abs(gap)
    d["tracer_in_bounds"] = np.isfinite(tracer) & (tracer >= 0) & (tracer <= 1)
    d["tracer_mix_doc_mgl"] = np.where(d.tracer_in_bounds, tracer * cm + (1 - tracer) * ct, np.nan)
    d["tracer_doc_departure_pct"] = 100 * (cd / d.tracer_mix_doc_mgl - 1)
    d["provided_fraction_sum"] = d["pQm.mtsum"] + d["pQt.mtsum"]
    d["provided_main_fraction_error"] = d["pQm.mtsum"] - fraction
    d["provided_trib_fraction_error"] = d["pQt.mtsum"] - qt / qsum
    d["provided_mix_doc_mgl"] = d["pQm.mtsum"] * cm + d["pQt.mtsum"] * ct
    return d


def geometry_tables(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Equal transects: a width repeated at depth positions is one width."""
    keys = ["location", "reach", "transect"]
    if raw.groupby(keys)["wettedwidth.m"].nunique().gt(1).any():
        raise ValueError("Contradictory width at one transect")
    positions = raw.groupby(keys, sort=True).agg(
        width_m=("wettedwidth.m", "first"),
        distance_m=("distance.to.confluence", "first"),
        depth_mean_m=("depth.m", "mean"),
        depth_max_m=("depth.m", "max"),
        depth_points=("depth.m", "count"),
        source_rows=("depth.m", "size"),
    ).reset_index()
    rows = []
    for (site, reach), group in positions.groupby(["location", "reach"]):
        width = group.width_m
        rows.append({"location": f"Con-{site.split('.')[0]}",
                     "reach": "upstream" if reach == "1.upstream" else "downstream",
                     "n_width_transects": len(group),
                     "n_depth_points": int(group.depth_points.sum()),
                     "n_depth_missing": int((group.source_rows - group.depth_points).sum()),
                     "width_mean_m": width.mean(),
                     "width_cv_pct": 100 * width.std(ddof=1) / width.mean(),
                     "depth_transect_mean_m": group.depth_mean_m.mean(),
                     "depth_max_m": group.depth_max_m.max()})
    return positions, pd.DataFrame(rows)


def campaign_summary(ledger: pd.DataFrame, geometry: pd.DataFrame) -> pd.DataFrame:
    rows = []
    fields = ["water_closure_pct", "tributary_water_fraction", "incoming_doc_mgs",
              "downstream_doc_mgs", "doc_flux_discrepancy_pct", "doc_departure_pct",
              "doc_departure_measured_q_pct", "spc_departure_pct", "tracer_doc_departure_pct",
              "source_doc_contrast_mgl", "mix_doc_mgl", "mix_doc_measured_q_mgl"]
    for (season, site), group in ledger.groupby(["season", "location"], sort=True):
        row = {"season": season, "location": site, "n_positions": len(group),
               "downstream_doc_mgl": group["DOC.mgL.mean"].mean(),
               "downstream_doc_min_mgl": group["DOC.mgL.mean"].min(),
               "downstream_doc_max_mgl": group["DOC.mgL.mean"].max(),
               "upstream_doc_mgl": group["DOC.mgL.mean.main"].mean(),
               "tributary_doc_mgl": group["DOC.mgL.mean.trib"].mean(),
               "downstream_doc_spatial_cv_pct": 100 * group["DOC.mgL.mean"].std(ddof=1) /
               group["DOC.mgL.mean"].mean(),
               "doc_departure_min_pct": group.doc_departure_pct.min(),
               "doc_departure_max_pct": group.doc_departure_pct.max(),
               "n_tracer_in_bounds": int(group.tracer_in_bounds.sum()),
               "main_wrt100_min": (100 / group["v.mmin.main"]).mean(),
               "downstream_wrt100_min": (100 / group["v.mmin"]).mean()}
        row.update({field: group[field].mean() for field in fields})
        row["wrt_downstream_main_ratio"] = row["downstream_wrt100_min"] / row["main_wrt100_min"]
        for reach in ["upstream", "downstream"]:
            geo = geometry[(geometry.location == site) & (geometry.reach == reach)]
            if len(geo) != 1:
                raise ValueError(f"Incomplete geometry at {site} {reach}")
            row.update({f"{reach}_{field}": geo.iloc[0][field] for field in
                        ["width_mean_m", "width_cv_pct", "depth_transect_mean_m",
                         "depth_max_m", "n_width_transects"]})
        row["width_downstream_upstream_ratio"] = row["downstream_width_mean_m"] / row["upstream_width_mean_m"]
        row["depth_downstream_upstream_ratio"] = row["downstream_depth_transect_mean_m"] / row["upstream_depth_transect_mean_m"]
        row["width_cv_change_pp"] = row["downstream_width_cv_pct"] - row["upstream_width_cv_pct"]
        row["geometry_season"] = "fall_only"
        rows.append(row)
    return pd.DataFrame(rows)


def descriptive_associations(campaigns: pd.DataFrame) -> pd.DataFrame:
    """Five spatial configurations, with omitted-configuration sensitivity."""
    specs = {
        "campaign_mean": ["wrt_downstream_main_ratio", "tributary_water_fraction", "source_doc_contrast_mgl"],
        "fall": ["width_downstream_upstream_ratio", "depth_downstream_upstream_ratio", "width_cv_change_pp"],
    }
    rows = []
    for population, features in specs.items():
        data = (campaigns.groupby("location").mean(numeric_only=True).reset_index()
                if population == "campaign_mean" else campaigns[campaigns.season == "fall"])
        for feature in features:
            sub = data[["location", feature, "doc_departure_pct"]].dropna()
            rho = sub[feature].corr(sub.doc_departure_pct, method="spearman")
            leave = [sub[sub.location != site][feature].corr(
                sub[sub.location != site].doc_departure_pct, method="spearman")
                for site in sub.location]
            rows.append({"population": population, "feature": feature,
                         "outcome": "doc_departure_pct", "n_confluences": len(sub),
                         "spearman": rho, "omitted_confluence_min": min(leave),
                         "omitted_confluence_max": max(leave),
                         "interpretation": "descriptive; five confluences in one network; no causal CI"})
    return pd.DataFrame(rows)
