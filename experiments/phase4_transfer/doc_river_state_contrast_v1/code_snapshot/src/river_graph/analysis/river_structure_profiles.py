"""Integrate fixed river outlines, actual corridor storage and DOC evidence."""
from __future__ import annotations

import numpy as np
import pandas as pd

from river_graph.analysis.river_storage_placement import blocked_summary

POSITION_ORDER = ("none_on_corridor", "branches_only", "common_only", "branches_and_common")


def corridor_storage_profile(corridor, vaa, weight_a):
    """Partition actual, already cropped channel lengths; never infer residence time."""
    required = {"comid", "segment", "length_km"}
    if required-set(corridor) or corridor.empty or corridor.comid.duplicated().any():
        raise ValueError("unique nonempty cropped corridor reaches required")
    if set(corridor.segment) != {"branch_a", "branch_b", "common"}:
        raise ValueError("both independent paths and common trunk required")
    if not np.isfinite(weight_a) or not 0 < weight_a < 1:
        raise ValueError("positive two-branch shares required")
    lengths = corridor.length_km.to_numpy(float)
    if not np.isfinite(lengths).all() or (lengths < 0).any():
        raise ValueError("finite nonnegative cropped lengths required")
    if vaa.comid.duplicated().any():
        raise ValueError("unique VAA COMIDs required")
    joined = corridor.merge(vaa[["comid", "wbareatype", "wbareacomi"]],
                            on="comid", how="left", validate="one_to_one", indicator=True)
    if joined._merge.ne("both").any():
        raise ValueError("every corridor reach must have a VAA record")
    storage = joined.wbareatype.isin(["LakePond", "Reservoir"]) & joined.length_km.gt(0)
    if joined.loc[storage, "wbareacomi"].isna().any() or (joined.loc[storage, "wbareacomi"] <= 0).any():
        raise ValueError("mapped storage requires positive physical waterbody IDs")
    result = {}
    for segment in ("branch_a", "branch_b", "common"):
        mask = joined.segment.eq(segment)
        length = joined.loc[mask, "length_km"].sum()
        if length <= 0:
            raise ValueError("positive length on each selected segment required")
        stored = joined.loc[mask & storage, "length_km"].sum()
        result[segment+"_km"] = float(length)
        result[segment+"_storage_km"] = float(stored)
        result[segment+"_storage_fraction"] = float(stored/length)
    wa, wb = weight_a, 1-weight_a
    branches = wa*result["branch_a_storage_km"]+wb*result["branch_b_storage_km"]
    common = result["common_storage_km"]
    path = wa*result["branch_a_km"]+wb*result["branch_b_km"]+result["common_km"]
    result["weighted_branch_storage_km"] = branches
    result["weighted_corridor_storage_km"] = branches+common
    result["corridor_storage_path_share"] = (branches+common)/path
    result["shared_storage_budget_fraction"] = common/(branches+common) if branches+common > 0 else np.nan
    result["storage_position"] = POSITION_ORDER[int(branches > 0)+2*int(common > 0)]
    result["n_unique_corridor_waterbodies"] = int(joined.loc[storage, "wbareacomi"].nunique())
    return result


def assemble_profiles(panel, whole, selected, partition, routing):
    """Station rows retain canonical geography and eligibility; all joins explicit."""
    for frame in (panel, whole, selected, partition):
        if frame.station.duplicated().any():
            raise ValueError("one row per station in each input table required")
    if panel.huc4.isna().any() or not panel.huc4.astype(str).str.fullmatch(r"\d{4}").all():
        raise ValueError("canonical four-digit HUC4 strings required")
    frame = panel.copy()
    identity = frame[["station", "comid", "cluster", "huc4"]].merge(
        whole[["station", "comid", "cluster", "huc4"]], on="station", how="outer",
        suffixes=("_panel", "_whole"), validate="one_to_one", indicator=True)
    if identity._merge.ne("both").any() or any(
        identity[name+"_panel"].ne(identity[name+"_whole"]).any() for name in ("comid", "cluster", "huc4")
    ):
        raise ValueError("whole-network descriptors must match the complete canonical station cohort")
    columns = ["station", "mean_delay", "path_variance", "storage_mean_share", "storage_exposed_flow_share"]
    frame = frame.merge(whole[columns], on="station", validate="one_to_one")
    if (frame.mean_delay <= 0).any() or (frame.path_variance < 0).any():
        raise ValueError("positive path means and nonnegative variances required")
    frame["arrival_dispersion"] = np.sqrt(frame.path_variance)/frame.mean_delay
    frame["mapped_storage_path_share"] = frame.storage_mean_share
    frame["mapped_storage_exposure"] = frame.storage_exposed_flow_share
    major = selected[["station", "common_fraction", "relative_arrival_cv", "selected_pair_area_share"]].rename(
        columns={"common_fraction": "major_common_share", "relative_arrival_cv": "major_arrival_dispersion",
                 "selected_pair_area_share": "major_pair_area_share"})
    if set(partition.station) != set(major.station):
        raise ValueError("mapped storage partitions must cover all selected confluences")
    frame = frame.merge(major, on="station", how="left", validate="one_to_one")
    frame = frame.merge(partition.drop(columns=["comid", "cluster", "huc4"], errors="ignore"),
                        on="station", how="left", validate="one_to_one")
    frame["major_confluence_available"] = frame.major_common_share.notna()
    # Routing descriptors are unique networks, rather than repeated station rows.
    actual = routing[routing.scenario.eq("actual_spread")]
    if actual.comid.duplicated().any():
        raise ValueError("one actual-spread routing record per receiving network required")
    frame = frame.merge(actual[["comid", "pulse_peak", "pulse_sd"]].rename(
        columns={"pulse_peak": "relative_time_pulse_peak", "pulse_sd": "relative_time_pulse_sd"}),
        on="comid", how="left", validate="many_to_one")
    if frame.relative_time_pulse_peak.isna().any():
        raise ValueError("all canonical networks require an existing identical-input routing result")
    return frame


def summarize_profiles(frame, metrics, draws=5000):
    """Metric-specific availability; bootstrap the actual geographic blocks."""
    rows = []
    for cluster, group in frame.groupby("cluster"):
        for metric in metrics:
            use = group[np.isfinite(group[metric])]
            if use.empty:
                continue
            row = blocked_summary(use, [metric], group="huc4", unit="station", draws=draws)
            row["cluster"] = cluster
            row["n_class_total"] = len(group)
            row["n_missing"] = len(group)-len(use)
            row["interval_scope"] = "HUC4 block-bootstrap station-equal mean" if use.huc4.nunique() > 1 else "unavailable: one HUC4"
            rows.append(row)
    return pd.concat(rows, ignore_index=True)


def storage_position_counts(frame):
    """Conditional denominator is eligible selected corridors, not all networks."""
    rows = []
    for cluster, group in frame.groupby("cluster"):
        use = group[group.major_confluence_available]
        for position in POSITION_ORDER:
            count = int(use.storage_position.eq(position).sum())
            rows.append({"cluster": cluster, "storage_position": position, "n": count,
                         "n_selected_corridors": len(use), "n_class_total": len(group),
                         "fraction": count/len(use) if len(use) else np.nan})
    return pd.DataFrame(rows)
