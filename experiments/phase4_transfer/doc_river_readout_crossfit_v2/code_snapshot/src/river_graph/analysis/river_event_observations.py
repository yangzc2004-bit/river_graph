"""Observed DOC sampling resolution and mapped stream connections."""

from __future__ import annotations

import io
from datetime import timedelta, timezone
from itertools import pairwise
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from pyproj import Transformer
from shapely.geometry import Point, shape
from shapely.ops import transform


def read_sites_csv(path: Path, variable: str) -> pd.DataFrame:
    """Keep UTC and the file's fixed UTC+1 calendar; do not assume DST."""
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    header = next(i for i, line in enumerate(lines) if line.startswith("TIMESTAMP,"))
    preamble = "\n".join(lines[:header]).replace(" ", "")
    if "TIMEZONE:UTC+1" not in preamble:
        raise ValueError(f"Unrecognized timezone: {path.name}")
    expected_unit = "mgCL-1" if variable == "DOC" else "m3s-1"
    if expected_unit not in preamble:
        raise ValueError(f"Unrecognized {variable} unit: {path.name}")
    raw = pd.read_csv(io.StringIO("\n".join(lines[header:])), dtype=str)
    times = pd.to_datetime(raw.TIMESTAMP, errors="raise")
    values = pd.to_numeric(raw[variable], errors="coerce")
    result = pd.DataFrame({
        "timestamp_local": times,
        "timestamp_utc": times.dt.tz_localize(timezone(timedelta(hours=1))).dt.tz_convert("UTC"),
        "date_local": times.dt.normalize(),
        "value": values,
        "raw_value": raw[variable],
        "censored": raw[variable].eq("LOD"),
    })
    if (values.dropna() < 0).any() or np.isinf(values.dropna()).any():
        raise ValueError(f"Invalid concentration/discharge: {path.name}")
    duplicated = result[result.timestamp_utc.duplicated(keep=False)]
    if not duplicated.empty and (duplicated.groupby("timestamp_utc").value.nunique(dropna=False) > 1).any():
        raise ValueError(f"Conflicting duplicate sample times: {path.name}")
    return result.drop_duplicates("timestamp_utc").sort_values("timestamp_utc").reset_index(drop=True)


def cadence(frame: pd.DataFrame) -> dict:
    valid = frame[frame.value.notna()].sort_values("timestamp_utc")
    gaps = valid.timestamp_utc.diff().dt.total_seconds().div(86400).dropna()
    spring = valid[valid.date_local.dt.month.between(3, 6)]
    # Never use a cross-winter gap as an interval within spring sampling.
    spring_gaps = spring.groupby(spring.date_local.dt.year).timestamp_utc.diff()
    spring_gaps = spring_gaps.dt.total_seconds().div(86400).dropna()
    return {
        "n_rows": len(frame), "n_valid": len(valid), "n_censored": int(frame.censored.sum()),
        "n_missing": int(frame.value.isna().sum()),
        "n_sample_days": int(valid.date_local.nunique()),
        "first_date": str(valid.date_local.min().date()),
        "last_date": str(valid.date_local.max().date()),
        "median_gap_days": gaps.median(), "min_gap_days": gaps.min(),
        "fraction_intervals_le1day": gaps.le(1).mean(),
        "fraction_intervals_le7days": gaps.le(7).mean(),
        "spring_median_gap_days": spring_gaps.median(),
        "spring_min_gap_days": spring_gaps.min(),
    }


def mapped_network(features: list[dict], stations: pd.DataFrame, max_snap_m: float = 20):
    """Insert each station at its projected position inside its mapped reach."""
    project = Transformer.from_crs(4326, 3006, always_xy=True).transform
    lines = [transform(project, shape(f["geometry"])) for f in features]
    graph = nx.DiGraph()
    snapping = []
    placements: dict[int, list[tuple[float, str]]] = {}
    for row in stations.itertuples():
        point = Point(*project(row.longitude, row.latitude))
        distances = np.array([line.distance(point) for line in lines])
        i = int(distances.argmin())
        along = lines[i].project(point)
        accepted = distances[i] <= max_snap_m
        snapping.append({"site": row.site, "arcid": features[i]["properties"]["ARCID"],
                         "snap_distance_m": distances[i], "along_fraction": along / lines[i].length,
                         "mapped": accepted, "x_m": point.x, "y_m": point.y})
        if accepted:
            placements.setdefault(i, []).append((along, f"site:{row.site}"))
    for i, (feature, line) in enumerate(zip(features, lines)):
        properties = feature["properties"]
        positions = [(0., f"node:{properties['FROM_NODE']}")]
        positions += sorted(placements.get(i, []))
        positions.append((line.length, f"node:{properties['TO_NODE']}"))
        for (start, a), (stop, b) in pairwise(positions):
            if a != b:
                graph.add_edge(a, b, length_m=stop - start, arcid=properties["ARCID"],
                               along_start_m=start, along_end_m=stop)
    return graph, pd.DataFrame(snapping), lines


def mapped_relations(graph: nx.DiGraph, snapping: pd.DataFrame, polygons: dict) -> pd.DataFrame:
    """Directed routes; metadata polygons are spatial context, not total basins."""
    sites = sorted(snapping.loc[snapping.mapped, "site"])
    project = Transformer.from_crs(4326, 3006, always_xy=True).transform
    projected = {site: transform(project, polygon) for site, polygon in polygons.items()}
    rows = []
    for source in sites:
        for receiver in sites:
            if source == receiver:
                continue
            a, b = f"site:{source}", f"site:{receiver}"
            if not nx.has_path(graph, a, b):
                continue
            path = nx.shortest_path(graph, a, b, weight="length_m")
            containment = np.nan
            if source in projected and receiver in projected:
                containment = projected[source].intersection(projected[receiver]).area / projected[source].area
            cyclic = nx.has_path(graph, b, a)
            intervening = [n[5:] for n in path[1:-1] if n.startswith("site:")]
            rows.append({
                "source": source, "receiver": receiver,
                "path_km": sum(graph[a][b]["length_m"] for a, b in pairwise(path)) / 1000,
                "metadata_polygon_overlap_share": containment,
                "mutual_reachability": cyclic,
                "intervening_stations": ";".join(intervening),
                "nearest_monitored_upstream": not intervening,
                "usable_mapped_connection": not cyclic,
            })
    return pd.DataFrame(rows)


def spring_windows(flow: pd.DataFrame, years: list[int]) -> pd.DataFrame:
    """Select annual March--June flow maxima without looking at DOC values."""
    rows = []
    for year in years:
        spring = flow[(flow.date_local.dt.year == year) & flow.date_local.dt.month.between(3, 6)]
        valid = spring[spring.value.notna()]
        if len(valid) < 98:  # at least 80% of the fixed 122-day spring period
            continue
        peak = valid.loc[valid.value.idxmax()]
        peak_date = peak.date_local
        rows.append({"year": year, "peak_date": peak_date,
                     "start_date": peak_date - pd.Timedelta(days=14),
                     "end_date": peak_date + pd.Timedelta(days=14),
                     "peak_q_m3s": peak.value, "n_spring_flow_days": len(valid)})
    return pd.DataFrame(rows)


def window_coverage(samples: pd.DataFrame, peak_date: pd.Timestamp) -> dict:
    """Report actual points around a flow-selected peak, never an interpolated DOC peak."""
    start, end = peak_date - pd.Timedelta(days=14), peak_date + pd.Timedelta(days=14)
    sub = samples[samples.value.notna() & samples.date_local.between(start, end)]
    days = sub.date_local.drop_duplicates().sort_values()
    including_edges = pd.DatetimeIndex([start, *days, end])
    gaps = pd.Series(including_edges).diff().dt.total_seconds().div(86400).dropna()
    rise = int(sub.date_local.lt(peak_date).sum())
    fall = int(sub.date_local.gt(peak_date).sum())
    near = int(sub.date_local.sub(peak_date).abs().le(pd.Timedelta(days=2)).sum())
    return {
        "n_samples": len(sub), "n_days": len(days),
        "n_before_peak": rise, "n_after_peak": fall, "n_within2days": near,
        "max_gap_with_edges_days": gaps.max(),
        "minimum_abs_peak_distance_days": sub.date_local.sub(peak_date).abs().dt.days.min(),
        "spans_response": len(days) >= 5 and rise >= 2 and fall >= 2 and near >= 1,
    }


def match_campaigns(a: pd.DataFrame, b: pd.DataFrame, receiver: pd.DataFrame, max_hours: float = 12):
    """One use per sample and a bound on the full three-station timestamp span."""
    sources = [f[f.value.notna()].sort_values("timestamp_utc") for f in (a, b)]
    used = [set(), set()]
    rows = []
    for r in receiver[receiver.value.notna()].sort_values("timestamp_utc").itertuples():
        chosen = []
        for source, excluded in zip(sources, used):
            candidates = source[~source.timestamp_utc.isin(excluded)]
            if candidates.empty:
                break
            offsets = candidates.timestamp_utc.sub(r.timestamp_utc).abs().dt.total_seconds()
            chosen.append(candidates.loc[offsets.idxmin()])
        if len(chosen) != 2:
            continue
        times = [r.timestamp_utc, chosen[0].timestamp_utc, chosen[1].timestamp_utc]
        span_hours = (max(times) - min(times)).total_seconds() / 3600
        if span_hours > max_hours:
            continue
        for selected, excluded in zip(chosen, used):
            excluded.add(selected.timestamp_utc)
        rows.append({"receiver_time_utc": r.timestamp_utc,
                     "source_a_time_utc": chosen[0].timestamp_utc,
                     "source_b_time_utc": chosen[1].timestamp_utc,
                     "date_local": r.date_local,
                     "doc_a": chosen[0].value, "doc_b": chosen[1].value,
                     "doc_receiver": r.value, "sampling_span_hours": span_hours})
    return pd.DataFrame(rows)
