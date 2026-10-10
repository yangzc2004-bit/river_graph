"""Local geometry of mapped confluences, separate from whole-network form."""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pandas as pd
from shapely.geometry import LineString
from shapely.ops import substring

from river_graph.analysis.river_monitored_footprint import cropped_corridor

METRICS = (
    "incoming_angle_deg",
    "branch_a_deflection_deg",
    "branch_b_deflection_deg",
    "area_weighted_deflection_deg",
    "downstream_sinuosity",
    "downstream_turning_deg",
)


def angle_degrees(a, b):
    """Unsigned angle between two nonzero finite planar directions."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != (2,) or b.shape != (2,) or not np.isfinite([a, b]).all():
        raise ValueError("two finite planar vectors required")
    norm = np.linalg.norm(a)*np.linalg.norm(b)
    if norm <= 1e-12:
        raise ValueError("directions must be nonzero")
    return float(np.degrees(np.arccos(np.clip(a@b/norm, -1., 1.))))


def outward_segments(reaches, pieces):
    """Branches point away from the junction; the common channel points downstream."""
    output = {}
    for segment in ("branch_a", "branch_b", "common"):
        group = reaches[reaches.segment.eq(segment)].sort_values("sequence")
        if group.empty:
            raise ValueError("two incoming branches and one downstream corridor required")
        if segment != "common":
            group = group.iloc[::-1]
        parts = []
        for cid in group.comid:
            line = pieces[int(cid)]
            if line.geom_type == "Point":
                continue
            coords = np.asarray(line.coords)[:, :2]
            parts.append(LineString(coords[::-1] if segment != "common" else coords))
        if not parts:
            raise ValueError("positive mapped length required for each corridor")
        output[segment] = parts
    return output


def local_line(parts, distance, *, gap_tolerance=20.):
    """Read only the supported local length, with explicit mapped gap checks.

    Small digitization gaps are bridged and their lengths retained in the
    centreline distance. Gaps farther than the requested window are irrelevant.
    A short route stays short; no extrapolation creates a direction.
    """
    if not np.isfinite([distance, gap_tolerance]).all() or distance <= 0 or gap_tolerance < 0:
        raise ValueError("positive distance and nonnegative map gap tolerance required")
    coords, length, maximum_gap = [], 0., 0.
    for part in parts:
        points = np.asarray(part.coords)[:, :2]
        if not np.isfinite(points).all() or part.length <= 0:
            raise ValueError("finite positive connected line pieces required")
        if coords:
            gap = float(np.linalg.norm(np.asarray(coords[-1])-points[0]))
            maximum_gap = max(maximum_gap, gap)
            if gap > gap_tolerance:
                return None, "mapped_gap_exceeds_tolerance", maximum_gap
            length += gap
        coords.extend(points.tolist())
        length += part.length
        if length >= distance-1e-8:
            line = LineString(coords)
            clipped = substring(line, 0., min(distance, line.length))
            if clipped.geom_type != "LineString" or clipped.length < distance-1e-7:
                raise ValueError("local clipping must preserve the requested length")
            return clipped, "measured", maximum_gap
    return None, "mapped_route_shorter_than_scale", maximum_gap


def line_direction(line):
    xy = np.asarray(line.coords)[:, :2]
    return xy[-1]-xy[0]


def total_turning(line, *, step=25.):
    """Absolute direction changes on evenly spaced 25 m centreline chords."""
    if not np.isfinite(step) or step <= 0:
        raise ValueError("positive sampling interval required")
    distances = np.r_[np.arange(0., line.length-1e-7, step), line.length]
    xy = np.array([line.interpolate(float(d)).coords[0][:2] for d in distances])
    vectors = np.diff(xy, axis=0)
    if (np.linalg.norm(vectors, axis=1) <= 1e-10).any():
        raise ValueError("sampled centreline has zero displacement chords")
    return sum(angle_degrees(a, b) for a, b in pairwise(vectors))


def measure_junction(reaches, projected_lines, weight_a, *, scales=(100., 250., 500.),
                     gap_tolerance=20., turning_step=25.):
    """Measure incoming angle, flow-direction turn and downstream bending.

    `weight_a` is a contributing-area fraction, never a measured discharge or
    width fraction. Each scale uses an exact mapped arclength in metres.
    """
    if not np.isfinite(weight_a) or not 0 < weight_a < 1:
        raise ValueError("positive contributing-area shares required")
    scales = np.asarray(scales, float)
    if scales.ndim != 1 or not len(scales) or len(np.unique(scales)) != len(scales) or not np.isfinite(scales).all() or (scales <= 0).any():
        raise ValueError("distinct positive finite measurement scales required")
    pieces, corridor_gap = cropped_corridor(reaches, projected_lines)
    segments = outward_segments(reaches, pieces)
    roots = {s: np.asarray(lines[0].coords)[0, :2] for s, lines in segments.items()}
    join_gap = max(np.linalg.norm(roots[s]-roots["common"]) for s in ("branch_a", "branch_b"))
    output, local = [], {}
    for distance in scales:
        row = {"scale_m": float(distance), "junction_gap_m": float(join_gap),
               "full_corridor_max_gap_m": corridor_gap}
        lines = {}
        for segment, parts in segments.items():
            line, status, gap = local_line(parts, distance, gap_tolerance=gap_tolerance)
            row[segment+"_status"], row[segment+"_max_gap_m"] = status, gap
            if line is not None:
                lines[segment] = line
        row["status"] = "measured" if len(lines) == 3 else "incomplete_local_geometry"
        if join_gap > gap_tolerance:
            row["status"] = "junction_gap_exceeds_tolerance"
        row.update({metric: np.nan for metric in METRICS})
        if row["status"] == "measured":
            directions = {s: line_direction(line) for s, line in lines.items()}
            if any(np.linalg.norm(d) <= 1e-10 for d in directions.values()):
                row["status"] = "zero_direction_chord"
            else:
                row["incoming_angle_deg"] = angle_degrees(directions["branch_a"], directions["branch_b"])
                for branch in ("branch_a", "branch_b"):
                    row[branch+"_deflection_deg"] = angle_degrees(-directions[branch], directions["common"])
                row["area_weighted_deflection_deg"] = weight_a*row["branch_a_deflection_deg"]+(1-weight_a)*row["branch_b_deflection_deg"]
                row["downstream_sinuosity"] = lines["common"].length/np.linalg.norm(directions["common"])
                row["downstream_turning_deg"] = total_turning(lines["common"], step=turning_step)
                local[float(distance)] = lines
        output.append(row)
    return pd.DataFrame(output), local


def paired_form_contrasts(geometry, pairs, *, draws=5000, seed=42):
    """Original elongated/broad matches; resample entire HUC4 pair groups."""
    if draws < 1 or geometry.duplicated(["station", "scale_m"]).any():
        raise ValueError("positive draws and unique station-scale geometry required")
    if {"class_a", "class_b"}.issubset(pairs) and not (pairs.class_a.eq(1) & pairs.class_b.eq(3)).all():
        raise ValueError("this contrast requires original elongated/broad pairs only")
    rows, details = [], []
    for scale, f in geometry.groupby("scale_m", sort=True):
        f = f[f.status.eq("measured")].set_index("station")
        kept = pairs[pairs.station_a.isin(f.index) & pairs.station_b.isin(f.index)]
        for metric in METRICS:
            values = f.loc[kept.station_b, metric].to_numpy()-f.loc[kept.station_a, metric].to_numpy()
            info = kept.copy()
            info["scale_m"], info["metric"], info["broad_minus_elongated"] = scale, metric, values
            if "junction_comid" in f:
                info["junction_a_comid"] = f.loc[kept.station_a, "junction_comid"].to_numpy()
                info["junction_b_comid"] = f.loc[kept.station_b, "junction_comid"].to_numpy()
                info["same_physical_junction"] = info.junction_a_comid.eq(info.junction_b_comid)
            details.append(info)
            groups = [values[kept.huc4.eq(h).to_numpy()] for h in sorted(kept.huc4.unique())]
            if not groups:
                rows.append({"scale_m": scale, "metric": metric, "n_pairs": 0, "n_huc4": 0,
                             "mean_difference": np.nan, "ci_low": np.nan, "ci_high": np.nan})
                continue
            rng = np.random.default_rng(seed)
            samples = [np.concatenate([groups[i] for i in selected]).mean()
                       for selected in rng.integers(0, len(groups), size=(draws, len(groups)))]
            lo, hi = np.quantile(samples, [.025, .975]) if len(groups) >= 2 else (np.nan, np.nan)
            rows.append({"scale_m": scale, "metric": metric, "n_pairs": len(kept), "n_huc4": len(groups),
                         "mean_difference": values.mean(), "ci_low": lo, "ci_high": hi})
    return pd.DataFrame(rows), pd.concat(details, ignore_index=True)
