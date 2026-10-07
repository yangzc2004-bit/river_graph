"""Measured branch/trunk partitions and matched DOC timing diagnostics."""
from __future__ import annotations

import numpy as np
import pandas as pd


def downstream_route(network, source, target):
    i, end = network.index(source), network.index(target)
    route, seen = [], set()
    while i not in seen:
        seen.add(i)
        route.append(i)
        if i == end:
            return route
        seq = network.downstream[i]
        pos = np.searchsorted(network.sorted_hydro, seq)
        if seq <= 0 or pos == len(network.sorted_hydro) or network.sorted_hydro[pos] != seq:
            break
        i = int(network.hydro_order[pos])
    raise ValueError("no primary downstream connection to receiver")


def partition_footprint(network, a, b, target, weight_a):
    """Station endpoints are (COMID, NHD measure); common reaches counted once."""
    if not np.isfinite(weight_a) or not 0 < weight_a < 1:
        raise ValueError("two positive branch shares required")
    ra, rb = (downstream_route(network, s[0], target[0]) for s in (a, b))
    joint = next(i for i in ra if i in set(rb))
    ia, ib = ra.index(joint), rb.index(joint)
    if ia == 0 or ib == 0:
        raise ValueError("independent upstream station branches required")
    if ra[ia:] != rb[ib:]:
        raise ValueError("common primary suffix must coincide")
    am, aim = network.position(a[1])
    bm, bim = network.position(b[1])
    tm, tim = network.position(target[1])
    rows = []
    for segment, indices, initial in (("branch_a", ra[:ia], am), ("branch_b", rb[:ib], bm),
                                       ("common", ra[ia:], 100.)):
        for k, i in enumerate(indices):
            start = initial if k == 0 else 100.
            end = tm if i == ra[-1] else 0.
            length = network.length[i]*(start-end)/100
            if length < 0:
                raise ValueError("station order gives negative path length")
            rows.append({"segment": segment, "sequence": k, "comid": int(network.comid[i]),
                "start_measure": start, "end_measure": end, "length_km": length,
                "storage": bool(network.storage[i]), "stream_order": network.order[i],
                "lateral_junction": bool(i not in (ra[0], rb[0], joint) and len(network.parents(i)) >= 2)})
    f = pd.DataFrame(rows)
    if f.comid.duplicated().any():
        raise ValueError("unique corridor reaches required")
    lengths = f.groupby("segment").length_km.sum()
    la, lb, common = (float(lengths[s]) for s in ("branch_a", "branch_b", "common"))
    mean_branch = weight_a*la+(1-weight_a)*lb
    mean_total = mean_branch+common
    if mean_total <= 0 or mean_branch <= 0:
        raise ValueError("positive independent-branch and total path means required")
    sd = np.sqrt(weight_a*(1-weight_a))*abs(la-lb)
    metrics = {"junction_comid": int(network.comid[joint]), "branch_a_km": la, "branch_b_km": lb,
        "common_km": common, "path_a_km": la+common, "path_b_km": lb+common,
        "unique_corridor_km": la+lb+common, "mean_branch_km": mean_branch, "mean_total_km": mean_total,
        "common_fraction": common/mean_total, "independent_branch_cv": sd/mean_branch,
        "total_path_cv": sd/mean_total, "branch_balance": 4*weight_a*(1-weight_a),
        "n_corridor_reaches": len(f), "station_measure_imputed": aim or bim or tim}
    for segment in ("branch_a", "branch_b", "common"):
        s = f[f.segment.eq(segment)]
        metrics[segment+"_storage_fraction"] = s.loc[s.storage, "length_km"].sum()/s.length_km.sum() if s.length_km.sum() > 0 else np.nan
        metrics[segment+"_lateral_junctions"] = int(s.lateral_junction.sum())
    return metrics, f


def matched_pulses(path_a, path_b, common, weight, *, sigma=.15, dt=.0025):
    """Shared translation isolated from shape on an identical centered grid."""
    if not np.isfinite([path_a, path_b, common, weight, sigma, dt]).all():
        raise ValueError("finite geometry and forcing required")
    if min(path_a, path_b) <= 0 or not 0 <= common <= min(path_a, path_b) or not 0 < weight < 1 or min(sigma, dt) <= 0:
        raise ValueError("causal paths, positive shares and forcing required")
    weights = np.array([weight, 1-weight])
    mean = np.dot(weights, [path_a, path_b])
    actual = np.array([path_a, path_b])/mean
    removed = actual-common/mean
    relative = actual-1.
    steps = int(np.ceil((np.max(abs(relative))+8*sigma)/dt))
    centered = np.arange(-steps, steps+1)*dt
    rows, curves = [], []
    for name, delay in (("actual", actual), ("equal_branches", np.ones(2)), ("no_common_trunk", removed)):
        centroid = np.dot(weights, delay)
        offsets = delay-centroid
        pulse = np.exp(-.5*((centered[:, None]-offsets)/sigma)**2)@weights
        integrated = np.trapezoid(pulse, centered)/(np.sqrt(2*np.pi)*sigma)
        rows.append({"scenario": name, "pulse_peak": pulse.max(), "pulse_centroid": centroid,
            "pulse_sd": np.sqrt(sigma**2+np.dot(weights, offsets**2)),
            "anomaly_mass_fraction": integrated, "steady_doc": 5., "mean_delay": centroid})
        curves.append(pd.DataFrame({"scenario": name, "centered_time": centered,
            "relative_time": centered+centroid, "outlet_anomaly": pulse}))
    return pd.DataFrame(rows), pd.concat(curves, ignore_index=True)


def timing_components(frame, max_path):
    """Same source information; current/previous interpolation decomposed exactly."""
    columns = ("doc_a_now", "doc_b_now", "doc_a_previous", "doc_b_previous", "weight_a", "path_a_km", "path_b_km", "common_km")
    values = frame[list(columns)].to_numpy(float)
    if not np.isfinite(values).all() or not np.isfinite(max_path) or max_path <= 0:
        raise ValueError("finite known source values and positive geometric scale required")
    a, b, ap, bp, w, la, lb, common = values.T
    if (values[:, :4] < 0).any() or ((w <= 0) | (w >= 1)).any() or (common < 0).any() or (common > np.minimum(la, lb)+1e-9).any() or (np.maximum(la, lb) > max_path+1e-9).any():
        raise ValueError("positive shares and causal measured source paths required")
    now = w*a+(1-w)*b
    previous = w*ap+(1-w)*bp
    mean = w*la+(1-w)*lb
    shared = common/max_path*(previous-now)
    equal_branch = (mean-common)/max_path*(previous-now)
    differential = w*(1-w)*(la-lb)/max_path*((ap-a)-(bp-b))
    out = frame.copy()
    out["same_month_proxy"] = now
    out["shared_delay_input"] = shared
    out["equal_branch_delay_input"] = equal_branch
    out["differential_arrival_input"] = differential
    out["common_only_proxy"] = now+shared
    out["mean_delay_proxy"] = now+shared+equal_branch
    out["actual_branch_proxy"] = now+shared+equal_branch+differential
    return out


def geometry_examples(frame):
    """Occupied geometric quadrants; no DOC values select actual corridors."""
    f = frame.copy()
    cut = {"branch_cv": float(f.independent_branch_cv.median()),
           "common_fraction": float(f.common_fraction.median())}
    f["example_group"] = 1+f.independent_branch_cv.ge(cut["branch_cv"]).astype(int)+2*f.common_fraction.ge(cut["common_fraction"]).astype(int)
    f["log_mean_path"] = np.log1p(f.mean_total_km)
    columns = ["independent_branch_cv", "common_fraction", "log_mean_path"]
    scale = f[columns].std(ddof=0).to_numpy()
    scale = np.where(scale > 0, scale, 1.)
    rows = []
    for group, s in f.groupby("example_group"):
        center = s[columns].median().to_numpy()
        distance = np.linalg.norm((s[columns].to_numpy()-center)/scale, axis=1)
        s = s.assign(example_distance=distance).sort_values(["example_distance", "pair_id"])
        rows.append({"example_group": int(group), "pair_id": s.pair_id.iloc[0], "n_connections": len(s)})
    return pd.DataFrame(rows), cut


def cropped_corridor(reaches, projected_lines):
    """Orient mapped reaches by downstream connectivity, then crop station ends.

    NHD measures run from 100 at the reach head to 0 at its outlet. Coordinate
    order is inferred independently and can run in either direction in the cache.
    Geometry is already projected in metres; VAA lengths define analysis metrics.
    """
    from shapely.geometry import LineString
    from shapely.ops import linemerge, substring

    lines = {}
    for cid in reaches.comid:
        line = projected_lines[int(cid)]
        if line.geom_type == "MultiLineString":
            line = linemerge(line)
        if line.geom_type != "LineString" or line.is_empty or line.length <= 0:
            raise ValueError("connected nonempty line geometry required")
        lines[int(cid)] = line
    common = reaches[reaches.segment.eq("common")].sort_values("sequence").comid.tolist()
    oriented, gaps = {}, []
    for branch in ("branch_a", "branch_b"):
        route = reaches[reaches.segment.eq(branch)].sort_values("sequence").comid.tolist()+common
        for k, cid in enumerate(route):
            xy = np.asarray(lines[cid].coords)[:, :2]
            if k < len(route)-1:
                nxt = np.asarray(lines[route[k+1]].coords)[[0, -1], :2]
                distance = np.linalg.norm(xy[[0, -1], None, :]-nxt[None, :, :], axis=2).min(axis=1)
                reverse = distance[0] < distance[1]
            else:
                prev = np.asarray(oriented[route[k-1]].coords)[-1, :2]
                distance = np.linalg.norm(xy[[0, -1]]-prev, axis=1)
                reverse = distance[1] < distance[0]
            candidate = LineString(xy[::-1] if reverse else xy)
            if cid in oriented and not candidate.equals_exact(oriented[cid], 1e-8):
                raise ValueError("common trunk orientation disagrees between branches")
            oriented[cid] = candidate
            if k:
                gaps.append(np.linalg.norm(np.asarray(oriented[route[k-1]].coords)[-1, :2]-np.asarray(candidate.coords)[0, :2]))
    pieces = {}
    for r in reaches.itertuples():
        line = oriented[r.comid]
        pieces[int(r.comid)] = substring(line, (100-r.start_measure)/100*line.length,
                                        (100-r.end_measure)/100*line.length)
    return pieces, float(max(gaps, default=0.))
