"""Real full-network topology and metre-based river planform measurements."""
from __future__ import annotations

import numpy as np
from pyproj import Transformer
from shapely import from_wkb, make_valid
from shapely import transform as transform_array
from shapely.geometry import LineString, shape
from shapely.ops import transform

COLUMNS = ("comid", "hydroseq", "dnhydroseq", "dnminorhyd", "arbolatesu", "lengthkm",
           "streamorde", "totdasqkm", "divergence", "fcode")
PROJECT = Transformer.from_crs("EPSG:4326", "EPSG:5070", always_xy=True).transform


class UpstreamNetwork:
    """VAA upstream sets including primary and secondary downstream connections."""

    def __init__(self, frame):
        self.frame = frame.reset_index(drop=True)
        if self.frame.comid.duplicated().any() or self.frame.hydroseq.duplicated().any():
            raise ValueError("unique VAA COMIDs/hydroseq required")
        self.comid = self.frame.comid.to_numpy(np.int64)
        self.hydro = self.frame.hydroseq.to_numpy(np.int64)
        self.downstream = self.frame.dnhydroseq.to_numpy(np.int64)
        self.order = np.argsort(self.comid)
        self.sorted_comid = self.comid[self.order]
        self.parent_order = np.argsort(self.downstream, kind="stable")
        self.sorted_downstream = self.downstream[self.parent_order]
        minor = self.frame.dnminorhyd.fillna(0).to_numpy(np.int64) if "dnminorhyd" in self.frame else np.zeros(len(self.frame), dtype=np.int64)
        positions = np.flatnonzero(minor > 0)
        self.minor_order = positions[np.argsort(minor[positions], kind="stable")]
        self.sorted_minor = minor[self.minor_order]
        self.cumulative = self.frame.arbolatesu.to_numpy(float)
        self.area = self.frame.totdasqkm.to_numpy(float)

    def index(self, comid):
        p = np.searchsorted(self.sorted_comid, int(comid))
        if p == len(self.comid) or self.sorted_comid[p] != int(comid):
            raise ValueError(f"COMID not found: {comid}")
        return int(self.order[p])

    def parents(self, index):
        seq = self.hydro[index]
        a, b = np.searchsorted(self.sorted_downstream, seq, side="left"), np.searchsorted(self.sorted_downstream, seq, side="right")
        c, d = np.searchsorted(self.sorted_minor, seq, side="left"), np.searchsorted(self.sorted_minor, seq, side="right")
        return np.unique(np.concatenate([self.parent_order[a:b], self.minor_order[c:d]]))

    def membership(self, comid):
        root = self.index(comid)
        seen, pending = set(), [root]
        while pending:
            i = pending.pop()
            if i in seen:
                continue
            seen.add(i)
            pending.extend(self.parents(i).tolist())
        indices = np.asarray(sorted(seen), dtype=int)
        main, visited, i = [], set(), root
        while i not in visited:
            visited.add(i)
            main.append(i)
            parents = self.parents(i)
            if not len(parents):
                break
            i = int(parents[np.argmax(self.cumulative[parents])])
        if len(parents) and i in visited:
            raise ValueError("cycle in mainstem topology")
        return indices, np.asarray(main, dtype=int)


def project_geometry(geometry):
    geo = geometry if hasattr(geometry, "geom_type") else shape(geometry)
    return transform(PROJECT, geo)


def project_lines(lines, *, chunk=1000):
    """Same CRS operation as project_geometry, vectorized in bounded chunks."""
    keys = list(lines)
    projected = {}
    for start in range(0, len(keys), chunk):
        part = keys[start:start+chunk]
        values = np.asarray([lines[cid] for cid in part], dtype=object)
        geoms = transform_array(values, lambda coords: np.column_stack(PROJECT(coords[:, 0], coords[:, 1])))
        projected.update(zip(part, geoms, strict=True))
    return projected


def _line_segments(line):
    pieces = list(line.geoms) if line.geom_type == "MultiLineString" else [line]
    midpoint, lengths, directions = [], [], []
    for piece in pieces:
        coords = np.asarray(piece.coords, float)[:, :2]
        delta = np.diff(coords, axis=0)
        length = np.linalg.norm(delta, axis=1)
        valid = length > 0
        midpoint.append((coords[1:]+coords[:-1])[valid]/2)
        lengths.append(length[valid])
        directions.append(delta[valid])
    return np.concatenate(midpoint), np.concatenate(lengths), np.concatenate(directions)


def metric_geometry_features(lines, basin, mainstem, *, outlet=None):
    """Shapes in metre coordinates; mainstem COMIDs ordered outlet to headwater."""
    basin = make_valid(basin)
    if basin.is_empty or basin.area <= 0 or len(mainstem) == 0:
        raise ValueError("positive-area basin and mainstem required")
    if set(mainstem)-set(lines):
        raise ValueError("mainstem geometry missing")
    rectangle = np.asarray(basin.minimum_rotated_rectangle.exterior.coords)[:4, :2]
    sides = np.linalg.norm(np.roll(rectangle, -1, axis=0)-rectangle, axis=1)
    long_side, short_side = float(sides.max()), float(sides.min())
    if short_side <= 0:
        raise ValueError("degenerate basin extent")
    positions, lengths, vectors, ids = [], [], [], []
    for comid, line in lines.items():
        p, w, v = _line_segments(line)
        positions.append(p)
        lengths.append(w)
        vectors.append(v)
        ids.append(np.full(len(w), comid, dtype=np.int64))
    p, w, v, ids = map(np.concatenate, (positions, lengths, vectors, ids))
    if not len(w) or w.sum() <= 0:
        raise ValueError("positive channel length required")
    centre = np.average(p, weights=w, axis=0)
    delta = p-centre
    covariance = (delta*w[:, None]).T@delta/w.sum()
    eigenvalues = np.linalg.eigvalsh(covariance)
    ratio = np.sqrt(eigenvalues[-1]/eigenvalues[0]) if eigenvalues[0] > 1e-8 else np.nan
    # NHD geometries can be oriented either way. Stitch endpoints by proximity,
    # beginning with the receiving reach's endpoint nearest the downstream end
    # of the next mainstem reach. Interior line curvature remains untouched.
    main_coords = []
    for k, cid in enumerate(mainstem):
        line = lines[cid]
        if line.geom_type != "LineString":
            raise ValueError("mainstem requires single flowline geometries")
        coords = np.asarray(line.coords, float)[:, :2]
        if k == 0:
            if outlet is not None:
                if np.linalg.norm(coords[-1]-outlet) < np.linalg.norm(coords[0]-outlet):
                    coords = coords[::-1]
            elif len(mainstem) > 1:
                following = np.asarray(lines[mainstem[1]].coords, float)[[0, -1], :2]
                # The endpoint nearer the next upstream reach goes LAST.
                if np.linalg.norm(following-coords[0], axis=1).min() < np.linalg.norm(following-coords[-1], axis=1).min():
                    coords = coords[::-1]
        elif np.linalg.norm(coords[-1]-main_coords[-1][-1]) < np.linalg.norm(coords[0]-main_coords[-1][-1]):
            coords = coords[::-1]
        main_coords.append(coords)
    connections = [np.linalg.norm(main_coords[i][-1]-main_coords[i+1][0]) for i in range(len(main_coords)-1)]
    coords = np.concatenate(main_coords)
    main_length = sum(lines[cid].length for cid in mainstem)
    chord = float(np.linalg.norm(coords[-1]-coords[0]))
    axis = coords[-1]-coords[0]
    trib = ~np.isin(ids, mainstem)
    offset = p-coords[0]
    side = axis[0]*offset[:, 1]-axis[1]*offset[:, 0]
    left, right = w[trib & (side > 0)].sum(), w[trib & (side < 0)].sum()
    angle = np.arctan2(v[trib, 1], v[trib, 0])
    alignment = abs(np.sum(w[trib]*np.exp(2j*angle)))/w[trib].sum() if w[trib].sum() > 0 else np.nan
    return {"basin_area_km2": basin.area/1e6, "basin_aspect": long_side/short_side,
            "basin_elongation": 2*np.sqrt(basin.area/np.pi)/long_side,
            "basin_compactness": 4*np.pi*basin.area/basin.length**2,
            "network_axis_ratio": ratio, "channel_length_km": w.sum()/1000,
            "drainage_density": w.sum()/1000/(basin.area/1e6),
            "mainstem_length_km": main_length/1000, "mainstem_share": main_length/w.sum(),
            "mainstem_sinuosity": main_length/chord if chord > 0 else np.nan,
            "side_imbalance": abs(left-right)/(left+right) if left+right > 0 else np.nan,
            "tributary_alignment": float(alignment), "mainstem_gap_max_m": max(connections, default=0),
            "outlet_x": float(coords[0, 0]), "outlet_y": float(coords[0, 1])}


def network_features(network, indices, main, geographic_lines, basin):
    lines = project_lines(geographic_lines)
    ids = network.comid[indices]
    if set(map(int, ids)) != set(lines):
        raise ValueError("complete expected upstream-network geometry required")
    result = metric_geometry_features(lines, project_geometry(basin), network.comid[main].tolist())
    selected = set(indices.tolist())
    junctions, balance, along = [], [], []
    cumulative = 0.
    for i in indices:
        parents = [p for p in network.parents(i) if int(p) in selected]
        if len(parents) >= 2:
            junctions.append(int(i))
            area = network.area[parents]
            if np.isfinite(area).all() and area.sum() > 0:
                balance.append(1-area.max()/area.sum())
    junction_set = set(junctions)
    for i in main:
        length = lines[int(network.comid[i])].length
        if int(i) in junction_set:
            along.append((cumulative+length/2)/(1000*result["mainstem_length_km"]))
        cumulative += length
    result.update(n_reaches=len(ids), n_junctions=len(junctions),
        junction_frequency=len(junctions)/result["basin_area_km2"],
        hierarchy_order=float(network.frame.iloc[indices].streamorde.max()),
        tributary_balance=float(np.median(balance)) if balance else 0.,
        confluence_position=float(np.mean(along)) if along else np.nan,
        n_mainstem_junctions=len(along), n_divergent_reaches=int(network.frame.iloc[indices].divergence.eq(2).sum()))
    return result


def line_from_geojson(feature):
    geom = shape(feature["geometry"])
    if geom.geom_type not in ("LineString", "MultiLineString") or geom.is_empty:
        raise ValueError("nonempty flowline geometry required")
    return geom


def read_cached_lines(connection, comids):
    """Read shared physical geometry by COMID, avoiding SQLite variable limits."""
    result = {}
    for start in range(0, len(comids), 500):
        ids = [int(v) for v in comids[start:start+500]]
        placeholders = ",".join("?" for _ in ids)
        for cid, wkb in connection.execute(f"SELECT comid, wkb FROM lines WHERE comid IN ({placeholders})", ids):
            result[cid] = from_wkb(wkb)
    return result


def straight_toy_network():
    """Analytical fixture: three connected lines in metre coordinates."""
    return {1: LineString([(0, 0), (0, 100)]),
            2: LineString([(0, 100), (-100, 200)]),
            3: LineString([(0, 100), (100, 200)])}
