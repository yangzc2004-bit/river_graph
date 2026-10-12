"""Label-boundary-safe, retrospective DOC interpolation experiments."""
from __future__ import annotations

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.spatial import cKDTree

RECIPES = {
    "seasonal_median": [{"id": "station_season"}],
    "nearest_spatial": [{"id": "nearest", "k": 1, "power": 2.}],
    "euclidean_idw": [{"id": f"k{k}_p{p}", "k": k, "power": float(p)}
                       for k in (5, 15) for p in (1, 2)],
    "river_idw": [{"id": f"k{k}_p{p}", "k": k, "power": float(p)}
                   for k in (5, 15) for p in (1, 2)],
    "spatiotemporal_idw": [{"id": f"lag{lag}_t{scale}", "k": 30,
                            "power": 2., "max_lag": lag, "time_scale": scale,
                            "space_scale": 100.}
                           for lag in (6, 12) for scale in (1., 3.)],
    "ordinary_kriging": [{"id": f"range{distance}_n{nugget}", "range": distance,
                          "nugget": nugget, "k": 30}
                         for distance in (100., 300., 1000.) for nugget in (.1, .5)],
    "spatiotemporal_kriging": [{"id": f"r{distance}_t{scale}", "k": 20,
                                "space_scale": distance, "time_scale": scale,
                                "max_lag": lag, "nugget": .2}
                               for distance in (100., 300.) for scale, lag in ((1., 6), (3., 12))],
    "linear_time": [{"id": "linear"}],
    "pchip_time": [{"id": "pchip"}],
}


def temporal_masks(observed, seed, lengths=(3, 6, 12)):
    """Fixed core queries; larger gaps remove more context, never change truth.

    Select intervals using observation dates only. Each station has distinct
    validation/test 3-month cores. Require brackets within 12 months outside
    the largest gap, and >=6 retained training observations. One core per role
    per eligible station avoids pretending sparse natural cells have truth.
    """
    observed = np.asarray(observed, bool)
    n, t = observed.shape
    rng = np.random.default_rng(seed)
    selected = []
    for node in range(n):
        dates = np.flatnonzero(observed[node])
        if len(dates) < 12:
            continue
        candidates = []
        for center in dates:
            left, right = center-5, center+6
            if left < 0 or right >= t:
                continue
            before = dates[(dates < left) & (dates >= left-12)]
            after = dates[(dates > right) & (dates <= right+12)]
            if len(before) and len(after):
                candidates.append(int(center))
        if len(candidates) < 2:
            continue
        pairs = [(a, b) for a in candidates for b in candidates
                 if a < b and b-a >= 24 and
                 np.count_nonzero((dates < a-5) | ((dates > a+6) & (dates < b-5)) | (dates > b+6)) >= 6]
        if not pairs:
            continue
        a, b = pairs[int(rng.integers(len(pairs)))]
        val, test = (a, b) if rng.integers(2) else (b, a)
        selected.append((node, val, test))
    masks = {}
    core = {role: [] for role in ("val", "test")}
    for node, val, test in selected:
        for role, center in (("val", val), ("test", test)):
            core[role].extend(node*t + np.flatnonzero(observed[node] &
                              (np.arange(t) >= center-1) & (np.arange(t) <= center+1)))
    for length in lengths:
        visible = observed.copy()
        excluded = np.zeros_like(observed)
        for node, val, test in selected:
            for center in (val, test):
                left = center-(length-1)//2
                right = left+length
                excluded[node, left:right] = observed[node, left:right]
                visible[node, left:right] = False
        masks[length] = {"train": np.flatnonzero(visible.ravel()),
                         "val": np.sort(np.asarray(core["val"], dtype=np.int64)),
                         "test": np.sort(np.asarray(core["test"], dtype=np.int64)),
                         "excluded": np.flatnonzero(excluded.ravel()),
                         "eligible_rows": np.asarray([item[0] for item in selected], dtype=np.int64),
                         "intervals": np.asarray(selected, dtype=np.int64).reshape(-1, 3)}
    return masks


def concentration(raw):
    return np.clip(np.asarray(raw, float), 0., np.expm1(15.))


class Interpolator:
    """Native-concentration interpolation; hidden labels are never received."""

    def __init__(self, train_cells, train_y, shape, euclidean, river):
        self.cells = np.asarray(train_cells, dtype=np.int64)
        self.values = np.asarray(train_y, float)
        self.n, self.t = shape
        if len(self.cells) != len(self.values) or not len(self.cells):
            raise ValueError("nonempty aligned support required")
        if len(np.unique(self.cells)) != len(self.cells) or not np.isfinite(self.values).all():
            raise ValueError("unique finite observed support required")
        self.nodes, self.times = self.cells//self.t, self.cells % self.t
        self.euclidean = np.asarray(euclidean, float)
        self.river = np.asarray(river, float)
        self.by_month = {int(m): np.flatnonzero(self.times == m) for m in np.unique(self.times)}
        self.by_node = {int(s): np.flatnonzero(self.nodes == s) for s in np.unique(self.nodes)}
        self.global_median = float(np.median(self.values))
        self.month_medians = {m: float(np.median(self.values[self.times % 12 == m]))
                              for m in np.unique(self.times % 12)}
        self.station_medians = {s: float(np.median(self.values[ids])) for s, ids in self.by_node.items()}
        self.station_seasons = {(s, int(m)): float(np.median(self.values[ids[self.times[ids] % 12 == m]]))
                                for s, ids in self.by_node.items() for m in np.unique(self.times[ids] % 12)}

    def fallback(self, cells):
        return np.asarray([self.station_seasons.get((int(c//self.t), int(c % self.t % 12)),
            self.station_medians.get(int(c//self.t), self.month_medians.get(int(c % self.t % 12), self.global_median)))
            for c in cells])

    def predict(self, family, recipe, cells):
        cells = np.asarray(cells, np.int64)
        if np.intersect1d(cells, self.cells).size:
            raise ValueError("query intersects visible DOC support")
        nodes, times = cells//self.t, cells % self.t
        raw = np.full(len(cells), np.nan)
        if family == "seasonal_median":
            raw = self.fallback(cells)
        elif family in ("linear_time", "pchip_time"):
            for node in np.unique(nodes):
                support = self.by_node.get(int(node), np.array([], int))
                if len(support) < 2:
                    continue
                order = np.argsort(self.times[support])
                x, y = self.times[support][order], self.values[support][order]
                ids = np.flatnonzero((nodes == node) & (times >= x[0]) & (times <= x[-1]))
                raw[ids] = (np.interp(times[ids], x, y) if family == "linear_time"
                            else PchipInterpolator(x, y, extrapolate=False)(times[ids]))
        elif family in ("spatiotemporal_idw", "spatiotemporal_kriging"):
            # A short temporal window bounds evidence; space is ECEF chord km.
            xyz = _coordinates(self.euclidean)
            for month in np.unique(times):
                support = np.flatnonzero(np.abs(self.times-month) <= recipe["max_lag"])
                if not len(support):
                    continue
                query = np.flatnonzero(times == month)
                points = np.column_stack((xyz[self.nodes[support]]/recipe["space_scale"],
                                          (self.times[support]-month)/recipe["time_scale"]))
                tree = cKDTree(points)
                qpoints = np.column_stack((xyz[nodes[query]]/recipe["space_scale"], np.zeros(len(query))))
                distances, neighbours = tree.query(qpoints, k=min(recipe["k"], len(support)))
                if family == "spatiotemporal_idw":
                    raw[query] = _weighted(distances, self.values[support][neighbours], recipe["power"])
                elif len(support) >= 3:
                    if neighbours.ndim == 1:
                        neighbours = neighbours[:, None]
                    for local, i in enumerate(query):
                        ids = neighbours[local]
                        pair = np.linalg.norm(points[ids][:, None]-points[ids][None, :], axis=-1)
                        covariance = np.exp(-pair)+np.eye(len(ids))*recipe["nugget"]
                        cross = np.exp(-np.linalg.norm(points[ids]-qpoints[local], axis=-1))
                        raw[i] = _ordinary(covariance, cross, self.values[support][ids])
        else:
            for month in np.unique(times):
                support = self.by_month.get(int(month), np.array([], int))
                if not len(support):
                    continue
                for i in np.flatnonzero(times == month):
                    matrix = self.river if family == "river_idw" else self.euclidean
                    distances = matrix[nodes[i], self.nodes[support]]
                    good = np.flatnonzero(np.isfinite(distances))
                    if not len(good):
                        continue
                    selected = good[np.argsort(distances[good], kind="stable")[:recipe["k"]]]
                    ids, d = support[selected], distances[selected]
                    if family == "ordinary_kriging":
                        if len(ids) < 3:
                            continue
                        pair = self.euclidean[np.ix_(self.nodes[ids], self.nodes[ids])]
                        covariance = np.exp(-pair/recipe["range"])+np.eye(len(ids))*recipe["nugget"]
                        cross = np.exp(-d/recipe["range"])
                        raw[i] = _ordinary(covariance, cross, self.values[ids])
                    else:
                        raw[i] = _weighted(d[None, :], self.values[ids][None, :], recipe["power"])[0]
        intrinsic = np.isfinite(raw)
        complete = concentration(np.where(intrinsic, raw, self.fallback(cells)))
        return complete, intrinsic, raw


def _weighted(distance, values, power):
    distance, values = np.asarray(distance), np.asarray(values)
    if distance.ndim == 1:
        distance, values = distance[:, None], values[:, None]
    weights = np.maximum(distance, 1e-6)**(-power)
    zero = distance <= 1e-6
    weights = np.where(zero.any(1)[:, None], zero.astype(float), weights)
    return (weights*values).sum(1)/weights.sum(1)


def _ordinary(covariance, cross, values):
    try:
        solved = np.linalg.solve(covariance, np.column_stack((cross, np.ones(len(values)))))
        weights = solved[:, 0]+solved[:, 1]*(1-solved[:, 0].sum())/solved[:, 1].sum()
        return float(weights @ values)
    except np.linalg.LinAlgError:
        return np.nan


def _coordinates(distances):
    """Exact Euclidean embedding of the recorded ECEF chord distances."""
    if not hasattr(_coordinates, "cache") or _coordinates.cache[0] is not distances:
        gram = -.5*(distances**2-distances[:, :1]**2-distances[:1, :]**2)
        eigen, vectors = np.linalg.eigh(gram)
        xyz = vectors[:, -3:]*np.sqrt(np.maximum(eigen[-3:], 0.))
        _coordinates.cache = (distances, xyz)
    return _coordinates.cache[1]
