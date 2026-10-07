"""Physical reach neighbourhoods, distinct from the sampled station graph."""
from __future__ import annotations

import heapq

import numpy as np
import pandas as pd

VAA_COLUMNS = (
    "comid", "fromnode", "tonode", "hydroseq", "dnhydroseq", "lengthkm",
    "streamorde", "startflag", "divergence", "totdasqkm", "slope",
    "wbareatype", "wbareacomi",
)
PROFILES = {
    "low_order": "Low-order tributary",
    "chain": "Chain conveyance",
    "confluence": "Major confluence vicinity",
    "mainstem": "Integrated mainstem",
    "storage": "Lake / reservoir path",
}


class ReachNetwork:
    """Array indexed VAA topology; memory stays proportional to reach count."""

    def __init__(self, frame):
        if set(VAA_COLUMNS)-set(frame) or frame.comid.duplicated().any():
            raise ValueError("unique COMIDs and all VAA structure columns required")
        self.frame = frame.reset_index(drop=True)
        self.comid = self.frame.comid.to_numpy(dtype=np.int64)
        self.comid_order = np.argsort(self.comid)
        self.sorted_comid = self.comid[self.comid_order]
        self.fromnode = self.frame.fromnode.to_numpy(dtype=np.int64)
        self.tonode = self.frame.tonode.to_numpy(dtype=np.int64)
        self.tonode_order = np.argsort(self.tonode, kind="stable")
        self.sorted_tonode = self.tonode[self.tonode_order]
        self.hydroseq = self.frame.hydroseq.to_numpy(dtype=np.int64)
        self.hydro_order = np.argsort(self.hydroseq)
        self.sorted_hydro = self.hydroseq[self.hydro_order]
        self.downstream = self.frame.dnhydroseq.to_numpy(dtype=np.int64)
        self.length = self.frame.lengthkm.to_numpy(dtype=float)
        if not np.isfinite(self.length).all() or (self.length < 0).any():
            raise ValueError("finite nonnegative reach lengths required")
        self.area = self.frame.totdasqkm.to_numpy(dtype=float)
        self.order = self.frame.streamorde.to_numpy(dtype=float)
        self.start = self.frame.startflag.to_numpy(dtype=int)
        self.divergence = self.frame.divergence.to_numpy(dtype=int)
        self.storage = self.frame.wbareatype.isin(["LakePond", "Reservoir"]).to_numpy()
        self.waterbody = self.frame.wbareacomi.fillna(0).to_numpy(dtype=np.int64)

    def index(self, comid):
        pos = int(np.searchsorted(self.sorted_comid, int(comid)))
        if pos == len(self.comid) or self.sorted_comid[pos] != int(comid):
            raise ValueError(f"COMID absent from VAA: {comid}")
        return int(self.comid_order[pos])

    def parents(self, index):
        node = self.fromnode[index]
        if node <= 0:
            return np.empty(0, dtype=np.int64)
        lo, hi = np.searchsorted(self.sorted_tonode, node, side="left"), np.searchsorted(self.sorted_tonode, node, side="right")
        return self.tonode_order[lo:hi]

    @staticmethod
    def position(measure):
        """NHD linear measure is 100 at the head and 0 at the outlet."""
        missing = not np.isfinite(measure)
        if not missing and not 0 <= measure <= 100:
            raise ValueError("station measure must be within [0, 100]")
        return (50. if missing else float(measure)), missing

    def neighbourhood(self, comid, measure, *, radii=(5., 20., 50.), max_reaches=200000):
        radii = np.asarray(radii, dtype=float)
        if not len(radii) or (radii <= 0).any() or not (np.diff(radii) > 0).all():
            raise ValueError("positive increasing radii required")
        root = self.index(comid)
        measure, missing = self.position(measure)
        # A reach is visited at its downstream end; the station reach starts
        # inside the reach. All other reaches contribute their full length.
        queue, visited = [(0., root)], set()
        totals = {name: np.zeros(len(radii)) for name in
                  ("length", "storage_length", "low_order_length", "reach_count", "junction_count", "major_count", "divergence_count")}
        waterbodies = [set() for _ in radii]
        junctions = {}
        nearest, nearest_major, largest_minor_share = np.inf, np.inf, 0.
        capped = False
        while queue:
            distance, i = heapq.heappop(queue)
            if i in visited or distance > radii[-1]:
                continue
            if len(visited) >= max_reaches:
                capped = True
                break
            visited.add(i)
            length = self.length[i]*((100-measure)/100 if i == root else 1.)
            portions = np.minimum(length, np.maximum(0., radii-distance))
            included = portions > 0
            totals["length"] += portions
            totals["reach_count"] += included
            totals["storage_length"] += portions*self.storage[i]
            totals["low_order_length"] += portions*(self.order[i] <= 3)
            totals["divergence_count"] += included*(self.divergence[i] > 0)
            for k in np.flatnonzero(included & self.storage[i]):
                waterbodies[k].add(int(self.waterbody[i]) if self.waterbody[i] > 0 else int(self.comid[i]))
            inlet_distance = distance+length
            parents = self.parents(i)
            if len(parents) >= 2 and inlet_distance <= radii[-1]:
                nearest = min(nearest, inlet_distance)
                areas = self.area[parents]
                finite = np.isfinite(areas).all() and (areas >= 0).all() and areas.sum() > 0
                share = float(1-areas.max()/areas.sum()) if finite else 0.
                if inlet_distance <= 5:
                    largest_minor_share = max(largest_minor_share, share)
                node = int(self.fromnode[i])
                previous = junctions.get(node, (np.inf, share))
                junctions[node] = (min(previous[0], inlet_distance), share)
                if share >= .20:
                    nearest_major = min(nearest_major, inlet_distance)
            if inlet_distance <= radii[-1]:
                for parent in parents:
                    if int(parent) not in visited:
                        heapq.heappush(queue, (inlet_distance, int(parent)))
        inlet = self.parents(root)
        for distance, share in junctions.values():
            totals["junction_count"] += distance <= radii
            totals["major_count"] += (distance <= radii)*(share >= .20)
        result = {
            "comid": int(comid), "stream_order": self.order[root],
            "drainage_area_km2": self.area[root], "slope": float(self.frame.slope.iloc[root]),
            "physical_headwater": bool(self.start[root] == 1), "inlet_reaches": len(inlet),
            "measure_imputed": missing, "station_measure": measure,
            "nearest_confluence_km": nearest if np.isfinite(nearest) else np.nan,
            "nearest_major_confluence_km": nearest_major if np.isfinite(nearest_major) else np.nan,
            "largest_minor_area_share_5km": largest_minor_share,
            "upstream_search_capped": capped, "searched_reaches": len(visited),
        }
        for k, radius in enumerate(radii):
            suffix = f"{radius:g}km"
            for key, values in totals.items():
                result[f"upstream_{key}_{suffix}"] = float(values[k])
            result[f"waterbody_count_{suffix}"] = len(waterbodies[k])
            result[f"storage_fraction_{suffix}"] = float(totals["storage_length"][k]/totals["length"][k]) if totals["length"][k] > 0 else np.nan
        result.update(
            low_order=bool(self.order[root] <= 3), mainstem=bool(self.order[root] >= 7),
            confluence=bool(nearest_major <= 5),
            chain=bool(nearest_major > 5 and self.start[root] != 1),
            storage=bool(result.get("upstream_storage_length_20km", 0) > 0),
            order_band="1–3" if self.order[root] <= 3 else "4–6" if self.order[root] <= 6 else "7–10",
        )
        return result

    def mainstem_path(self, source_comid, source_measure, target_comid, target_measure, *, max_distance=3000.):
        """Audit a compressed monitored edge along VAA's main downstream route."""
        source, target = self.index(source_comid), self.index(target_comid)
        sm, s_missing = self.position(source_measure)
        tm, t_missing = self.position(target_measure)
        if source == target:
            distance = (sm-tm)/100*self.length[source]
            return {"mainstem_connected": distance >= 0, "path_length_km": distance if distance >= 0 else np.nan,
                    "path_reaches": 1, "path_junctions": 0, "path_major_junctions": 0,
                    "path_storage_km": distance*int(self.storage[source]) if distance >= 0 else np.nan,
                    "path_position_imputed": s_missing or t_missing}
        i, distance, storage, junctions, major_junctions, visited = source, 0., 0., 0, 0, set()
        connected = False
        while i not in visited and distance <= max_distance:
            visited.add(i)
            fraction = sm/100 if i == source else (100-tm)/100 if i == target else 1.
            length = self.length[i]*fraction
            distance += length
            storage += length*int(self.storage[i])
            if i != source:
                parents = self.parents(i)
                junctions += int(len(parents) >= 2)
                areas = self.area[parents]
                if len(parents) >= 2 and np.isfinite(areas).all() and (areas >= 0).all() and areas.sum() > 0:
                    major_junctions += int(1-areas.max()/areas.sum() >= .20)
            if i == target:
                connected = True
                break
            seq = self.downstream[i]
            pos = int(np.searchsorted(self.sorted_hydro, seq))
            if seq <= 0 or pos == len(self.hydroseq) or self.sorted_hydro[pos] != seq:
                break
            i = int(self.hydro_order[pos])
        return {"mainstem_connected": connected, "path_length_km": distance if connected else np.nan,
                "path_reaches": len(visited), "path_junctions": junctions if connected else np.nan,
                "path_major_junctions": major_junctions if connected else np.nan,
                "path_storage_km": storage if connected else np.nan,
                "path_position_imputed": s_missing or t_missing}


def source_doc_summaries(dataset, train_cells):
    """Read only permitted DOC labels, even for derived seasonal features."""
    y = np.asarray(dataset["y"])
    x, x_mask = np.asarray(dataset["x"]), np.asarray(dataset["x_mask"])
    cells = np.asarray(train_cells)
    if cells.ndim != 1 or cells.dtype.kind not in "iu" or len(np.unique(cells)) != len(cells) or (cells < 0).any() or (cells >= y.size).any():
        raise ValueError("unique in-range training cells required")
    names = np.asarray(dataset["site_no"], str)
    months = pd.DatetimeIndex(dataset["months"]).month.to_numpy()
    flow_index = list(dataset["feature_channels"]).index("discharge")
    rows = []
    for station in np.unique(cells//y.shape[1]):
        t = cells[cells//y.shape[1] == station] % y.shape[1]
        values = y[station, t].astype(float)
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("source DOC must be finite and nonnegative")
        mean = values.mean()
        seasonal = [np.median(values[months[t] == m]) for m in np.unique(months[t])]
        valid = x_mask[station, t, flow_index].astype(bool)
        q = x[station, t, flow_index].astype(float)
        valid &= np.isfinite(q) & (q >= 0)
        slope, n_pair = np.nan, int(valid.sum())
        if n_pair >= 24:
            qlog = np.log1p(q[valid])
            season = 2*np.pi*months[t[valid]]/12
            design = np.column_stack([np.ones(n_pair), qlog, np.sin(season), np.cos(season)])
            if np.linalg.matrix_rank(design) == 4:
                slope = float(np.linalg.lstsq(design, np.log1p(values[valid]), rcond=None)[0][1])
        rows.append({"station": names[station], "n_doc": len(values), "doc_mean": mean,
                     "doc_median": np.median(values), "doc_q90": np.quantile(values, .90),
                     "doc_cv": values.std()/mean if mean > 0 else np.nan,
                     "doc_iqr": np.quantile(values, .75)-np.quantile(values, .25),
                     "represented_seasons": len(seasonal),
                     "seasonal_amplitude": max(seasonal)-min(seasonal) if len(values) >= 12 and len(seasonal) >= 6 else np.nan,
                     "cq_log1p_slope": slope, "n_doc_flow": n_pair})
    return pd.DataFrame(rows)


def source_pair_associations(dataset, train_cells, edges, *, lags=(0, 1, 3, 6, 12)):
    """All requested lags, using source labels only; no best-lag selection."""
    truth = np.asarray(dataset["y"])
    visible = np.full(truth.shape, np.nan)
    visible.ravel()[train_cells] = np.log1p(truth.ravel()[train_cells])
    month = pd.DatetimeIndex(dataset["months"]).month.to_numpy()
    anomalies = visible.copy()
    for i in range(len(visible)):
        for m in range(1, 13):
            select = (month == m) & np.isfinite(visible[i])
            if select.any():
                anomalies[i, select] -= visible[i, select].mean()
    positions = {str(name): i for i, name in enumerate(dataset["site_no"])}
    rows = []
    for edge in edges.itertuples():
        source, target = positions[edge.source], positions[edge.target]
        for lag in lags:
            a = visible[source, :truth.shape[1]-lag]
            b = visible[target, lag:]
            valid = np.isfinite(a) & np.isfinite(b)
            if valid.sum() < 12:
                rho = np.nan
            else:
                ra, rb = pd.Series(a[valid]).rank(), pd.Series(b[valid]).rank()
                rho = ra.corr(rb) if ra.std() > 0 and rb.std() > 0 else np.nan
            aa, bb = anomalies[source, :truth.shape[1]-lag], anomalies[target, lag:]
            if valid.sum() >= 12 and np.std(aa[valid]) > 0 and np.std(bb[valid]) > 0:
                arho = pd.Series(aa[valid]).rank().corr(pd.Series(bb[valid]).rank())
            else:
                arho = np.nan
            rows.append({"source": edge.source, "target": edge.target, "lag_months": lag,
                         "n_pairs": int(valid.sum()), "rho_log_doc": rho,
                         "rho_seasonal_anomaly": arho})
    return pd.DataFrame(rows)
