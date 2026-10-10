"""Position of potential terrestrial DOC sources along physical river paths."""
from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

PLACEMENT = ("wetland_distance_ratio", "forest_distance_ratio",
             "wetland_near_excess", "forest_near_excess",
             "wetland_riparian_enrichment", "forest_riparian_enrichment")


class UpstreamDistances:
    """Shortest directed channel paths, including NHD secondary downstream links."""

    def __init__(self, frame):
        if frame.comid.duplicated().any() or frame.hydroseq.duplicated().any():
            raise ValueError("unique reach and hydroseq required")
        self.comids = frame.comid.to_numpy(np.int64)
        self.length = frame.lengthkm.to_numpy(float)
        if not np.isfinite(self.length).all() or (self.length < 0).any():
            raise ValueError("nonnegative finite reach length required")
        self.order = np.argsort(self.comids)
        self.sorted_comids = self.comids[self.order]
        hydro = frame.hydroseq.to_numpy(np.int64)
        order = np.argsort(hydro)
        sorted_hydro = hydro[order]
        source, target = [], []
        for col in ("dnhydroseq", "dnminorhyd"):
            seq = frame[col].fillna(0).to_numpy(np.int64)
            pos = np.searchsorted(sorted_hydro, seq)
            safe = np.minimum(pos, len(order)-1)
            valid = (seq > 0) & (pos < len(order)) & (sorted_hydro[safe] == seq)
            upstream = np.flatnonzero(valid)
            source.extend(order[pos[valid]])
            target.extend(upstream)
        pairs = np.unique(np.column_stack([source, target]), axis=0)
        weight = (self.length[pairs[:, 0]]+self.length[pairs[:, 1]])/2
        self.graph = csr_matrix((weight, (pairs[:, 0], pairs[:, 1])), shape=(len(frame), len(frame)))

    def index(self, comid):
        pos = np.searchsorted(self.sorted_comids, int(comid))
        if pos == len(self.comids) or self.sorted_comids[pos] != int(comid):
            raise ValueError(f"reach not in routing graph: {comid}")
        return int(self.order[pos])

    def distances(self, receiving_comid, member_comids):
        root = self.index(receiving_comid)
        distance = dijkstra(self.graph, directed=True, indices=root)
        member_comids = np.asarray(member_comids, np.int64)
        pos = np.searchsorted(self.sorted_comids, member_comids)
        safe = np.minimum(pos, len(self.comids)-1)
        if (pos == len(self.comids)).any() or not np.array_equal(self.sorted_comids[safe], member_comids):
            raise ValueError("membership reach not in routing graph")
        positions = self.order[pos]
        # Graph edges connect reach midpoints. Add receiving half-length to
        # express distance to its outlet, consistent with whole-reach basins.
        return distance[positions]+self.length[root]/2


def source_placement(area, distance, wetland, forest, *, riparian_area=None,
                     riparian_wetland=None, riparian_forest=None, near_km=50.):
    """Finite, positive-area catchments only; missing cover never becomes zero."""
    area, distance, wetland, forest = [np.asarray(v, float) for v in (area, distance, wetland, forest)]
    if not (area.shape == distance.shape == wetland.shape == forest.shape) or area.ndim != 1:
        raise ValueError("aligned one-dimensional catchment data required")
    if (area < 0).any() or not np.isfinite(area).all():
        raise ValueError("finite nonnegative catchment areas required")
    cover_ok = np.isfinite(wetland) & np.isfinite(forest) & (wetland >= 0) & (forest >= 0) & (wetland <= 100) & (forest <= 100)
    valid = (area > 0) & np.isfinite(distance) & (distance >= 0) & cover_ok
    total = area.sum()
    included = area[valid].sum()
    result = {"represented_area_km2": included, "total_area_km2": total,
              "represented_area_fraction": included/total if total > 0 else np.nan,
              "represented_reaches": int(valid.sum())}
    if not valid.any():
        return result | {name: np.nan for name in PLACEMENT}
    a, d = area[valid], distance[valid]
    mean_distance = np.average(d, weights=a)
    near = d <= near_km
    area_near = a[near].sum()/a.sum()
    result.update(drainage_mean_distance_km=mean_distance, near_drainage_fraction=area_near)
    for name, cover, rp_cover in (("wetland", wetland, riparian_wetland), ("forest", forest, riparian_forest)):
        mass = a*cover[valid]/100
        amount = mass.sum()
        source_distance = np.average(d, weights=mass) if amount > 0 else np.nan
        result.update({f"{name}_area_km2": amount, f"{name}_pct": 100*amount/a.sum(),
                       f"{name}_mean_distance_km": source_distance,
                       f"{name}_distance_ratio": source_distance/mean_distance if mean_distance > 0 else np.nan,
                       f"{name}_near_excess": mass[near].sum()/amount-area_near if amount > 0 else np.nan})
        result[f"{name}_riparian_enrichment"] = np.nan
        result[f"{name}_riparian_area_coverage"] = np.nan
        if riparian_area is not None and rp_cover is not None:
            ra, rc = np.asarray(riparian_area, float), np.asarray(rp_cover, float)
            if ra.shape != area.shape or rc.shape != area.shape:
                raise ValueError("riparian values must align with catchments")
            rv = valid & np.isfinite(ra) & (ra > 0) & np.isfinite(rc) & (rc >= 0) & (rc <= 100)
            denominator = ra[valid & np.isfinite(ra) & (ra > 0)].sum()
            coverage = ra[rv].sum()/denominator if denominator > 0 else np.nan
            result[f"{name}_riparian_area_coverage"] = coverage
            if rv.any() and coverage >= .95:
                # Contrast uses the same represented catchments in both AOIs.
                result[f"{name}_riparian_enrichment"] = np.average(rc[rv], weights=ra[rv])-np.average(cover[rv], weights=area[rv])
    return result
