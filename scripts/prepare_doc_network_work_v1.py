"""Audit public model sources and build label-free NHD mainstem SSN distances."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from river_graph.experiments.provenance import sha256_file
from river_graph.topology.river_structure import VAA_COLUMNS, ReachNetwork

ROOT = Path("cache/doc_network_work_v1")
DATA = Path("data/processed/mississippi_graph_graphfix_st357.pt")
NODES = Path("data/processed/graph_nodes_graphfix_st357.csv")
VAA = Path("cache/nldplus_vaa.parquet")


def build_geometry(network, nodes):
    paths, positions, weights, order = [], [], [], []
    missing_positions = 0
    branch_weights, area_fallbacks = {}, set()
    for row in nodes.itertuples(index=False):
        start = network.index(row.comid)
        measure, missing = network.position(row.measure)
        missing_positions += int(missing)
        distance = -network.length[start]*(100-measure)/100
        location, weight, route = {}, {}, []
        i, probability = start, 1.
        while i not in location:
            location[i], weight[i] = distance, probability
            route.append(i)
            distance += network.length[i]
            seq = network.downstream[i]
            pos = int(np.searchsorted(network.sorted_hydro, seq))
            if seq <= 0 or pos == len(network.sorted_hydro) or network.sorted_hydro[pos] != seq:
                break
            child = int(network.hydro_order[pos])
            if child not in branch_weights:
                parents = network.parents(child)
                parents = parents[network.downstream[parents] == network.hydroseq[child]]
                areas = network.area[parents]
                if not np.isfinite(areas).all() or (areas <= 0).any():
                    areas = np.ones(len(parents))
                    if len(parents) > 1:
                        area_fallbacks.add(child)
                branch_weights[child] = dict(zip(parents, np.sqrt(areas/areas.sum()), strict=True))
            if i not in branch_weights[child]:
                raise ValueError("mainstem parent missing from confluence")
            probability *= branch_weights[child][i]
            i = child
        else:
            raise ValueError("cycle in NHD mainstem topology")
        paths.append(route)
        positions.append(location)
        weights.append(weight)
        order.append((start, measure))
    n = len(nodes)
    distance, up = np.full((n, n), np.inf), np.eye(n)
    np.fill_diagonal(distance, 0.)
    for i in range(n):
        for j in range(i+1, n):
            a, am = order[i]
            b, bm = order[j]
            connected = False
            for first, second, reach, measure in ((i, j, b, bm), (j, i, a, am)):
                if reach in positions[first]:
                    d = positions[first][reach] + network.length[reach]*(100-measure)/100
                    if d >= -1e-8:
                        distance[i, j] = distance[j, i] = max(d, 0.)
                        up[i, j] = up[j, i] = weights[first][reach]
                        connected = True
                        break
            if not connected:
                common = next((reach for reach in paths[i] if reach in positions[j]), None)
                if common is not None:
                    d = positions[i][common] + positions[j][common]
                    if d < 0:
                        raise ValueError("negative distance to common junction")
                    distance[i, j] = distance[j, i] = d
    latitude = np.deg2rad(nodes.dec_lat_va.to_numpy(float))
    longitude = np.deg2rad(nodes.dec_long_va.to_numpy(float))
    xyz = 6371.*np.column_stack([np.cos(latitude)*np.cos(longitude),
                                np.cos(latitude)*np.sin(longitude), np.sin(latitude)])
    euclidean = np.linalg.norm(xyz[:, None]-xyz[None], axis=-1)
    return {"distance": distance, "tailup_weight": up, "euclidean": euclidean}, {
        "stations": n, "position_imputations": missing_positions,
        "junctions_with_equal_weight_fallback": len(area_fallbacks),
        "flow_connected_pairs": int(np.count_nonzero(np.triu(up > 0, 1))),
        "shared_downstream_pairs": int(np.count_nonzero(np.triu(np.isfinite(distance), 1))),
        "distance_definition": "NHDPlus mainstem lengths and station measures; true first downstream meeting reach",
        "tailup_weights": "product of sqrt(mainstem-parent drainage area / sum mainstem-parent areas); equal branch shares where an area's missing/nonpositive, with counts reported",
        "euclidean_definition": "Earth-centered chord distance in km (positive-definite exponential kernel)",
        "mainstem_only": True}


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    destination = ROOT / "preparation.json"
    if destination.exists():
        saved = json.loads(destination.read_text())
        for name, expected in saved["files"].items():
            if sha256_file(name) != expected:
                raise ValueError(f"prepared dependency changed: {name}")
        print(json.dumps({"stage": "already_prepared", "geometry": saved["geometry"]}))
        return
    nodes = pd.read_csv(NODES, dtype={"site_no": str, "huc_cd": str})
    dataset = torch.load(DATA, weights_only=False, map_location="cpu")
    np.testing.assert_array_equal(nodes.site_no, np.asarray(dataset["site_no"], str))
    frame = pd.read_parquet(VAA, columns=list(VAA_COLUMNS))
    geometry, audit = build_geometry(ReachNetwork(frame), nodes)
    for range_km in (1., 100., 1000., 10000.):
        for name, matrix in (("tailup", geometry["tailup_weight"]*np.exp(-geometry["distance"]/range_km)),
                             ("taildown", np.exp(-geometry["distance"]/range_km)),
                             ("euclidean", np.exp(-geometry["euclidean"]/range_km))):
            minimum = float(np.linalg.eigvalsh(matrix).min())
            if minimum < -1e-8:
                raise ValueError(f"non-positive covariance: {name}, {range_km}, {minimum}")
    np.savez_compressed(ROOT / "geometry.npz", **geometry)
    sources = ROOT / "sources"
    files = [DATA, NODES, VAA, ROOT / "geometry.npz",
             sources / "PhyST-HC/scripts/Functions_DOC.py",
             sources / "PhyST-HC/scripts/Inputs.py", sources / "PhyST-HC/README.md",
             sources / "SSN2/R/cov_matrix.R", sources / "SSN2/R/cov_vector.R"]
    record = {"geometry": audit, "files": {str(path): sha256_file(path) for path in files},
              "source_commits": {name: subprocess.check_output(["git", "-C", str(sources/name), "rev-parse", "HEAD"], text=True).strip()
                                 for name in ("PhyST-HC", "SSN2")},
              "dataset_keys": list(dataset), "native_reproductions": False}
    destination.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"stage": "prepared", "geometry": audit, "source_commits": record["source_commits"]}))


if __name__ == "__main__":
    main()
