"""Replay source participation, source-role isolation and the fixed river inventory."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from analyze_doc_river_active_structure_v1 import CELLS, DATA, ROOT, build_tables

from river_graph.analysis.river_mechanisms import permitted_doc
from river_graph.experiments.provenance import sha256_file


def main():
    metadata = json.loads((ROOT/"analysis_sources.json").read_text())
    for group in ("input_hashes", "code_hashes", "output_hashes"):
        for path, expected in metadata[group].items():
            assert sha256_file(Path(path)) == expected, path
    replay = build_tables()
    for name, expected in replay.items():
        if name in ("monthly_participation", "source_flow_weights"):
            saved = pd.read_parquet(ROOT/"analysis"/f"{name}.parquet")
            pd.testing.assert_frame_equal(saved, expected)
        else:
            saved = pd.read_csv(ROOT/"analysis"/f"{name}.csv", dtype={"target": str, "source_station": str, "huc4": str})
            assert saved.shape == expected.shape, name
            for column in expected:
                if pd.api.types.is_numeric_dtype(expected[column]):
                    assert np.allclose(saved[column], expected[column], equal_nan=True), (name, column)
                else:
                    assert saved[column].eq(expected[column]).all(), (name, column)
    weights = replay["source_flow_weights"]
    sums = weights.groupby(["target", "date"]).flow_weight.sum()
    assert np.allclose(sums, 1.)
    assert not weights.duplicated(["target", "source_station", "date"]).any()
    assert weights.groupby(["target", "source_station"]).fixed_path_km.nunique().eq(1).all()
    assert len(replay["availability"]) == 32
    data = torch.load(DATA, weights_only=False, map_location="cpu")
    cells = np.load(CELLS)
    visible = permitted_doc(data, cells)
    changed = dict(data)
    target = data["y"].clone() if torch.is_tensor(data["y"]) else np.asarray(data["y"]).copy()
    allowed = np.zeros(target.numel() if torch.is_tensor(target) else target.size, dtype=bool)
    allowed[cells] = True
    target.reshape(-1)[~allowed] = 123456.
    changed["y"] = target
    assert np.array_equal(visible, permitted_doc(changed, cells), equal_nan=True)
    report = {"status": "pass", "fixed_networks": 32,
        "hydro_networks": len(replay["network_hydro_changes"]),
        "doc_networks": len(replay["network_doc_changes"]),
        "source_weights_sum_to_one": True, "physical_paths_unchanged": True,
        "non_source_doc_perturbation": "bitwise unchanged permitted DOC",
        "all_point_tables_replayed": True}
    (ROOT/"verification.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
