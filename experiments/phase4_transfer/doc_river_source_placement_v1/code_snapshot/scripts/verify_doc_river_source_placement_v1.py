"""Independently verify landscapes, distance alignment and source-only responses."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from river_graph.analysis.river_doc_structure import source_station_response
from river_graph.analysis.river_source_placement import source_placement
from river_graph.experiments.provenance import sha256_file

ROOT = Path("experiments/phase4_transfer/doc_river_source_placement_v1")
CACHE = Path("data/raw/river_source_placement_v1")


def main():
    checked = 0
    for name in ("location_sources.json", "placement_sources.json", "analysis_sources.json"):
        value = json.loads((ROOT/name).read_text())
        for path, expected in value["source_hashes"].items():
            if sha256_file(path) != expected:
                raise ValueError(f"source changed: {path}")
            checked += 1
    acquisition = json.loads((ROOT/"landscape_sources.json").read_text())
    if sha256_file(acquisition["landscape_file"]) != acquisition["landscape_sha256"]:
        raise ValueError("landscape cache changed")
    landscape = pd.read_parquet(acquisition["landscape_file"]).set_index("comid")
    if not landscape.index.is_unique:
        raise ValueError("duplicate landscape COMID")
    placement = pd.read_csv(ROOT/"analysis/station_source_placement.csv", dtype={"station": str})
    area = pd.read_parquet("cache/nldplus_vaa.parquet", columns=["comid", "areasqkm"]).set_index("comid").areasqkm
    for r in placement.itertuples():
        with np.load(CACHE/"routing"/f"comid_{r.comid}.npz") as z:
            members, distance = z["comids"], z["distance_km"]
        with np.load(Path("data/raw/river_planform_v1/members_full")/f"comid_{r.comid}.npz") as z:
            np.testing.assert_array_equal(members, np.unique(z["comids"]))
        f = landscape.reindex(members)
        wetland = f.pctwdwet2019cat.to_numpy(float)+f.pcthbwet2019cat.to_numpy(float)
        forest = f.pctdecid2019cat.to_numpy(float)+f.pctconif2019cat.to_numpy(float)+f.pctmxfst2019cat.to_numpy(float)
        invalid = ~(f.nlcd2019_catpctfull.to_numpy(float) >= 95)
        wetland[invalid], forest[invalid] = np.nan, np.nan
        result = source_placement(area.reindex(members).to_numpy(float), distance, wetland, forest,
            riparian_area=f.catareasqkmrp100.to_numpy(float),
            riparian_wetland=f.pctwdwet2019catrp100.to_numpy(float)+f.pcthbwet2019catrp100.to_numpy(float),
            riparian_forest=f.pctdecid2019catrp100.to_numpy(float)+f.pctconif2019catrp100.to_numpy(float)+f.pctmxfst2019catrp100.to_numpy(float))
        for key, value in result.items():
            np.testing.assert_allclose(getattr(r, key), value, rtol=1e-10, atol=1e-10, equal_nan=True)
    dataset = torch.load("data/processed/mississippi_graph_graphfix_st357.pt", map_location="cpu", weights_only=False)
    cells = np.load("experiments/phase4_transfer/doc_river_mechanisms_v1/analysis/source_cells.npy")
    expected = []
    for split in (142, 143, 144):
        with np.load(f"experiments/phase4_transfer/unified_doc_spatial_v1/confirmation/masks/split{split}.npz") as z:
            expected.extend(z["train"])
    np.testing.assert_array_equal(cells, np.unique(expected))
    response, _, _ = source_station_response(dataset, cells)
    changed = dataset.copy()
    changed_y = np.asarray(dataset["y"]).copy()
    hidden = np.ones(changed_y.size, dtype=bool)
    hidden[cells] = False
    changed_y.ravel()[hidden] = 123456.
    changed["y"] = changed_y
    perturbed, _, _ = source_station_response(changed, cells)
    pd.testing.assert_frame_equal(response, perturbed)
    panel = pd.read_csv(ROOT/"analysis/station_doc_source_panel.csv", dtype={"station": str})
    merged = panel.merge(response[["station", "doc_median", "n_doc"]], on="station", suffixes=("", "_recomputed"), validate="one_to_one")
    np.testing.assert_allclose(merged.doc_median, merged.doc_median_recomputed)
    np.testing.assert_array_equal(merged.n_doc, merged.n_doc_recomputed)
    if (panel.loc[panel.included, "mapping_status"] == "gross_area_mismatch").any():
        raise ValueError("mapping discrepancy entered primary cohort")
    predictions = pd.read_csv(ROOT/"analysis/huc4_blocked_predictions.csv", dtype={"station": str, "huc4": str})
    for (_, _), sub in predictions.groupby(["population", "fold"]):
        arm = sub.groupby("model").station.apply(lambda v: tuple(sorted(v)))
        if arm.nunique() != 1:
            raise ValueError("models use different evaluation stations")
    for population, sub in predictions.groupby("population"):
        if sub.groupby("station").fold.nunique().max() != 1 or sub.groupby("huc4").fold.nunique().max() != 1:
            raise ValueError(f"HUC4 fold leakage: {population}")
    result = {"status": "passed", "source_identities_checked": checked, "landscape_reaches": len(landscape),
        "stations_recomputed": len(placement), "allowed_source_cells": len(cells),
        "hidden_DOC_perturbation": "all response summaries unchanged", "equal_evaluation_population": True,
        "HUC4_folds": "each region and station appears in one held-out fold", "new_neural_training": False}
    (ROOT/"verification.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
