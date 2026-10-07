"""Recompute DOC populations, signal statistics and morphology associations."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from analyze_doc_river_form_process_v1 import (
    DATASET,
    OLD,
    ROOT,
    build_panels,
    summarize,
)

from river_graph.analysis.river_mechanisms import permitted_doc
from river_graph.experiments.provenance import sha256_file


def equivalent(rebuilt, path):
    saved = pd.read_csv(path, dtype={"target": str, "source": str, "source_a": str, "source_b": str,
                                    "station": str, "huc4": str, "group": str, "omitted_block": str,
                                    "max_leverage_receiver": str})
    pd.testing.assert_frame_equal(rebuilt.reset_index(drop=True), saved.reset_index(drop=True),
                                  check_dtype=False, check_exact=False, atol=1e-11, rtol=1e-11)


def main():
    sources = json.loads((ROOT/"analysis_sources.json").read_text())
    for path, expected in sources["source_hashes"].items():
        if sha256_file(Path(path)) != expected:
            raise ValueError(f"source changed: {path}")
    data = torch.load(DATASET, map_location="cpu", weights_only=False)
    cells = np.load(OLD/"source_cells.npy")
    visible = permitted_doc(data, cells)
    truth = np.asarray(data["y"], float).copy()
    hidden = np.ones(truth.size, bool)
    hidden[cells] = False
    truth.ravel()[hidden] = 100000
    np.testing.assert_array_equal(permitted_doc({"y": truth}, cells), visible)
    stations = {str(s): i for i, s in enumerate(data["site_no"])}
    months = {pd.Timestamp(d): i for i, d in enumerate(data["months"])}
    inventory = pd.read_csv(OLD/"confluence_inventory.csv", dtype={"source_a": str, "source_b": str, "target": str}).set_index("pair_id")
    records = pd.read_parquet(OLD/"mixing_records.parquet")
    checked = 0
    for pair, frame in records[records.population.eq("monthly")].groupby("pair_id"):
        candidate = inventory.loc[pair]
        assert candidate.eligible_monthly and candidate.branch_status == "independent"
        t = np.array([months[pd.Timestamp(d)] for d in frame.date])
        for column, station in (("doc_a", candidate.source_a), ("doc_b", candidate.source_b), ("doc_target", candidate.target)):
            np.testing.assert_array_equal(frame[column], visible[stations[station], t])
        checked += len(frame)
    cases, paths, panels, ledger = build_panels()
    out = ROOT/"analysis"
    equivalent(cases, out/"mixing_connections.csv")
    equivalent(paths, out/"path_connections.csv")
    equivalent(ledger, out/"inclusion_ledger.csv")
    for name, frame in panels.items():
        assert not frame.target.duplicated().any()
        equivalent(frame, out/f"{name}_receivers.csv")
    for path in (panels["path_hydro_common_calendar"], panels["path_hydro_adjusted"]):
        assert path.target.to_list() == panels["path_hydro_adjusted"].target.to_list()
        np.testing.assert_array_equal(path.n_months, panels["path_hydro_adjusted"].n_months)
        np.testing.assert_array_equal(path.n_connections, panels["path_hydro_adjusted"].n_connections)
    for name, frame in summarize(panels, sources["bootstrap_draws"]).items():
        equivalent(frame, out/f"{name}.csv")
    descriptors = pd.read_csv(out/"class_descriptors.csv")
    singleton = descriptors[descriptors.n_blocks.eq(1)]
    assert singleton[["ci_low", "ci_high"]].isna().all().all()
    for name in ("manifest.json", "manifest_cn.json"):
        figure = json.loads((ROOT/"figures"/name).read_text())
        for path, expected in {**figure["source_hashes"], **figure["figure_hashes"]}.items():
            assert sha256_file(Path(path)) == expected
    record = {"status": "passed", "source_files_checked": len(sources["source_hashes"]),
              "monthly_mixing_record_rows_checked": checked, "non_source_DOC_perturbation": "unchanged",
              "all_receiver_panels_rebuilt": True, "all_bootstrap_tables_recomputed": True,
              "hydro_sensitivity": "same station pairs and same monthly population, before/after adjustment",
              "single_block_class_intervals": "not estimable, stored NA",
              "models_retrained": False}
    (ROOT/"verification.json").write_text(json.dumps(record, indent=2)+"\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
