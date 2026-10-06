"""Check shape/DOC units, held-out roles, paired populations and stored metrics."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from river_graph.analysis.river_doc_structure import source_station_response
from river_graph.experiments.provenance import sha256_file

ROOT = Path("experiments/phase4_transfer/doc_river_planform_doc_v1")


def main():
    meta = json.loads((ROOT/"sources.json").read_text())
    for name, digest in meta["source_hashes"].items():
        if sha256_file(Path(name)) != digest:
            raise ValueError(f"changed source: {name}")
    out = ROOT/"analysis"
    dataset = torch.load("data/processed/mississippi_graph_graphfix_st357.pt", map_location="cpu", weights_only=False)
    cells = np.load(out/"source_cells.npy")
    expected = []
    for split in (142, 143, 144):
        with np.load(f"experiments/phase4_transfer/unified_doc_spatial_v1/confirmation/masks/split{split}.npz") as m:
            expected.extend(m["train"])
    np.testing.assert_array_equal(cells, np.unique(expected))
    original, seasonal, threshold = source_station_response(dataset, cells)
    altered = dict(dataset)
    y = np.asarray(dataset["y"]).copy()
    hidden = np.ones(y.size, bool)
    hidden[cells] = False
    y.ravel()[hidden] = 1e8
    altered["y"] = y
    perturbed, perturbed_season, perturbed_threshold = source_station_response(altered, cells)
    pd.testing.assert_frame_equal(original, perturbed)
    pd.testing.assert_frame_equal(seasonal, perturbed_season)
    assert threshold == perturbed_threshold == meta["q90_mg_L"]
    inclusion = pd.read_csv(out/"inclusion.csv", dtype={"station": str})
    assert len(inclusion) == 357 and not inclusion.station.duplicated().any()
    panel = pd.read_csv(out/"station_doc_response.csv", dtype={"station": str})
    assert not panel.station.duplicated().any()
    assert panel.cluster.notna().all() and panel.cluster.isin([1, 2, 3]).all()
    p = pd.read_csv(out/"huc4_blocked_predictions.csv", dtype={"station": str, "huc4": str})
    assert not p.duplicated(["station", "model"]).any()
    assert p.groupby("huc4").fold.nunique().eq(1).all()
    assert p.groupby("station").model.nunique().eq(3).all()
    truth = panel.set_index("station").doc_median
    np.testing.assert_allclose(p.y_true, p.station.map(truth))
    assert np.isfinite(p[["y_true", "y_pred", "y_log1p", "y_pred_log1p"]]).all().all()
    gains = pd.read_csv(out/"huc4_blocked_gains.csv")
    for r in gains.itertuples():
        names = ("y_log1p", "y_pred_log1p") if r.space == "log1p" else ("y_true", "y_pred")
        f = p.copy()
        f["error"] = abs(f[names[0]]-f[names[1]])
        metrics = f.groupby(["model", "fold"]).error.mean().groupby("model").mean()
        np.testing.assert_allclose([r.candidate_mae, r.reference_mae],
                                   [metrics[r.model], metrics["environment_area"]])
        assert r.gain_ci_low_pct <= r.gain_ci_high_pct
    messages = pd.read_csv(out/"historical_message_gains.csv")
    station = pd.read_csv(out/"historical_message_station_errors.csv")
    for r in messages.loc[~messages["tail"] & messages.group.isin(["all_classified", "class_1", "class_2", "class_3"])].itertuples():
        s = station[(station.family == r.family) & (station.candidate == r.candidate)]
        if r.group != "all_classified":
            s = s[s.cluster.eq(int(r.group[-1]))]
        np.testing.assert_allclose([r.candidate_mae, r.reference_mae],
                                   [np.average(s.candidate_mae, weights=s.n_cells), np.average(s.reference_mae, weights=s.n_cells)])
        assert len(s) == r.n_stations_unique and s.n_cells.sum() == r.n_station_months_unique
    edge = pd.read_csv(out/"upstream_associations.csv", dtype={"source": str, "target": str})
    assert edge.groupby(["source", "target"]).lag_months.nunique().eq(5).all()
    diagnostics = pd.read_csv(out/"regression_diagnostics.csv")
    assert np.isfinite(diagnostics.condition).all()
    record = {"status": "passed", "source_hashes_checked": len(meta["source_hashes"]),
              "source_cells": len(cells), "eligible_station_responses": int(panel.eligible.sum()),
              "hidden_label_perturbation": "unchanged responses and Q90",
              "blocked_prediction_populations": "equal; HUC4 never splits across folds",
              "message_means": "match paired cell-weighted seed-mean station errors",
              "external_or_geographical_test_read": False}
    (ROOT/"verification.json").write_text(json.dumps(record, indent=2)+"\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
