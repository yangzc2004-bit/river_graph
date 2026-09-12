"""Regression tests for the defects found in review of cea452c..419080d.

Each test reproduces a specific failure that the earlier revision had:

1. a recovered copy and the overwritten file of the same name were both read,
   double-counting test cells;
2. HUC2 was inferred from the station-id prefix;
3. the config hash ignored dataset and mask content, so a modified dataset still
   returned "cached, skip";
4. an interrupted pair (new predictions, old metrics) was accepted as complete;
5. ``--rebuild-predictions`` trained without storing predictions;
6. identity conflicts exited 0;
7. the bootstrap resampled rows rather than stations.

No test trains a real model.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from river_graph.experiments.prediction_sources import (
    PredictionSourceError,
    load_sources,
    model_mask_of,
    sha256_file,
)

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


run_gnn = _load("run_gnn_review", "run_gnn.py")
analyze = _load("analyze_review", "analyze_predictions.py")


N, T = 5, 8


def _dataset() -> dict:
    ymask = torch.zeros(N, T, dtype=torch.bool)
    for i in range(N):
        for j in range(T):
            if (i + j) % 2 == 0:
                ymask[i, j] = True
    y = torch.zeros(N, T, dtype=torch.float32)
    y[ymask] = torch.arange(int(ymask.sum()), dtype=torch.float32) * 0.5 + 1.0
    return {
        "site_no": [f"{i:08d}" for i in range(N)],
        "months": [f"2020-{j + 1:02d}-01" for j in range(T)],
        "y": y,
        "y_mask": ymask,
    }


# --------------------------------------------------------------------------
# 1. duplicate name across batches must select exactly one source
# --------------------------------------------------------------------------


def _fake_manifest(tmp_path: Path, entries: dict) -> Path:
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(entries), encoding="utf-8")
    return path


def test_model_mask_parsing():
    assert model_mask_of(Path("A__m.parquet")) == ("A", "m")
    with pytest.raises(PredictionSourceError):
        model_mask_of(Path("nounderscores.parquet"))


def test_real_manifest_selects_unique_sources():
    """The live manifest must yield one source per (model, mask)."""
    sources, report = load_sources()
    keys = [s.key for s in sources]
    assert len(keys) == len(set(keys)), "duplicate (model, mask) selected"
    assert report["duplicate_keys"] == []
    assert report["missing"] == []
    assert report["hash_changed"] == []
    # the conflicted current-batch copy must not be among the selected files
    conflicted = [s for s in sources
                  if s.path.name == "G0_gcn_none__e1_r20_seed42.parquet"]
    assert len(conflicted) == 1
    assert conflicted[0].batch == "predictions_historical", (
        "the recovered copy is the frozen-table source; the overwritten current "
        "copy must not be selected")


def test_same_name_in_both_batches_selects_one(tmp_path, monkeypatch):
    """A name present in both directories must be counted once, not twice."""
    cur = tmp_path / "predictions"
    hist = tmp_path / "predictions_historical"
    cur.mkdir()
    hist.mkdir()
    (cur / "M__m1.parquet").write_bytes(b"current-copy")
    (hist / "M__m1.parquet").write_bytes(b"recovered-copy")

    monkeypatch.setattr("river_graph.experiments.prediction_sources.BATCH_DIRS",
                        {"predictions": cur, "predictions_historical": hist})
    manifest = _fake_manifest(tmp_path, {
        "predictions": [{"file": "M__m1.parquet", "batch": "predictions",
                         "used_for_phase_a": False}],
        "historical_recovered": [{"file": "M__m1.parquet",
                                  "batch": "predictions_historical",
                                  "used_for_phase_a": True}],
        "conflicts": [], "zero_coverage": [],
    })
    sources, report = load_sources(manifest)
    assert len(sources) == 1
    assert sources[0].batch == "predictions_historical"
    assert report["duplicate_keys"] == []


def test_hash_change_is_reported_not_used(tmp_path, monkeypatch):
    d = tmp_path / "predictions"
    d.mkdir()
    target = d / "M__m1.parquet"
    target.write_bytes(b"content-A")
    monkeypatch.setattr("river_graph.experiments.prediction_sources.BATCH_DIRS",
                        {"predictions": d})
    manifest = _fake_manifest(tmp_path, {
        "predictions": [{"file": "M__m1.parquet", "batch": "predictions",
                         "used_for_phase_a": True, "sha256": sha256_file(target)}],
        "historical_recovered": [], "conflicts": [], "zero_coverage": [],
    })
    sources, report = load_sources(manifest)
    assert len(sources) == 1 and report["hash_changed"] == []

    target.write_bytes(b"content-B")
    sources, report = load_sources(manifest)
    assert sources == []
    assert len(report["hash_changed"]) == 1


def test_unmanifested_file_is_never_used(tmp_path, monkeypatch):
    d = tmp_path / "predictions"
    d.mkdir()
    (d / "M__m1.parquet").write_bytes(b"x")
    (d / "STRAY__m9.parquet").write_bytes(b"y")
    monkeypatch.setattr("river_graph.experiments.prediction_sources.BATCH_DIRS",
                        {"predictions": d})
    manifest = _fake_manifest(tmp_path, {
        "predictions": [{"file": "M__m1.parquet", "batch": "predictions",
                         "used_for_phase_a": True}],
        "historical_recovered": [], "conflicts": [], "zero_coverage": [],
    })
    sources, report = load_sources(manifest, verify_hashes=False)
    assert [s.key for s in sources] == ["M__m1"]
    assert report["unmanifested_on_disk"] == ["predictions/STRAY__m9.parquet"]


# --------------------------------------------------------------------------
# 2. HUC2 must come from the authoritative huc_cd, not the station prefix
# --------------------------------------------------------------------------


def test_huc2_uses_graph_nodes_not_station_prefix():
    dataset = {"site_no": ["06438000", "03010958", "07374000"]}
    table = analyze.huc2_lookup(dataset).set_index("station")
    # 06438000 is HUC2 10; its id starts with "06"
    assert int(table.loc["06438000", "huc2"]) == 10
    assert table.loc["06438000", "huc2_source"] == "graph_nodes.huc_cd"
    # 03010958 is HUC2 05; its id starts with "03"
    assert int(table.loc["03010958", "huc2"]) == 5
    assert int(table.loc["07374000", "huc2"]) == 8
    # the id prefix must disagree, which is exactly why it cannot be used
    assert int(table.loc["06438000", "huc2"]) != int("06438000"[:2])

    every = analyze.huc2_lookup({"site_no": list(
        pd.read_csv(ROOT / "data/processed/graph_nodes.csv",
                    dtype={"site_no": str})["site_no"])})
    assert (every["huc2"] != -1).all(), "every graph node has a HUC2"
    values = set(every["huc2"].astype(int))
    assert values.issubset(set(range(1, 23))), f"not a HUC2 region: {values}"
    # this dataset is the Mississippi basin; anything outside is a parsing bug
    assert values == {5, 6, 7, 8, 10, 11}, values


def test_huc2_code_parsing_handles_every_level():
    """huc_cd is stored as an int, so lengths vary and the level must be read."""
    assert analyze.huc2_from_code(5010001) == 5        # 05010001, leading 0 lost
    assert analyze.huc2_from_code(8070100) == 8        # 08070100
    assert analyze.huc2_from_code(101202021305) == 10  # HUC12 keeps its digits
    assert analyze.huc2_from_code(60300020403) == 6    # 11 digits, 0-stripped
    assert analyze.huc2_from_code(None) is None
    assert analyze.huc2_from_code("") is None
    assert analyze.huc2_from_code("nan") is None
    assert analyze.huc2_from_code(123) is None


def test_station_id_keeps_leading_zero():
    assert analyze.normalize_site(6438000) == "06438000"
    assert analyze.normalize_site("06438000") == "06438000"
    assert analyze.normalize_site("383703104423901") == "383703104423901"


# --------------------------------------------------------------------------
# 3. identity must cover dataset and mask CONTENT
# --------------------------------------------------------------------------


def test_config_hash_changes_with_dataset_and_mask(tmp_path):
    from river_graph.experiments.provenance import (
        config_hash,
        config_payload,
        file_identity,
    )
    ds_a = tmp_path / "a.pt"
    ds_b = tmp_path / "a2.pt"
    mk = tmp_path / "m.npz"
    ds_a.write_bytes(b"dataset-A")
    ds_b.write_bytes(b"dataset-B")
    mk.write_bytes(b"mask")
    params = {"model_name": "X", "seed": 0}

    base = config_hash(config_payload(params, file_identity(ds_a),
                                      file_identity(mk)))
    other_ds = config_hash(config_payload(params, file_identity(ds_b),
                                          file_identity(mk)))
    other_mask = config_hash(config_payload(
        params, file_identity(ds_a), file_identity(tmp_path / "missing.npz")))
    assert base != other_ds, "a different dataset must change the identity"
    assert base != other_mask, "a different mask must change the identity"


def test_identity_problems_reports_the_changed_field(tmp_path):
    from river_graph.experiments.provenance import (
        config_payload,
        file_identity,
        identity_problems,
    )
    ds_a = tmp_path / "a.pt"
    ds_b = tmp_path / "b.pt"
    mk = tmp_path / "m.npz"
    ds_a.write_bytes(b"A")
    ds_b.write_bytes(b"B")
    mk.write_bytes(b"m")
    want = config_payload({"seed": 0}, file_identity(ds_a), file_identity(mk))
    meta = {"config": dict(want)}
    assert identity_problems(meta, want) == []
    problems = identity_problems(meta, config_payload(
        {"seed": 0}, file_identity(ds_b), file_identity(mk)))
    assert any("dataset_sha256" in p for p in problems)
    assert identity_problems(None, want) == ["no provenance sidecar"]


# --------------------------------------------------------------------------
# fixtures for the runner tests
# --------------------------------------------------------------------------


class CountingModel:
    instances = 0

    def __init__(self, **kwargs):
        type(self).instances += 1

    def fit_predict(self, dataset, split):
        y = dataset["y"].numpy()
        return np.where(dataset["y_mask"].numpy(), y + 0.25, np.nan)


class ShiftedModel(CountingModel):
    """Predicts a larger offset, so metrics differ from CountingModel."""

    def fit_predict(self, dataset, split):
        y = dataset["y"].numpy()
        return np.where(dataset["y_mask"].numpy(), y + 5.0, np.nan)


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    ds_dir = tmp_path / "data"
    ds_dir.mkdir()
    ds_path = ds_dir / "synthetic.pt"
    torch.save(_dataset(), ds_path)
    masks = tmp_path / "masks"
    masks.mkdir()
    flat = np.flatnonzero(_dataset()["y_mask"].numpy().ravel())
    np.savez(masks / "m1.npz", train=flat[:12], val=flat[12:14], test=flat[14:])
    results = tmp_path / "results"
    results.mkdir()
    preds = tmp_path / "preds"
    preds.mkdir()
    monkeypatch.setattr(run_gnn, "load_dataset", lambda _p: _dataset())
    CountingModel.instances = 0
    return {"dataset": str(ds_path), "masks": masks, "results": results,
            "preds": preds, "tmp": tmp_path}


def _args(sandbox, **over):
    class A:
        pass

    a = A()
    a.only = None
    a.variants = ["river"]
    a.lr = 1e-3
    a.arch = "gcn"
    a.dataset = sandbox["dataset"]
    a.tag = "T"
    a.model_name = None
    a.share_weights = False
    a.edge_dropout = 0.0
    a.wd = 0.0
    a.seed = 0
    a.save_predictions = False
    a.rebuild_predictions = False
    a.force = False
    a.env_groups = None
    a.verify_predictions = False
    for k, v in over.items():
        setattr(a, k, v)
    return a


def _run(sandbox, args, factory=CountingModel):
    return run_gnn.run(args, model_factory=factory,
                       results_dir=sandbox["results"],
                       masks_dir=sandbox["masks"], pred_dir=sandbox["preds"])


# --------------------------------------------------------------------------
# 4. an interrupted pair must not be accepted as a complete cache
# --------------------------------------------------------------------------


def test_interrupted_pair_is_repaired_not_trusted(sandbox):
    _run(sandbox, _args(sandbox, save_predictions=True))
    stored = json.loads((sandbox["results"] / "T_river.json").read_text())
    assert abs(stored["m1"]["mae"] - 0.25) < 1e-9

    # a newer run wrote predictions but died before updating the metrics JSON
    _run(sandbox, _args(sandbox, save_predictions=True, force=True),
         factory=ShiftedModel)
    (sandbox["results"] / "T_river.json").write_text(json.dumps(stored),
                                                     encoding="utf-8")
    CountingModel.instances = 0

    report = _run(sandbox, _args(sandbox, save_predictions=True))

    assert report.metric_drift == ["T_river__m1"], "drift must be detected"
    assert report.trained == 0, "repair must not retrain"
    assert CountingModel.instances == 0
    repaired = json.loads((sandbox["results"] / "T_river.json").read_text())
    assert abs(repaired["m1"]["mae"] - 5.0) < 1e-9, \
        "metrics must be repaired from the stored predictions"


def test_consistent_pair_still_skips(sandbox):
    _run(sandbox, _args(sandbox, save_predictions=True))
    CountingModel.instances = 0
    report = _run(sandbox, _args(sandbox, save_predictions=True))
    assert report.skipped == 1
    assert report.metric_drift == []
    assert report.trained == 0
    assert CountingModel.instances == 0


# --------------------------------------------------------------------------
# 5. rebuild / force must store predictions
# --------------------------------------------------------------------------


def test_rebuild_implies_save_and_stores_predictions(sandbox):
    (sandbox["results"] / "T_river.json").write_text(
        json.dumps({"m1": {"mae": 9.9, "r2": -9.0, "n": 1}}), encoding="utf-8")

    report = _run(sandbox, _args(sandbox, rebuild_predictions=True))

    assert report.trained == 1
    assert report.missing_predictions == []
    assert (sandbox["preds"] / "T_river__m1.parquet").exists(), \
        "--rebuild-predictions trained but stored no prediction"
    assert (sandbox["preds"] / "T_river__m1.meta.json").exists()
    stored = json.loads((sandbox["results"] / "T_river.json").read_text())
    assert abs(stored["m1"]["mae"] - 0.25) < 1e-9


def test_force_implies_save(sandbox):
    _run(sandbox, _args(sandbox, save_predictions=True))
    report = _run(sandbox, _args(sandbox, force=True), factory=ShiftedModel)
    assert report.trained == 1
    meta = run_gnn.read_meta("T_river", "m1", sandbox["preds"])
    assert abs(meta["metrics"]["mae"] - 5.0) < 1e-9, \
        "the sidecar must describe the predictions it sits next to"


# --------------------------------------------------------------------------
# 6. identity conflicts must fail the process
# --------------------------------------------------------------------------


def test_identity_conflict_reports_failure(sandbox):
    from river_graph.experiments.predictions import save_predictions
    from river_graph.experiments.provenance import (
        build_meta,
        config_payload,
        file_identity,
    )
    ds = _dataset()
    split = {"train": np.array([0, 1]), "val": np.array([2]),
             "test": np.array([3, 4, 5])}
    other_ds = sandbox["tmp"] / "other.pt"
    torch.save(ds, other_ds)
    meta = build_meta(
        model_name="T_river", mask_name="m1", dataset_path=other_ds,
        split=split, params={"seed": 0},
        masks_dir=sandbox["masks"])
    save_predictions(np.zeros((N, T)), ds, split, "T_river", "m1", "synthetic",
                     out_dir=sandbox["preds"], meta=meta)
    (sandbox["results"] / "T_river.json").write_text(
        json.dumps({"m1": {"mae": 1.0, "r2": 0.1, "n": 3}}), encoding="utf-8")

    report = _run(sandbox, _args(sandbox, save_predictions=True))

    assert report.identity_mismatch == ["T_river__m1"]
    assert report.failed, "an identity conflict must be a failure"
    assert report.identity_reasons["T_river__m1"]
    # sanity: the identity really does differ
    want = config_payload({"seed": 0}, file_identity(sandbox["dataset"]),
                          file_identity(sandbox["masks"] / "m1.npz"))
    assert meta["config"]["dataset_sha256"] != want["dataset_sha256"]


def test_cli_exit_code_for_identity_conflict(sandbox, monkeypatch):
    """main() must not exit 0 when runs were refused."""
    import argparse

    def fake_parse():
        return _args(sandbox, save_predictions=True)

    report = run_gnn.RunReport(identity_mismatch=["T_river__m1"],
                               identity_reasons={"T_river__m1": ["x"]})
    monkeypatch.setattr(argparse.ArgumentParser, "parse_args",
                        lambda self: fake_parse())
    monkeypatch.setattr(run_gnn, "run", lambda args: report)

    with pytest.raises(SystemExit) as excinfo:
        run_gnn.main()
    assert excinfo.value.code not in (0, None), \
        "identity conflicts must return a non-zero exit code"


def test_cli_exit_code_zero_when_clean(sandbox, monkeypatch):
    import argparse

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args",
                        lambda self: _args(sandbox, save_predictions=True))
    monkeypatch.setattr(run_gnn, "run", lambda args: run_gnn.RunReport())
    monkeypatch.setattr(run_gnn, "rebuild_flat_csv", lambda *a, **k: None)
    run_gnn.main()  # must not raise


# --------------------------------------------------------------------------
# 7. bootstrap must resample stations, keeping all of a station's rows
# --------------------------------------------------------------------------


def test_bootstrap_is_station_clustered():
    rows = []
    # two stations, each appearing under two masks of the same scenario
    for station, base in (("A", 1.0), ("B", 3.0), ("C", 5.0)):
        for mask in ("m1", "m2"):
            rows.append({"model": "M", "scenario": "E1", "mask": mask,
                         "station": station, "n_predicted": 10,
                         "mae": base, "rmse": base})
    df = pd.DataFrame(rows)
    out = analyze.station_bootstrap(df, n_boot=200, seed=0)
    assert len(out) == 1
    row = out.iloc[0]
    assert int(row["n_stations_resampled"]) == 3, \
        "the resampling unit must be the unique station, not the row"
    assert int(row["n_rows_used"]) == 6
    assert int(row["rows_per_station_max"]) == 2
    assert "station" in row["bootstrap_unit"]
    # cell-weighted point estimate over 60 cells: (1+3+5)*20/60 = 3.0
    assert abs(row["mae_point"] - 3.0) < 1e-9
    assert row["mae_lo95"] <= row["mae_point"] <= row["mae_hi95"]

