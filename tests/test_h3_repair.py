"""Regression tests for the R1-R3 repair round.

Every test here corresponds to a defect that was found by review in the
T01-T16 round, so the same failure would be caught again:

  R1  a provider covariate value that cannot be a measurement reached the
      model inputs
  R2  the committed repository could not construct the H3X models at all
  R3a the "E2a" mask still saw future DOC context
  R3b expansion verification would skip the reused pilot records
  R3c the outer-test export was unlocked by a pilot promotion alone
  R3d the truncation report was hard-coded
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from river_graph.data.quality import (
    RULES,
    apply_rules,
    rules_digest,
    summarize,
)
from river_graph.experiments.h3_masks import assert_visible_roles
from river_graph.experiments.h3_runs import (
    artifact_entries,
    artifact_manifest_hash,
    identity_mismatches,
    require_decision,
    task_grid,
)
from river_graph.experiments.h3_training import load_protocol
from river_graph.models.h3 import (
    ENV_FEATURE_NAMES,
    EnvironmentalPredictor,
    H3ResidualImputer,
    make_h2x_trunk,
)

ROOT = Path(__file__).resolve().parents[1]
V04 = ROOT / "data/processed/mississippi_graph_v04.pt"
V05 = ROOT / "data/processed/mississippi_graph_v05.pt"
RAW_STATION = ROOT / "data/raw/wqp_results/05357225.csv"


# --------------------------------------------------------------- R1: rules


def frame(rows) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["site_no", "date", "value", "unit"])


def test_the_1310c_water_temperature_is_rejected():
    audited = apply_rules(
        frame([("05357225", "2017-06-26", 1310.0, "deg C")]), "temperature"
    )
    assert audited.qc_status.tolist() == ["rejected"]
    assert audited.qc_reason.tolist() == ["above_physical_max"]


def test_ordinary_water_temperatures_are_kept():
    audited = apply_rules(
        frame(
            [
                ("a", "2000-01-01", 0.2, "deg C"),
                ("b", "2000-07-01", 25.4, "deg C"),
                ("c", "2000-08-01", 38.0, "deg C"),
                ("d", "2000-01-15", -0.3, "deg C"),
                ("e", "2000-02-01", -4.0, "deg C"),
            ]
        ),
        "temperature",
    )
    assert audited.qc_status.tolist() == ["accepted"] * 5


def test_fahrenheit_rows_are_marked_missing_not_converted():
    """The deg F label is not trustworthy here, so no conversion is applied."""
    audited = apply_rules(
        frame(
            [
                ("05420500", "2017-05-15", 18.4, "deg F"),
                ("06306300", "2014-07-23", 25.5, "deg F"),
            ]
        ),
        "temperature",
    )
    assert audited.qc_status.tolist() == ["rejected", "rejected"]
    assert audited.qc_reason.tolist() == ["unsupported_unit"] * 2
    assert audited.value.tolist() == [18.4, 25.5]  # raw value kept for the record


def test_missing_unit_and_missing_value_are_distinguished():
    audited = apply_rules(
        frame([("a", "2000-01-01", 10.0, None), ("b", "2000-01-01", None, "deg C")]),
        "temperature",
    )
    assert audited.qc_reason.tolist() == ["missing_unit", "missing_value"]


def test_ph_outside_zero_to_fourteen_is_rejected():
    audited = apply_rules(
        frame([("03085000", "1981-09-02", 75.0, "standard units")]), "ph"
    )
    assert audited.qc_reason.tolist() == ["above_physical_max"]


def test_negative_discharge_is_kept_because_reverse_flow_is_real():
    audited = apply_rules(
        frame([("08012150", "2000-01-01", -3150.0, "ft3/s")]), "discharge"
    )
    assert audited.qc_status.tolist() == ["accepted"]


def test_doc_has_no_value_range_rule():
    """A high DOC must never be removed for being hard to predict."""
    rule = RULES["doc"]
    assert rule["min"] is None and rule["max"] is None
    audited = apply_rules(
        frame([("a", "2000-01-01", 445.0, "mg/L"), ("b", "2000-01-01", 4160.0, "mg/L")]),
        "doc",
    )
    assert audited.qc_status.tolist() == ["accepted", "accepted"]


def test_rule_digest_is_stable_and_changes_with_the_rules(monkeypatch):
    first = rules_digest()
    assert first == rules_digest()
    monkeypatch.setitem(RULES["temperature"], "max", 39.0)
    assert rules_digest() != first


def test_summarize_counts_by_variable_and_reason():
    audited = pd.concat(
        [
            apply_rules(
                frame([("a", "2000-01-01", 1310.0, "deg C")]), "temperature"
            ),
            apply_rules(frame([("b", "2000-01-01", 12.0, "deg C")]), "temperature"),
        ],
        ignore_index=True,
    )
    summary = summarize(audited)
    assert summary["rows"] == 2 and summary["accepted"] == 1
    assert summary["by_reason"] == {"temperature:above_physical_max": 1}


@pytest.mark.skipif(not RAW_STATION.is_file(), reason="raw cache not present")
def test_the_real_cached_record_is_rejected():
    from river_graph.data.wqp import load_station_results

    results = load_station_results(RAW_STATION)
    sub = results[results["variable"] == "temperature"]
    row = sub[sub["date"] == "2017-06-26"]
    assert len(row) == 1 and float(row["value"].iloc[0]) == 1310.0
    audited = apply_rules(
        pd.DataFrame(
            {
                "site_no": row["site_no"].values,
                "date": row["date"].astype(str).values,
                "value": row["value"].values,
                "unit": row["unit"].values,
            }
        ),
        "temperature",
    )
    assert audited.qc_status.tolist() == ["rejected"]


@pytest.mark.skipif(
    not (V04.is_file() and V05.is_file()), reason="dataset versions not present"
)
def test_v05_fixes_the_temperature_channel_and_keeps_the_doc_labels():
    v04 = torch.load(V04, weights_only=False)
    v05 = torch.load(V05, weights_only=False)
    rule = RULES["temperature"]
    temp = v05["x"][:, :, 0]
    assert float(temp.min()) >= rule["min"]
    assert float(temp.max()) <= rule["max"]
    # the defect that motivated the rebuild is really gone
    assert float(v04["x"][:, :, 0].max()) > rule["max"]
    # and nothing else moved
    assert torch.equal(v04["y"], v05["y"])
    assert torch.equal(v04["y_mask"], v05["y_mask"])
    assert torch.equal(v04["regime"], v05["regime"])
    assert torch.equal(v04["edge_attr"], v05["edge_attr"])


# ------------------------------------------------------- R2: dependencies


def test_h2x_trunk_uses_only_the_released_interface():
    trunk = make_h2x_trunk(14, 6, hidden=64, layers=2, dropout=0.1, env_dim=9,
                           env_emb=32)
    assert sum(p.numel() for p in trunk.parameters()) == 23461
    model = H3ResidualImputer(EnvironmentalPredictor(21, 64, 2, 0.1), trunk)
    info = model.initialise(0, 200003)
    assert info["correction_linear_layers"] == 17
    head = model.correction.head
    assert torch.equal(head.weight, torch.zeros_like(head.weight))


def test_a_non_static_gate_mode_is_refused_rather_than_ignored():
    with pytest.raises(ValueError, match="static"):
        make_h2x_trunk(14, 6, gate_mode="dynamic")


def test_env_arm_constructs_from_the_environmental_view_only():
    model = EnvironmentalPredictor(len(ENV_FEATURE_NAMES), 16, 2, 0.0)
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(4, len(ENV_FEATURE_NAMES)))
    assert out.shape == (4,) and torch.isfinite(out).all()


# ------------------------------------------------- R3a: mask visibility


def test_a_mask_without_visible_roles_is_refused():
    with pytest.raises(ValueError, match="visible_roles"):
        assert_visible_roles({"train": np.array([0]), "val": np.array([1])})


def test_a_mask_may_never_declare_val_or_test_visible():
    for role in ("val", "test"):
        split = {
            "train": np.array([0], dtype=np.int64),
            "val": np.array([1], dtype=np.int64),
            "test": np.array([2], dtype=np.int64),
            "visible_roles": np.array(["train", role], dtype="<U16"),
        }
        with pytest.raises(ValueError, match="never open"):
            assert_visible_roles(split)


def test_unknown_or_absent_visible_roles_are_refused():
    split = {
        "train": np.array([0], dtype=np.int64),
        "visible_roles": np.array(["train", "nonsense"], dtype="<U16"),
    }
    with pytest.raises(ValueError, match="unknown visible role"):
        assert_visible_roles(split)
    split["visible_roles"] = np.array(["train", "val_context"], dtype="<U16")
    with pytest.raises(ValueError, match="not present"):
        assert_visible_roles(split)


# --------------------------------------------- R3b/R3c: identity and gates


def test_identity_mismatches_reports_source_protocol_and_data_differences():
    stored = {
        "seed": 0,
        "training": {"lr": 0.001},
        "dataset": {"dataset_sha256": "aaa"},
        "source_sha256": {"src/x.py": "111", "src/y.py": "222"},
    }
    expected = json.loads(json.dumps(stored))
    assert identity_mismatches(stored, expected) == []
    expected["training"]["lr"] = 999
    expected["dataset"]["dataset_sha256"] = "bbb"
    expected["source_sha256"]["src/x.py"] = "999"
    problems = identity_mismatches(stored, expected)
    assert any("training" in p for p in problems)
    assert any("dataset" in p for p in problems)
    assert any("source_sha256[src/x.py]" in p for p in problems)


def test_require_decision_refuses_missing_and_non_promoting_decisions(tmp_path):
    with pytest.raises(SystemExit, match="does not exist"):
        require_decision(tmp_path, "expand", "export")
    (tmp_path / "expansion_decision.json").write_text(
        json.dumps({"outcome": "structure_not_supported"}), encoding="utf-8"
    )
    with pytest.raises(SystemExit, match="not 'promote'"):
        require_decision(tmp_path, "expand", "export")
    (tmp_path / "expansion_decision.json").write_text(
        json.dumps({"outcome": "promote"}), encoding="utf-8"
    )
    assert require_decision(tmp_path, "expand", "export")["outcome"] == "promote"


def _fake_run_root(tmp_path: Path, protocol: dict, include_all: bool) -> Path:
    """A root holding run records for the whole expansion grid."""
    root = tmp_path / "root"
    (root / "runs").mkdir(parents=True)
    (root / "checkpoints").mkdir(parents=True)
    grid = task_grid("expand", protocol)
    for index, (arm, seed, mask) in enumerate(grid):
        if not include_all and index >= len(grid) - 5:
            continue
        stem = arm + "_s" + str(seed) + "__" + mask
        checkpoint = root / "checkpoints" / (stem + ".pt")
        checkpoint.write_bytes(b"weights-" + stem.encode())
        record = {
            "status": "complete",
            "stage": "pilot" if index < 27 else "expand",
            "config_hash": hashlib.sha256(stem.encode()).hexdigest(),
            "config": {"arm": arm, "seed": seed, "mask": mask},
            "artifacts": {
                "checkpoint": {"path": "checkpoints/" + stem + ".pt"}
            },
        }
        (root / "runs" / (stem + ".json")).write_text(
            json.dumps(record), encoding="utf-8"
        )
    return root


def _unlock_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "run_h3_frozen_eval", ROOT / "scripts/run_h3_frozen_eval.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_outer_test_export_is_refused_with_a_pilot_decision_only(tmp_path):
    protocol = load_protocol()
    root = _fake_run_root(tmp_path, protocol, include_all=True)
    (root / "pilot_decision.json").write_text(
        json.dumps({"outcome": "promote"}), encoding="utf-8"
    )
    module = _unlock_module()
    with pytest.raises(SystemExit, match="expansion_decision.json does not exist"):
        module.unlock(root, protocol, "architecture")


def test_outer_test_export_is_refused_when_the_expansion_grid_is_incomplete(
    tmp_path,
):
    protocol = load_protocol()
    root = _fake_run_root(tmp_path, protocol, include_all=False)
    (root / "pilot_decision.json").write_text(
        json.dumps({"outcome": "promote"}), encoding="utf-8"
    )
    (root / "expansion_decision.json").write_text(
        json.dumps({"outcome": "promote", "complete": True,
                    "artifact_manifest_sha256": "0" * 64}), encoding="utf-8"
    )
    module = _unlock_module()
    with pytest.raises(SystemExit, match="configurations are missing"):
        module.unlock(root, protocol, "architecture")


def test_outer_test_export_is_refused_when_the_run_set_changed(tmp_path):
    protocol = load_protocol()
    root = _fake_run_root(tmp_path, protocol, include_all=True)
    (root / "expansion_decision.json").write_text(
        json.dumps({"outcome": "promote", "complete": True,
                    "artifact_manifest_sha256": "0" * 64}), encoding="utf-8"
    )
    module = _unlock_module()
    with pytest.raises(SystemExit, match="no longer matches"):
        module.unlock(root, protocol, "architecture")


def test_outer_test_export_unlocks_only_on_a_matching_manifest(tmp_path):
    protocol = load_protocol()
    root = _fake_run_root(tmp_path, protocol, include_all=True)
    records = {}
    for path in sorted((root / "runs").glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        record["_manifest"] = str(path)
        records[(record["config"]["arm"], record["config"]["seed"],
                 record["config"]["mask"])] = record
    grid = [(arm, int(seed), mask) for arm, seed, mask in
            task_grid("expand", protocol)]
    digest = artifact_manifest_hash(
        artifact_entries([(records[key], Path(records[key]["_manifest"]))
                          for key in grid])
    )
    (root / "expansion_decision.json").write_text(
        json.dumps({"outcome": "promote", "complete": True,
                    "artifact_manifest_sha256": digest}), encoding="utf-8"
    )
    module = _unlock_module()
    unlocked = module.unlock(root, protocol, "architecture")
    assert unlocked["manifest_sha256"] == digest
    assert len(unlocked["grid"]) == 120
    # a network claim additionally needs the T18 arm, which is absent here
    with pytest.raises(SystemExit, match="no-message control is incomplete"):
        module.unlock(root, protocol, "network")


def test_verification_selects_by_grid_not_by_stored_stage():
    """R3b: reused pilot records must not be skipped by an expand filter."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "verify_h3", ROOT / "scripts/verify_h3.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    protocol = load_protocol()
    grid = task_grid("expand", protocol)
    records = {}
    for index, (arm, seed, mask) in enumerate(grid):
        records[(arm, seed, mask)] = {
            "config": {"arm": arm, "seed": seed, "mask": mask,
                       "stage": "pilot" if index < 27 else "expand"}
        }
    selected = module.select(records, grid)
    assert len(selected) == 120
    with pytest.raises(ValueError, match="does not match the stage grid"):
        module.select({k: v for k, v in list(records.items())[:-1]}, grid)
    extra = dict(records)
    extra[("env", 99, "e1_r20_seed42")] = {"config": {}}
    with pytest.raises(ValueError, match="does not match the stage grid"):
        module.select(extra, grid)


def test_strict_audit_uses_the_recomputed_identity(tmp_path):
    """The stored hash must be compared against a freshly computed identity."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "h3_runs_check", ROOT / "src/river_graph/experiments/h3_runs.py"
    )
    assert spec is not None
    protocol = load_protocol()
    stored = {
        "seed": 0,
        "training": protocol["training"],
        "dataset": {"dataset_sha256": "aaa"},
        "source_sha256": {"src/x.py": "111"},
    }
    expected = json.loads(json.dumps(stored))
    assert identity_mismatches(stored, expected) == []
    expected["source_sha256"]["src/x.py"] = "222"
    assert identity_mismatches(stored, expected)


# ------------------------------------------------------- R3d: clip reporting


def test_clipping_report_is_computed_from_the_runs():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "analyze_h3_pilot", ROOT / "scripts/analyze_h3_pilot.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    per_run = pd.DataFrame(
        [
            {"arm": "env", "seed": 0, "mask": "m", "scenario": "E2b", "cells": 100,
             "clipped_cells": 1, "log_rmse": 0.30, "raw_log_rmse": 0.37},
            {"arm": "h2x", "seed": 0, "mask": "m", "scenario": "E2b", "cells": 100,
             "clipped_cells": 3, "log_rmse": 0.29, "raw_log_rmse": 0.33},
            {"arm": "h3a", "seed": 0, "mask": "m", "scenario": "E2b", "cells": 100,
             "clipped_cells": 3, "log_rmse": 0.27, "raw_log_rmse": 0.32},
        ]
    )
    per_seed = per_run.copy()
    report = module.clipping_report(per_run, per_seed)
    # every arm is described, not just the one that used to be special-cased
    assert set(report["per_arm"]) == {"env", "h2x", "h3a"}
    assert report["per_arm"]["h3a"]["clipped_cells"] == 3
    assert report["per_arm"]["env"]["clipped_cells"] == 1
    assert pytest.approx(0.03) == report["per_arm"]["h2x"]["clipped_fraction"]
    assert report["per_arm"]["env"]["truncation_effect"] > 0
    assert report["e2b_seeds_better_than_both_clipped"] == [0]
