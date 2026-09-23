"""Fail-closed contracts of the Phase-2B/2C executor (no training).

The executor must refuse to dispatch on any frozen-policy mismatch: wrong
policy hash, missing h2x policy keys, wrong dataset/mask hash, or a stale
runtime snapshot pin. These tests tamper with each input in turn.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_executor():
    spec = importlib.util.spec_from_file_location(
        "run2b_executor_test", ROOT / "scripts" / "run2b_executor.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run2b_executor_test"] = mod
    spec.loader.exec_module(mod)
    return mod


ex = _load_executor()


def test_policies_load_on_the_frozen_tree():
    pol = ex.load_and_check_policies()
    assert set(ex.ARMS) == {"H2", "H2E", "H2X", "H2X_nomsg"}
    assert pol["pol2c"]["seed_set"] == [42, 43, 44, 45, 46]
    assert pol["h2x"]["allow_nearest_distance_km_max"] is None


def test_wrong_policy_hash_is_refused(monkeypatch, tmp_path):
    tampered = tmp_path / "phase2_2c_policy.json"
    data = json.loads(ex.POL2C_PATH.read_text(encoding="utf-8"))
    data["h2x_policy_sha256"] = "0" * 64
    tampered.write_text(json.dumps(data), encoding="utf-8")
    monkeypatch.setattr(ex, "POL2C_PATH", tampered)
    with pytest.raises(ex.PolicyError, match="h2x_policy_sha256"):
        ex.load_and_check_policies()


def test_missing_h2x_key_is_refused(monkeypatch, tmp_path):
    import hashlib

    tampered = tmp_path / "h2x_policy.json"
    data = json.loads(ex.H2X_PATH.read_text(encoding="utf-8"))
    del data["hard_input_exclusions"]
    tampered.write_text(json.dumps(data), encoding="utf-8")
    # keep both hash bindings valid so the missing-key check is what fires
    new_sha = hashlib.sha256(tampered.read_bytes()).hexdigest()
    good = json.loads(ex.POL2C_PATH.read_text(encoding="utf-8"))
    good["h2x_policy_sha256"] = new_sha
    pol2c = tmp_path / "phase2_2c_policy.json"
    pol2c.write_text(json.dumps(good), encoding="utf-8")
    cfg_data = json.loads(ex.CFG_PATH.read_text(encoding="utf-8"))
    cfg_data["policy_binding"]["h2x_policy_sha256"] = new_sha
    cfg_path = tmp_path / "phase2_ablation_stcore_v1.json"
    cfg_path.write_text(json.dumps(cfg_data), encoding="utf-8")
    monkeypatch.setattr(ex, "H2X_PATH", tampered)
    monkeypatch.setattr(ex, "POL2C_PATH", pol2c)
    monkeypatch.setattr(ex, "CFG_PATH", cfg_path)
    with pytest.raises(ex.PolicyError, match="hard_input_exclusions"):
        ex.load_and_check_policies()


def test_wrong_dataset_hash_is_refused(monkeypatch, tmp_path):
    tampered = tmp_path / "phase2_2c_policy.json"
    data = json.loads(ex.POL2C_PATH.read_text(encoding="utf-8"))
    data["dataset_sha256"] = "f" * 64
    tampered.write_text(json.dumps(data), encoding="utf-8")
    monkeypatch.setattr(ex, "POL2C_PATH", tampered)
    with pytest.raises(ex.PolicyError, match="dataset"):
        ex.load_and_check_policies()


def test_stale_runtime_snapshot_pin_is_refused(monkeypatch, tmp_path):
    pin = tmp_path / "runtime_snapshot.pin"
    pin.write_text("0" * 64 + "\n", encoding="utf-8")
    monkeypatch.setattr(ex, "SNAPSHOT_PIN", pin)
    monkeypatch.setattr(ex, "RECORDS", tmp_path)
    with pytest.raises(ex.PolicyError, match="stale runtime snapshot"):
        ex.check_or_pin_runtime_snapshot()
