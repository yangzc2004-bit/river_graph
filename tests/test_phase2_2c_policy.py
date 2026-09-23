"""Frozen 2C policy contracts (configs/phase2_2c_policy.json + spec §8).

Pins the pre-committed 2C semantics before any 2C run exists: the 5-seed set,
the E2a/E2b no-equivalence boundary, the >=10% union gate with per-scenario
no-harm margins, the h2x policy hash binding, and the required run-identity
fields. No scientific claim is produced here.
"""

import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads(
    (ROOT / "configs" / "phase2_ablation_stcore_v1.json").read_text(encoding="utf-8")
)
POL = json.loads(
    (ROOT / "configs" / "phase2_2c_policy.json").read_text(encoding="utf-8")
)
H2X = json.loads(
    (ROOT / "experiments" / "phase2_ablation_stcore_v1" / "h2x_policy.json").read_text(
        encoding="utf-8"
    )
)
ENDPOINTS = json.loads(
    (ROOT / "docs" / "paper" / "primary_endpoints.json").read_text(encoding="utf-8")
)
SPEC = (ROOT / "docs" / "paper" / "phase2_ablation_spec.md").read_text(encoding="utf-8")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_final_2c_matrix_is_frozen_to_5_seeds_240_configs():
    fin = CFG["stages"]["final_2c"]
    assert fin["seeds"] == [42, 43, 44, 45, 46]
    assert fin["configs"] == 240
    assert fin["claims_allowed"] is True
    assert POL["seed_set"] == [42, 43, 44, 45, 46]
    assert POL["arm_set"] == ["H2", "H2E", "H2X", "H2X_nomsg"]
    assert len(POL["mask_set"]) == 8


def test_no_equivalence_boundary_is_frozen_in_config_policy_and_spec():
    sp = CFG["scenario_policy"]
    assert set(sp["families"]) == {"e2a_strict", "e2b_partial", "e3_spatial"}
    assert "never be pooled" in sp["no_equivalence_boundary"]
    assert "never licenses" in sp["no_equivalence_boundary"]
    assert "non-operative" in sp["pilot_reconciliation"]
    assert "15%" in sp["pilot_reconciliation"]
    est = POL["estimands"]
    assert "never be pooled" in est["no_equivalence_boundary"]
    assert "without post-cutoff DOC context" in est["e2a"]
    assert "running network" in est["e2b"]
    assert "8.1" in SPEC and "no-equivalence" in SPEC.lower()
    assert "8.3" in SPEC and "non-operative" in SPEC


def test_gates_match_endpoints_v1_plus_per_scenario_margins():
    gates = POL["gates"]
    assert ">=10%" in gates["primary"] and "2 of {E2a, E2b, E3}" in gates["primary"]
    margins = gates["no_harm_margins"]
    assert set(margins) == {"e1_global", "e2a_per_scenario", "e2b_per_scenario",
                            "e3_per_scenario"}
    assert all("5%" in v for v in margins.values())
    assert "3 of 5" in gates["direction_consistency"]
    assert "H2X >= H2E >= H2X_nomsg" in gates["dominance_preference"]
    assert "never reported as a positive topology claim" in gates["dominance_preference"]
    # endpoints file still carries the same operative primary gate
    ep = ENDPOINTS["endpoints"]["doc_mae"]["gates"]
    assert ">=10%" in ep["phase2_h2x_vs_best_nongraph"]
    assert ENDPOINTS["version"] == "primary_endpoints_v1"
    assert "3 of 5" in ep["phase2_seed_consistency"] or "3 seeds" in ep["phase2_seed_consistency"]


def test_hash_bindings_are_live():
    assert POL["h2x_policy_sha256"] == sha(
        ROOT / "experiments/phase2_ablation_stcore_v1/h2x_policy.json"
    )
    assert POL["endpoints_sha256"] == sha(ROOT / "docs/paper/primary_endpoints.json")
    assert CFG["policy_binding"]["h2x_policy_sha256"] == POL["h2x_policy_sha256"]
    ds = ROOT / POL["dataset_path"]
    if ds.is_file():
        assert sha(ds) == POL["dataset_sha256"]
    else:
        pytest.skip("local ST-core dataset missing")
    for name, want in POL["mask_sha256s"].items():
        got = sha(ROOT / POL["masks_path"] / f"{name}.npz")
        assert got == want, name


def test_h2x_policy_forbids_distance_inputs_and_keeps_treatment_statement():
    assert H2X["allow_nearest_distance_km_max"] is None
    excl = H2X["hard_input_exclusions"]
    joined = " ".join(excl).lower()
    for needle in ("network distance", "nearest", "basin mean pooling", "ground-truth"):
        assert needle in joined, needle
    assert "transport" in H2X["treatment_statement"].lower()
    assert "encoder" in H2X["treatment_statement"].lower()
    assert "ensemble" in H2X["treatment_statement"].lower()


def test_run_identity_fields_are_required_by_policy():
    required = ["config_hash", "run_identity_sha256", "runtime_code_snapshot_sha256"]
    assert POL["run_identity_required"] == required
    assert CFG["policy_binding"]["run_policy"] == (
        "experiments/phase2_ablation_stcore_v1/phase2_run_policy.json"
    )
    run_pol = json.loads(
        (ROOT / "experiments/phase2_ablation_stcore_v1" / "phase2_run_policy.json")
        .read_text(encoding="utf-8")
    )
    assert run_pol["run_identity_required"] == required
    for legacy in run_pol["never_write"]:
        assert legacy in CFG["output"]["never_write"]
