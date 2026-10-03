"""Verify that the matched 60-epoch experiment extends the original traces.

No model is fitted. This complements verify_doc_encoder_residual_v1.py, which
replays saved model predictions and validation-selected support adapters.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from run_unified_doc_spatial import verify_files, verify_runtime_snapshot

from river_graph.experiments.provenance import sha256_file

ROOT = Path("experiments/phase4_transfer/doc_encoder_budget_v1")
REFERENCE = Path("experiments/phase4_transfer/doc_encoder_residual_v1")
ARMS = ("frozen", "last_self", "last_self_ecology")
SHAPES = ("constant", "gru_tuned_anchor")
DISPATCH = (
    '    if len(sys.argv) > 2 and sys.argv[1:3] == ["--experiment", "doc-encoder-budget-v1"]:\n'
    '        from run_doc_encoder_budget_v1 import main as encoder_budget_main\n'
    '        sys.argv = [sys.argv[0], *sys.argv[3:]]\n'
    '        encoder_budget_main()\n'
    '        return\n'
)


def check_sources(reference, root):
    runtime = {name: verify_runtime_snapshot(path) for name, path in (("old", reference), ("new", root))}
    old = json.loads((reference / "runtime_snapshot.json").read_text())
    new = json.loads((root / "runtime_snapshot.json").read_text())
    changed = [key for key, value in old.items() if new.get(key) != value]
    if changed != ["scripts/run_ladder.py"]:
        raise ValueError(f"Unexpected shared execution source changes: {changed}")
    before = (reference / "code_snapshot/scripts/run_ladder.py").read_text()
    after = (root / "code_snapshot/scripts/run_ladder.py").read_text()
    if after.count(DISPATCH) != 1 or after.replace(DISPATCH, "") != before:
        raise ValueError("run_ladder change is not the expected dispatch-only addition")
    return runtime, {"path": "scripts/run_ladder.py", "added_dispatch_only": True,
                     "old_sha256": old["scripts/run_ladder.py"], "new_sha256": new["scripts/run_ladder.py"]}


def check_tensors(before, after, modules):
    count = 0
    for module in modules:
        if set(before[module]) != set(after[module]):
            raise ValueError(f"Checkpoint tensor keys changed: {module}")
        for key, tensor in before[module].items():
            torch.testing.assert_close(tensor, after[module][key], rtol=0, atol=0)
            count += 1
    return count


def check_query_frames(before, after, names):
    keys = ["model_name", "k", "cell"]
    old = before[before.model_name.isin(names)].sort_values(keys).reset_index(drop=True)
    new = after[after.model_name.isin(names)].sort_values(keys).reset_index(drop=True)
    if set(old.model_name) != set(names) or set(new.model_name) != set(names):
        raise ValueError("A required unchanged reference or early-stop query product is absent")
    pd.testing.assert_frame_equal(old, new, check_exact=True)


def check_run(reference, root, partition, seed):
    name = f"split{partition}_seed{seed}"
    old, new = reference / "runs" / name, root / "runs" / name
    oc = json.loads((old / "config.json").read_text())
    nc = json.loads((new / "config.json").read_text())
    if oc["epochs"] != 30 or nc["epochs"] != 60 or oc["patience"] != 5 or nc["patience"] != 5:
        raise ValueError("Expected only a 30-to-60 cap change with patience five")
    differences = [key for key in set(oc) | set(nc) if oc.get(key) != nc.get(key)]
    if set(differences) != {"epochs", "started_at", "runtime_snapshot_hash"}:
        raise ValueError(f"Unexpected configuration differences: {differences}")
    if (oc["split_seed"], oc["seed"]) != (partition, seed) or (nc["split_seed"], nc["seed"]) != (partition, seed):
        raise ValueError("Run identity differs from its requested partition and seed")
    verify_files(old, "complete.json", oc)
    verify_files(new, "complete.json", nc)
    of, nf = (pd.read_parquet(path / "full_grid.parquet") for path in (old, new))
    oq, nq = (pd.read_parquet(path / "predictions.parquet") for path in (old, new))
    if json.loads((old / "feature_definition.json").read_text()) != json.loads((new / "feature_definition.json").read_text()):
        raise ValueError("Regime feature definitions changed")
    for column in ("cell", "station", "month", "analyte", "visibility_role", "context_pred", "ecological_memory"):
        np.testing.assert_array_equal(of[column], nf[column])
    states = {label: {kind: json.loads((path / f"{kind}.json").read_text()) for kind in ("adapters", "mixers")}
              for label, path in (("old", old), ("new", new))}
    rows = []
    for arm in ARMS:
        before = json.loads((old / f"{arm}.json").read_text())
        after = json.loads((new / f"{arm}.json").read_text())
        if (before["config"]["epochs"] != 30 or after["config"]["epochs"] != 60
                or {key: value for key, value in before["config"].items() if key != "epochs"}
                != {key: value for key, value in after["config"].items() if key != "epochs"}):
            raise ValueError(f"Model settings changed beyond epoch cap: {name}/{arm}")
        if after["trace"][:len(before["trace"])] != before["trace"]:
            raise ValueError(f"Original training prefix differs: {name}/{arm}")
        stopped = before["trace"][-1]["stale_epochs"] >= before["config"]["patience"]
        if (after["best_epoch"] < before["best_epoch"]
                or after["validation_metrics"]["validation_mae"] > before["validation_metrics"]["validation_mae"]):
            raise ValueError("Extending the candidate trace lost an earlier validation checkpoint")
        b = torch.load(old / f"{arm}.pt", weights_only=False, map_location="cpu")
        a = torch.load(new / f"{arm}.pt", weights_only=False, map_location="cpu")
        if b["spatial_architecture"] != a["spatial_architecture"]:
            raise ValueError("Spatial architecture changed")
        initial_count = check_tensors(b, a, ("initial_spatial", "initial_temporal", "initial_decay"))
        selected_count = 0
        if stopped:
            if {**after, "config": {**after["config"], "epochs": 30}} != before:
                raise ValueError(f"Previously exhausted patience changed the summary: {name}/{arm}")
            selected_count = check_tensors(b, a, ("spatial", "temporal", "decay", "head"))
            for column in (f"{arm}_delta", f"{arm}_pred", f"{arm}_integrated_k0_pred"):
                np.testing.assert_array_equal(of[column], nf[column])
            names = [f"{arm}_{shape}" for shape in SHAPES] + [f"{arm}_integrated_{shape}" for shape in SHAPES]
            check_query_frames(oq, nq, names)
            for shape in SHAPES:
                if (states["old"]["adapters"][f"{arm}_{shape}"] != states["new"]["adapters"][f"{arm}_{shape}"]
                        or states["old"]["mixers"][f"{arm}_integrated_{shape}"]
                        != states["new"]["mixers"][f"{arm}_integrated_{shape}"]):
                    raise ValueError("An unchanged early-stop model changed its support adaptation")
        elif before["epochs_run"] != 30 or after["epochs_run"] <= 30:
            raise ValueError("An unexhausted 30-epoch fit did not continue")
        rows.append({"run": name, "arm": arm, "old_epochs_run": before["epochs_run"], "new_epochs_run": after["epochs_run"],
                     "old_best_epoch": before["best_epoch"], "new_best_epoch": after["best_epoch"],
                     "old_best_validation_mae": before["validation_metrics"]["validation_mae"],
                     "new_best_validation_mae": after["validation_metrics"]["validation_mae"],
                     "exact_trace_prefix_rows": len(before["trace"]), "trace_prefix_exact": True,
                     "old_patience_exhausted": stopped, "original_tensor_checks": initial_count,
                     "early_stop_selected_tensor_checks": selected_count,
                     "early_stop_summary_weights_fullgrid_queries_adapters_unchanged": True if stopped else None})
    references = [f"{base}_{shape}" for base in ("context", "prior_concentration", "prior_ecological_affine") for shape in SHAPES]
    check_query_frames(oq, nq, references)
    package = {"run": name, "old_complete_sha256": sha256_file(old / "complete.json"),
               "new_complete_sha256": sha256_file(new / "complete.json"), "config_differences": sorted(differences),
               "context_and_frozen_reference_queries_bitwise_unchanged": True}
    return package, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--reference-root", type=Path, default=REFERENCE)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    runtime, dispatch = check_sources(args.reference_root, args.root)
    packages, rows = [], []
    for partition in (142, 143, 144):
        for seed in (42, 43, 44):
            package, checked = check_run(args.reference_root, args.root, partition, seed)
            packages.append(package)
            rows.extend(checked)
    report = {
        "checked_at": datetime.now(timezone.utc).isoformat(), "old_root": str(args.reference_root), "new_root": str(args.root),
        "verifier_sha256": sha256_file(__file__),
        "shared_verifier_helpers_sha256": sha256_file("scripts/run_unified_doc_spatial.py"),
        "runtime_snapshot_hashes": runtime, "shared_model_and_underlying_runner_sources_identical": True,
        "routing_only_snapshot_difference": dispatch, "verified_packages": len(packages), "models": len(rows),
        "old_patience_exhausted_models": sum(row["old_patience_exhausted"] for row in rows),
        "continued_beyond_thirty_models": sum(row["new_epochs_run"] > 30 for row in rows),
        "total_extra_epochs": sum(row["new_epochs_run"] - row["old_epochs_run"] for row in rows),
        "exact_prefix_rows": sum(row["exact_trace_prefix_rows"] for row in rows),
        "original_tensor_checks": sum(row["original_tensor_checks"] for row in rows),
        "early_stop_selected_tensor_checks": sum(row["early_stop_selected_tensor_checks"] for row in rows),
        "protocol": "Only the maximum epoch setting changes from 30 to 60; patience 5 and original initialization stay fixed. "
                    "Every old trace row, including epoch 0, is compared exactly. Previously exhausted patience additionally "
                    "requires exact selected state, products and adapters. No model is fitted by this audit.",
        "packages": packages, "results": rows, "status": "verified",
    }
    path = args.output or args.root / "verification" / "budget_continuation_checks.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(path)


if __name__ == "__main__":
    main()
