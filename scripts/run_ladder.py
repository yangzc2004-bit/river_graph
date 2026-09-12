"""Standard entry point for real training runs: always store predictions.

``scripts/run_gnn.py`` keeps ``--save-predictions`` as an explicit opt-in flag,
because existing analysis and the frozen tables were produced without it and
because silently enabling it would change what a bare invocation does. For
*new* paper runs that ambiguity is undesirable: a run whose predictions are
not stored cannot be audited later, and the historical R1 defect came exactly
from that gap.

This wrapper therefore forces ``--save-predictions`` on, forwards every other
argument to the runner verbatim, and adds one guarantee the raw runner cannot
make: it refuses to proceed if the run would overwrite an existing prediction
whose provenance config hash differs, unless ``--force`` is given.

Usage (identical to run_gnn.py, plus --dry-run):
    python scripts/run_ladder.py --only e1_r20_seed42 --variants river \
        --arch directed --seed 0 --model-name H1_directed_river_s0
    python scripts/run_ladder.py --arch transport_enc --dataset \
        data/processed/mississippi_graph_v04.pt --model-name H2X_new --dry-run
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

from river_graph.experiments.predictions import (
    PRED_DIR,
    cache_state,
    prediction_path,
    read_meta,
)
from river_graph.experiments.provenance import config_hash, identity_problems


def load_runner() -> ModuleType:
    """Import scripts/run_gnn.py lazily (it pulls in torch, which dry-run
    does not need)."""
    if "run_gnn_module" in sys.modules:
        return sys.modules["run_gnn_module"]
    spec = importlib.util.spec_from_file_location(
        "run_gnn_module", Path(__file__).resolve().parent / "run_gnn.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_gnn_module"] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Training wrapper that always stores per-cell predictions.")
    ap.add_argument("--only", default=None)
    ap.add_argument("--variants", nargs="+", default=["river"])
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--arch", default="gcn",
                    choices=["gcn", "directed", "transport", "transport_enc"])
    ap.add_argument("--dataset", default="data/processed/mississippi_graph_v02.pt")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--model-name", default=None)
    ap.add_argument("--share-weights", action="store_true")
    ap.add_argument("--edge-dropout", type=float, default=0.0)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--env-groups", nargs="*", default=None,
                    choices=["hydro", "landcover", "climate", "soil", "topo"])
    ap.add_argument("--force", action="store_true",
                    help="allow overwriting a prediction stored by a "
                         "different configuration")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the intended run names, config hashes and "
                         "cache states, then exit without loading the dataset "
                         "or training anything")
    # accepted for symmetry with run_gnn.py; this wrapper always enables saving
    ap.add_argument("--save-predictions", action="store_true",
                    help="(always on in this wrapper; accepted for symmetry)")
    ap.add_argument("--rebuild-predictions", action="store_true")

    args = ap.parse_args()
    args.save_predictions = True

    print("run_ladder: --save-predictions is forced on for every run")

    if args.dry_run:
        run_gnn = load_runner()
        masks = sorted(p.stem for p in Path("experiments/masks").glob("*.npz"))
        if args.only:
            masks = [m for m in masks if m.startswith(args.only)]
        datasets_seen: dict[str, dict] = {}
        for variant in args.variants:
            args.variant = variant
            name = run_gnn.model_name_for(args, variant)
            print(f"\n{variant}: model_name={name}")
            for mask in masks:
                # Use the SAME identity function as the real run: it includes the
                # dataset and mask content hashes, which the bare parameter dict
                # does not. A dry-run hash that omits them would differ from the
                # stored pin and report a phantom conflict.
                want_fields = run_gnn.expected_identity(args, name, mask)
                want = config_hash(want_fields)
                dpath = want_fields.get("dataset_path") or ""
                if dpath not in datasets_seen:
                    datasets_seen[dpath] = {
                        "sha": (want_fields.get("dataset_sha256") or "?")[:12],
                        "exists": want_fields.get("dataset_sha256") is not None,
                    }
                info = datasets_seen[dpath]
                state = cache_state(name, mask)
                parquet = prediction_path(name, mask)
                note = state.value
                if parquet.exists():
                    meta = read_meta(name, mask)
                    if meta is None:
                        note += " (no provenance sidecar: legacy file)"
                    else:
                        problems = identity_problems(meta, want_fields)
                        if problems:
                            note += " CONFLICT: stored identity differs"
                            for problem in problems:
                                note += f"\n{'':28}- {problem}"
                print(f"    {mask:22} {note}")
            print(f"  config_hash={want[:12]} seed={want_fields['seed']} "
                  f"arch={want_fields['architecture']} "
                  f"dataset={dpath or '(missing)'} "
                  f"dataset_sha={info['sha']}"
                  + ("" if info["exists"] else "  [dataset not found]"))
            print(f"    masks={len(masks)}")
        return

    report = load_runner().run(args)
    print()
    print(f"run ladder summary: {report.summary()}")
    if report.identity_mismatch:
        print("refused (different configuration already stored):")
        for name in report.identity_mismatch:
            print(f"  - {name}")
        print("pass --force to replace them deliberately")
        raise SystemExit(3)
    if report.missing_predictions:
        print("predictions missing (this wrapper always saves, so these "
              "indicate a failed write):")
        for name in report.missing_predictions:
            print(f"  - {name}")
        raise SystemExit(2)
    stored = list(PRED_DIR.glob("*.meta.json"))
    print(f"provenance sidecars now present: {len(stored)}")


if __name__ == "__main__":
    main()
