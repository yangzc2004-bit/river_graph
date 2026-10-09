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
    if len(sys.argv) > 2 and sys.argv[1:3] == ["--experiment", "doc-source-level-attention-v1"]:
        from run_doc_source_level_attention_v1 import main as source_level_main
        sys.argv = [sys.argv[0], *sys.argv[3:]]
        source_level_main()
        return
    if len(sys.argv) > 2 and sys.argv[1:3] == ["--experiment", "doc-ratio-source-attention-v1"]:
        from run_doc_ratio_source_attention_v1 import main as ratio_attention_main
        sys.argv = [sys.argv[0], *sys.argv[3:]]
        return ratio_attention_main()
    if len(sys.argv) > 2 and sys.argv[1:3] == ["--experiment", "doc-relative-source-attention-geographical-v1"]:
        from run_doc_relative_source_attention_geographical_v1 import (
            main as relative_geographical_attention_main,
        )
        sys.argv = [sys.argv[0], *sys.argv[3:]]
        return relative_geographical_attention_main()
    if len(sys.argv) > 2 and sys.argv[1:3] == ["--experiment", "doc-relative-source-attention-v1"]:
        from run_doc_relative_source_attention_v1 import main as relative_attention_main
        sys.argv = [sys.argv[0], *sys.argv[3:]]
        return relative_attention_main()
    if len(sys.argv) > 2 and sys.argv[1:3] == ["--experiment", "doc-current-source-attention-geographical-v1"]:
        from run_doc_current_source_attention_geographical_v1 import (
            main as geographical_attention_main,
        )
        sys.argv = [sys.argv[0], *sys.argv[3:]]
        return geographical_attention_main()
    if len(sys.argv) > 2 and sys.argv[1:3] == ["--experiment", "doc-current-source-attention-v1"]:
        from run_doc_current_source_attention_v1 import main as current_attention_main
        sys.argv = [sys.argv[0], *sys.argv[3:]]
        return current_attention_main()
    if len(sys.argv) > 2 and sys.argv[1:3] == ["--experiment", "doc-source-innovation-geographical-v1"]:
        from run_doc_source_innovation_geographical_v1 import (
            main as innovation_geographical_main,
        )
        sys.argv = [sys.argv[0], *sys.argv[3:]]
        return innovation_geographical_main()
    if len(sys.argv) > 2 and sys.argv[1:3] == ["--experiment", "doc-source-innovation-learning-v1"]:
        from run_doc_source_innovation_learning_v1 import (
            main as innovation_learning_main,
        )
        sys.argv = [sys.argv[0], *sys.argv[3:]]
        return innovation_learning_main()
    if len(sys.argv) > 2 and sys.argv[1:3] == ["--experiment", "doc-source-innovation-transfer-v1"]:
        from run_doc_source_innovation_transfer_v1 import main as source_innovation_main
        sys.argv = [sys.argv[0], *sys.argv[3:]]
        return source_innovation_main()
    if len(sys.argv) > 2 and sys.argv[1:3] == ["--experiment", "doc-full-encoder-v1"]:
        from run_doc_full_encoder_v1 import main as full_encoder_main
        sys.argv = [sys.argv[0], *sys.argv[3:]]
        return full_encoder_main()
    if len(sys.argv) > 2 and sys.argv[1:3] == ["--experiment", "doc-longer-history-v1"]:
        from run_doc_longer_history_v1 import main as longer_history_main
        sys.argv = [sys.argv[0], *sys.argv[3:]]
        return longer_history_main()
    if len(sys.argv) > 2 and sys.argv[1:3] == ["--experiment", "doc-joint-source-states-v1"]:
        from run_doc_joint_source_states_v1 import main as joint_source_states_main
        sys.argv = [sys.argv[0], *sys.argv[3:]]
        return joint_source_states_main()
    if len(sys.argv) > 2 and sys.argv[1:3] == ["--experiment", "doc-extended-optimization-v1"]:
        from run_doc_extended_optimization_v1 import main as extended_optimization_main
        sys.argv = [sys.argv[0], *sys.argv[3:]]
        return extended_optimization_main()
    if len(sys.argv) > 2 and sys.argv[1:3] == ["--experiment", "doc-leaf-median-residual-v1"]:
        from run_doc_leaf_median_residual_v1 import main as leaf_median_residual_main
        sys.argv = [sys.argv[0], *sys.argv[3:]]
        return leaf_median_residual_main()
    if len(sys.argv) > 2 and sys.argv[1:3] == ["--experiment", "doc-leaf-distribution-reference-v1"]:
        from run_doc_leaf_distribution_reference_v1 import main as leaf_reference_main
        sys.argv = [sys.argv[0], *sys.argv[3:]]
        return leaf_reference_main()
    if len(sys.argv) > 2 and sys.argv[1:3] == ["--experiment", "doc-native-reference-objective-v1"]:
        from run_doc_native_reference_objective_v1 import main as native_reference_main
        sys.argv = [sys.argv[0], *sys.argv[3:]]
        return native_reference_main()
    if len(sys.argv) > 2 and sys.argv[1:3] == ["--experiment", "doc-reference-trajectory-v1"]:
        from run_doc_reference_trajectory_v1 import main as reference_trajectory_main
        sys.argv = [sys.argv[0], *sys.argv[3:]]
        return reference_trajectory_main()
    if len(sys.argv) > 2 and sys.argv[1:3] == ["--experiment", "doc-source-auxiliary-states-v1"]:
        from run_doc_source_auxiliary_states_v1 import main as source_auxiliary_main
        sys.argv = [sys.argv[0], *sys.argv[3:]]
        return source_auxiliary_main()
    if len(sys.argv) > 2 and sys.argv[1:3] == ["--experiment", "doc-antecedent-residual-v1"]:
        from run_doc_antecedent_residual_v1 import main as antecedent_residual_main
        sys.argv = [sys.argv[0], *sys.argv[3:]]
        return antecedent_residual_main()
    if len(sys.argv) > 2 and sys.argv[1:3] == ["--experiment", "doc-antecedent-hydro-v1"]:
        from run_doc_antecedent_hydro_v1 import main as antecedent_hydro_main
        sys.argv = [sys.argv[0], *sys.argv[3:]]
        return antecedent_hydro_main()
    if len(sys.argv) > 2 and sys.argv[1:3] == ["--experiment", "doc-composition-encoder-v1"]:
        from run_doc_composition_encoder_v1 import main as composition_encoder_main
        sys.argv = [sys.argv[0], *sys.argv[3:]]
        return composition_encoder_main()
    if len(sys.argv) > 2 and sys.argv[1:3] == ["--experiment", "doc-ecological-composition-v1"]:
        from run_doc_ecological_composition_v1 import (
            main as ecological_composition_main,
        )
        sys.argv = [sys.argv[0], *sys.argv[3:]]
        return ecological_composition_main()
    if len(sys.argv) > 2 and sys.argv[1:3] == ["--experiment", "doc-nonlinear-geographical-replication-v1"]:
        from run_doc_nonlinear_geographical_replication_v1 import (
            main as nonlinear_geographical_main,
        )
        sys.argv = [sys.argv[0], *sys.argv[3:]]
        return nonlinear_geographical_main()
    if len(sys.argv) > 2 and sys.argv[1:3] == ["--experiment", "doc-nonlinear-native-residual-v1"]:
        from run_doc_nonlinear_native_residual_v1 import main as nonlinear_native_main
        sys.argv = [sys.argv[0], *sys.argv[3:]]
        return nonlinear_native_main()
    if (len(sys.argv) > 2 and sys.argv[1] == "--experiment"
            and sys.argv[2] in {"doc-regional-source-training-v1", "doc-regional-source-training-v2"}):
        from run_doc_regional_source_training_v1 import main as regional_source_main
        sys.argv = [sys.argv[0], *sys.argv[3:]]
        return regional_source_main()
    if (len(sys.argv) > 2 and sys.argv[1] == "--experiment"
            and sys.argv[2] in {"doc-log-concentration-residual-v1", "doc-log-concentration-residual-v2"}):
        from run_doc_log_concentration_residual_v1 import main as log_concentration_main
        sys.argv = [sys.argv[0], *sys.argv[3:]]
        return log_concentration_main()
    if len(sys.argv) > 2 and sys.argv[1:3] == ["--experiment", "doc-concentration-hydro-v1"]:
        from run_doc_concentration_hydro_v1 import main as concentration_hydro_main
        sys.argv = [sys.argv[0], *sys.argv[3:]]
        return concentration_hydro_main()
    if (len(sys.argv) > 2 and sys.argv[1] == "--experiment"
            and sys.argv[2] in {"doc-temporal-compatibility-v1", "doc-temporal-compatibility-v2"}):
        from run_doc_temporal_compatibility_v1 import (
            main as temporal_compatibility_main,
        )
        sys.argv = [sys.argv[0], *sys.argv[3:]]
        return temporal_compatibility_main()
    if len(sys.argv) > 2 and sys.argv[1:3] == ["--experiment", "doc-station-balanced-residual-v1"]:
        from run_doc_station_balanced_residual_v1 import main as station_balanced_main
        sys.argv = [sys.argv[0], *sys.argv[3:]]
        return station_balanced_main()
    if len(sys.argv) > 2 and sys.argv[1:3] == ["--experiment", "doc-portable-source-fit-v1"]:
        from run_doc_portable_source_fit_v1 import main as portable_source_main
        sys.argv = [sys.argv[0], *sys.argv[3:]]
        return portable_source_main()
    if len(sys.argv) > 2 and sys.argv[1:3] == ["--experiment", "doc-geographical-confirmation-v1"]:
        from run_doc_geographical_confirmation_v1 import main
        sys.argv = [sys.argv[0], *sys.argv[3:]]
        return main()
    if len(sys.argv) > 2 and sys.argv[1:3] == ["--experiment", "doc-unmonitored-residual-v1"]:
        from run_doc_unmonitored_residual_v1 import main as unmonitored_residual_main
        sys.argv = [sys.argv[0], *sys.argv[3:]]
        unmonitored_residual_main()
        return
    if len(sys.argv) > 2 and sys.argv[1:3] == ["--experiment", "doc-unmonitored-trees-v1"]:
        from run_doc_unmonitored_trees_v1 import main as unmonitored_trees_main
        sys.argv = [sys.argv[0], *sys.argv[3:]]
        unmonitored_trees_main()
        return
    if len(sys.argv) > 2 and sys.argv[1:3] == ["--experiment", "doc-source-retrieval-v3"]:
        from run_doc_source_retrieval_v3 import main as source_response_main
        sys.argv = [sys.argv[0], *sys.argv[3:]]
        source_response_main()
        return
    if len(sys.argv) > 2 and sys.argv[1:3] == ["--experiment", "doc-source-retrieval-v2"]:
        from run_doc_source_retrieval_v2 import main as source_contrast_main
        sys.argv = [sys.argv[0], *sys.argv[3:]]
        source_contrast_main()
        return
    if len(sys.argv) > 2 and sys.argv[1:3] == ["--experiment", "doc-source-retrieval-v1"]:
        from run_doc_source_retrieval_v1 import main as source_retrieval_main
        sys.argv = [sys.argv[0], *sys.argv[3:]]
        source_retrieval_main()
        return
    if len(sys.argv) > 2 and sys.argv[1:3] == ["--experiment", "doc-nested-confirmation-v1"]:
        from run_doc_nested_confirmation_v1 import main as nested_confirmation_main
        sys.argv = [sys.argv[0], *sys.argv[3:]]
        nested_confirmation_main()
        return
    if len(sys.argv) > 2 and sys.argv[1:3] == ["--experiment", "doc-nested-chemistry-v1"]:
        from run_doc_nested_chemistry_v1 import main as nested_chemistry_main
        sys.argv = [sys.argv[0], *sys.argv[3:]]
        nested_chemistry_main()
        return
    if len(sys.argv) > 2 and sys.argv[1:3] == ["--experiment", "doc-chemistry-confirmation-v1"]:
        from run_doc_chemistry_confirmation_v1 import (
            main as chemistry_confirmation_main,
        )
        sys.argv = [sys.argv[0], *sys.argv[3:]]
        chemistry_confirmation_main()
        return
    if len(sys.argv) > 2 and sys.argv[1:3] == ["--experiment", "doc-chemical-kernel-v1"]:
        from run_doc_chemical_kernel_v1 import main as chemical_kernel_main
        sys.argv = [sys.argv[0], *sys.argv[3:]]
        chemical_kernel_main()
        return
    if len(sys.argv) > 2 and sys.argv[1:3] == ["--experiment", "doc-chemistry-support-v1"]:
        from run_doc_chemistry_support_v1 import main as chemistry_support_main
        sys.argv = [sys.argv[0], *sys.argv[3:]]
        chemistry_support_main()
        return
    if len(sys.argv) > 2 and sys.argv[1:3] == ["--experiment", "doc-chemistry-decoder-v1"]:
        from run_doc_chemistry_decoder_v1 import main as decoder_main
        sys.argv = [sys.argv[0], *sys.argv[3:]]
        decoder_main()
        return
    if len(sys.argv) > 2 and sys.argv[1:3] == ["--experiment", "doc-auxiliary-chemistry-v1"]:
        from run_doc_auxiliary_chemistry_v1 import main as auxiliary_main
        sys.argv = [sys.argv[0], *sys.argv[3:]]
        auxiliary_main()
        return
    if len(sys.argv) > 2 and sys.argv[1:3] == ["--experiment", "doc-distribution-head-v1"]:
        from run_doc_distribution_head_v1 import main as distribution_main
        sys.argv = [sys.argv[0], *sys.argv[3:]]
        distribution_main()
        return
    if len(sys.argv) > 2 and sys.argv[1:3] == ["--experiment", "doc-recurrent-clock-v1"]:
        from run_doc_recurrent_clock_v1 import main as clock_main
        sys.argv = [sys.argv[0], *sys.argv[3:]]
        clock_main()
        return
    if len(sys.argv) > 2 and sys.argv[1:3] == ["--experiment", "doc-daily-hydro-readout-v1"]:
        from run_doc_daily_hydro_readout_v1 import main as readout_main
        sys.argv = [sys.argv[0], *sys.argv[3:]]
        readout_main()
        return
    if len(sys.argv) > 2 and sys.argv[1:3] == ["--experiment", "doc-daily-hydro-support-basis-v1"]:
        from run_doc_daily_hydro_support_basis_v1 import main as run_support_basis
        sys.argv = [sys.argv[0], *sys.argv[3:]]
        run_support_basis()
        return
    if len(sys.argv) > 2 and sys.argv[1:3] == ["--experiment", "doc-daily-hydro-memory-v1"]:
        from run_doc_daily_hydro_memory_v1 import main as daily_memory_main
        sys.argv = [sys.argv[0], *sys.argv[3:]]
        daily_memory_main()
        return
    if len(sys.argv) > 2 and sys.argv[1:3] == ["--experiment", "doc-daily-hydro-fallback-v1"]:
        from run_doc_daily_hydro_fallback_v1 import main as daily_fallback_main
        sys.argv = [sys.argv[0], *sys.argv[3:]]
        daily_fallback_main()
        return
    if len(sys.argv) > 2 and sys.argv[1:3] == ["--experiment", "doc-daily-hydro-residual-v1"]:
        from run_doc_daily_hydro_residual_v1 import main as daily_hydro_main
        sys.argv = [sys.argv[0], *sys.argv[3:]]
        daily_hydro_main()
        return
    if len(sys.argv) > 2 and sys.argv[1:3] == ["--experiment", "doc-selective-budget-v1"]:
        from run_doc_selective_budget_v1 import main as selective_budget_main
        sys.argv = [sys.argv[0], *sys.argv[3:]]
        selective_budget_main()
        return
    if len(sys.argv) > 2 and sys.argv[1:3] == ["--experiment", "doc-selective-residual-v1"]:
        from run_doc_selective_residual_v1 import main as selective_residual_main
        sys.argv = [sys.argv[0], *sys.argv[3:]]
        selective_residual_main()
        return
    if len(sys.argv) > 2 and sys.argv[1:3] == ["--experiment", "doc-encoder-budget-v1"]:
        from run_doc_encoder_budget_v1 import main as encoder_budget_main
        sys.argv = [sys.argv[0], *sys.argv[3:]]
        encoder_budget_main()
        return
    if len(sys.argv) > 2 and sys.argv[1:3] == ["--experiment", "doc-encoder-residual-v1"]:
        from run_doc_encoder_residual_v1 import main as encoder_residual_main
        sys.argv = [sys.argv[0], *sys.argv[3:]]
        encoder_residual_main()
        return
    if len(sys.argv) > 2 and sys.argv[1:3] == ["--experiment", "doc-regime-residual-v1"]:
        from run_doc_regime_residual_v1 import main as regime_residual_main
        sys.argv = [sys.argv[0], *sys.argv[3:]]
        regime_residual_main()
        return
    if len(sys.argv) > 2 and sys.argv[1:3] == ["--experiment", "doc-ecological-transfer-v2"]:
        from run_doc_ecological_transfer_v2 import main as support_transfer_main
        sys.argv = [sys.argv[0], *sys.argv[3:]]
        support_transfer_main()
        return
    if len(sys.argv) > 2 and sys.argv[1:3] == ["--experiment", "doc-ecological-transfer-v1"]:
        from run_doc_ecological_transfer_v1 import main as ecological_transfer_main
        sys.argv = [sys.argv[0], *sys.argv[3:]]
        ecological_transfer_main()
        return
    if len(sys.argv) > 2 and sys.argv[1:3] == ["--experiment", "doc-flow-interaction-v1"]:
        from run_doc_flow_interaction_v1 import main as flow_interaction_main
        sys.argv = [sys.argv[0], *sys.argv[3:]]
        flow_interaction_main()
        return
    if len(sys.argv) > 2 and sys.argv[1:3] == ["--experiment", "doc-flow-residual-v1"]:
        from run_doc_flow_residual_v1 import main as flow_residual_main
        sys.argv = [sys.argv[0], *sys.argv[3:]]
        flow_residual_main()
        return
    if len(sys.argv) > 2 and sys.argv[1:3] == ["--experiment", "doc-tail-residual-v1"]:
        from run_doc_tail_residual_v1 import main as tail_residual_main
        sys.argv = [sys.argv[0], *sys.argv[3:]]
        tail_residual_main()
        return
    if len(sys.argv) > 2 and sys.argv[1:3] == ["--experiment", "unified-doc-spatial-v4"]:
        from run_unified_doc_spatial_v4 import main as unified_v4_main
        sys.argv = [sys.argv[0], *sys.argv[3:]]
        unified_v4_main()
        return
    if len(sys.argv) > 2 and sys.argv[1:3] == ["--experiment", "unified-doc-spatial-v3"]:
        from run_unified_doc_spatial_v3 import main as unified_v3_main
        sys.argv = [sys.argv[0], *sys.argv[3:]]
        unified_v3_main()
        return
    if len(sys.argv) > 2 and sys.argv[1:3] == ["--experiment", "unified-doc-spatial-v2"]:
        from run_unified_doc_spatial_v2 import main as unified_v2_main
        sys.argv = [sys.argv[0], *sys.argv[3:]]
        unified_v2_main()
        return
    # New experimental family, with its own versioned products and verifier.
    # Legacy snapshot arguments and behavior remain unchanged.
    if len(sys.argv) > 2 and sys.argv[1:3] == ["--experiment", "unified-doc-spatial"]:
        from run_unified_doc_spatial import main as unified_main
        sys.argv = [sys.argv[0], *sys.argv[3:]]
        unified_main()
        return
    if len(sys.argv) > 2 and sys.argv[1:3] == ["--experiment", "kgml"]:
        from run_kgml_local_transport import main as kgml_main
        sys.argv = [sys.argv[0], *sys.argv[3:]]
        kgml_main()
        return
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
