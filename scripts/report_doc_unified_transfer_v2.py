"""Evidence-bound architecture decision and primary-checkpoint reconstruction."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from reconstruct_doc_unified_v2 import reconstruct
from verify_doc_unified_transfer_v2 import paired_interval

from river_graph.experiments.provenance import sha256_file

ROOT = Path("experiments/phase4_transfer/doc_unified_transfer_v2")
REPORT = Path("docs/doc_unified_transfer_results_20261012.md")
LABELS = {"direction_heads": "Real upstream/downstream/global heads", "generic_dense": "Generic dense spatial attention",
          "self_only": "Self-only spatial path", "permuted_river": "Permuted river topology"}


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    audit = json.loads((ROOT/"analysis/independent_audit.json").read_text())
    if not audit["passed"]:
        raise ValueError("audit not passed")
    for name, expected in audit["files"].items():
        if sha256_file(ROOT/name) != expected:
            raise ValueError("audited evidence changed")
    summary = pd.read_csv(ROOT/"analysis/summary.csv")
    detail = pd.read_csv(ROOT/"analysis/by_fit.csv")
    choice = json.loads((ROOT/"selected_model.json").read_text())
    diagnostic = json.loads((ROOT/"analysis/diagnostics.json").read_text())
    schema = json.loads((ROOT/"schema_inspection.json").read_text())
    with np.load(ROOT/"external/inputs.npz") as receiving:
        grid_availability = receiving["x_mask"].mean(axis=(0, 1)).tolist()
    selected = choice["selected_arm"]
    primary = detail[detail.id == choice["primary_checkpoint_id"]]
    release = ROOT/"release"
    release.mkdir(exist_ok=True)
    field = reconstruct(ROOT, ROOT/"external/inputs.npz", release/"primary_external_field_v2.npz")
    with np.load(release/"primary_external_field_v2.npz") as current, np.load(ROOT/"fits"/choice["primary_checkpoint_id"]/"external_field.npz") as previous:
        replay = float(np.abs(current["y_pred"]-previous["y_pred"]).max())
    if replay > 1e-6:
        raise ValueError("release differs from audited primary field")
    intervals = [paired_interval(ROOT, domain, selected, reference, metric)
        for domain in ("source", "external") for reference in LABELS if reference != selected
        for metric in ("mae", "log1p_mae")]
    (ROOT/"analysis/selected_model_intervals.json").write_text(json.dumps(intervals, indent=2)+"\n")
    external = summary[(summary.domain == "external_river_K0") & (summary.arm == selected)].iloc[0]
    native_change = float(external.mae_improvement_vs_self_percent)
    log_change = float(external.log1p_mae_improvement_vs_self_percent)
    selection_name = ("environment MLP + GRU + upstream/downstream/global graph Transformer" if selected == "direction_heads" else
                      "environment MLP + GRU + generic dense spatial Transformer" if selected == "generic_dense" else
                      "environment MLP + GRU + retained feed-forward/self path, without interstation messaging")
    transfer_supported = float(external.r2) > 0 and native_change > 0 and log_change > 0
    decision = {"source_selected_architecture": selected, "architecture": selection_name, "primary_checkpoint": choice["primary_checkpoint_id"],
        "external_test_used_for_selection": False, "external_replication_r2": float(external.r2),
        "external_mae_improvement_vs_self_percent": native_change, "external_log_mae_improvement_vs_self_percent": log_change,
        "external_descriptive_support_for_selected_spatial_arm": bool(transfer_supported),
        "global_generalization_established": False, "receiving_network_count": 1, "primary_field": str(release/"primary_external_field_v2.npz"),
        "primary_release_replay_max_abs": replay, "source_selection_sha256": sha256_file(ROOT/"selected_model.json"),
        "receiving_full_grid_hydro_availability": {"temperature": grid_availability[0], "discharge": grid_availability[1]},
        "scope": "architecture fixed by source validation, evidence limited to three random source splits and one previously inspected external receiving network; no test-driven architecture reselection"}
    (ROOT/"research_decision.json").write_text(json.dumps(decision, indent=2)+"\n")
    lines = ["# Unified DOC model: spatial contribution and external replication", "", "Completed 12 October 2026. Share with caveats: source-spatial evidence is supported, but this one external replication fails to establish a universal transfer model.", "",
        f"Source-validation-selected architecture: **{selection_name}**. All arms belong to the same jointly trained model lineage; no local-anchor/forest correction branch was substituted.", "",
        "## Matched results", "", "Arithmetic mean of nine fit-level metrics (three existing whole-station splits x three optimization seeds); predictions are not ensembled. Native MAE/RMSE units are mg/L. Log-MAE uses log1p DOC. Bold marks the numerically best metric within each domain. All query predictions have 100% coverage.", ""]
    for domain in ("source_hidden_station", "external_river_K0"):
        part = summary[summary.domain == domain]
        lines += [f"### {domain}", "", "| Spatial setting | MAE | log-MAE | RMSE | R2 | Station-macro MAE |", "|---|---:|---:|---:|---:|---:|"]
        for arm, label in LABELS.items():
            row = part[part.arm == arm].iloc[0]
            numbers = []
            for key in ("mae", "log1p_mae", "rmse", "r2", "station_macro_mae"):
                best = part[key].max() if key == "r2" else part[key].min()
                value = f"{row[key]:.4f}"
                numbers.append(f"**{value}**" if row[key] == best else value)
            lines.append("| "+label+" | "+" | ".join(numbers)+" |")
        lines.append("")
    lines += ["## Architecture choice and spatial interpretation", "",
        "Choice was sealed before either source-test evaluation or receiving-field prediction. Both primary validation ratios must improve and the average paired joint ratio must improve by at least1%; the lowest eligible real/generic spatial score wins, otherwise self-only remains. The permutation control is never eligible. This rule is new to this explicitly versioned spatial-inclusion experiment; prior paper endpoints were not changed.", "",
        "Validation ratios versus self-only:", "", "| Setting | Native ratio | Log ratio | Joint ratio |", "|---|---:|---:|---:|"]
    for arm, values in choice["validation_ratios"].items():
        lines.append(f"| {LABELS[arm]} | {values['native_ratio']:.6f} | {values['log_ratio']:.6f} | {values['joint']:.6f} |")
    lines += ["", "The self-only control keeps static river/environment covariates and every projection/feed-forward/residual self path. Its comparison isolates interstation messaging, not every conceivable spatial input. The generic arm has all-node attention and self/other biases; river-direction information is not encoded in attention masks. The permutation retains exact category counts and degree distributions but reassigns them across identities, with some true connected pairs possibly remaining; it is not a strict non-connected control.", "",
        f"Schema inspection confirms zero source/receiving COMID overlap. The source headwater flag is inherited from its original station-graph construction: {schema['source_flag_current_graph_mismatch']} of357 flags differs from zero in-degree in the later repaired source graph. Source values were retained unchanged for model lineage and matched comparisons. Receiving flags use the same original construction formula on their own cohort. Water-temperature units are degC and discharge units are ft3/s.", "",
        "The directional recipe wins the frozen source-validation ranking and has the lowest source-test means, but its paired native/log error intervals versus generic attention include zero. It is a source-domain candidate, not proven uniquely optimal among spatial architectures.", "",
        "## Receiving-river interpretation", "",
        f"External HUC02040104 ([USGS basin identification](https://www.usgs.gov/data/delaware-river-basin-catchments-based-strahler-stream-order-1-national-hydrography-dataset-nhd)) contains {diagnostic['receiving_stations']} stations, {diagnostic['receiving_graph_edges']} directed station edges and {diagnostic['receiving_doc_cells']} observed monthly DOC queries. Station overlap with source is zero. Temperature/discharge availability at receiving DOC queries is {100*diagnostic['receiving_hydro_at_doc']['temperature']:.2f}%/{100*diagnostic['receiving_hydro_at_doc']['discharge']:.2f}%.", "",
        f"Across the entire receiving station-month grid, temperature/discharge availability is only {100*grid_availability[0]:.2f}%/{100*grid_availability[1]:.2f}%. Query-cell covariates are therefore substantially more available than naturally unobserved grid-cell covariates. Output coverage does not establish accuracy in those information-poor cells; the evaluation does not establish reconstruction from universally available meteorological forcing alone.", "",
        f"The source-selected architecture changes receiving native/log MAE by {native_change:+.2f}%/{log_change:+.2f}% improvement versus self-only, with R2={float(external.r2):.4f}. This is descriptive evidence; the architecture is not reselected using receiving DOC. "+("Both endpoints improve and R2 is positive in this replication case." if transfer_supported else "The receiving results do not provide joint positive evidence for the selected spatial arm; it must not be advertised as an established universal transfer model."), "",
        "The receiving graph contains only receiving nodes; all preprocessing statistics and neural weights are source fitted, and no receiving DOC support, calibration or fine-tuning is used. All36 complete receiving fields were sealed before its DOC archive was opened. The same receiving case was previously inspected by other architecture releases. One network does not establish global river generalization, and nine source fits do not constitute nine independent external rivers.", "",
        "## Prechosen single checkpoint", "",
        f"Primary checkpoint `{choice['primary_checkpoint_id']}` was prechosen as split142 seed42 of the source-selected architecture. Its values differ from the nine-fit means:", "",
        "| Domain | MAE | log-MAE | RMSE | R2 |", "|---|---:|---:|---:|---:|"]
    for row in primary.itertuples():
        lines.append(f"| {row.domain} | {row.mae:.4f} | {row.log1p_mae:.4f} | {row.rmse:.4f} | {row.r2:.4f} |")
    lines += ["", f"Primary receiving reconstruction is a complete {field['shape'][0]} x {field['shape'][1]} station-month grid saved in `release/primary_external_field_v2.npz` with a provenance sidecar. Scoring parquets contain observed queries only. Unobserved cells have no ground-truth validation. Monthly station reconstruction is not validated continuous-time or arbitrary-location reconstruction.", "",
        "```bash", "uv run python scripts/reconstruct_doc_unified_v2.py \\",
        "  --inputs experiments/phase4_transfer/doc_unified_transfer_v2/external/inputs.npz \\",
        "  --output path/to/new_reconstruction.npz", "```", "",
        "## Verification and uncertainty", "",
        f"Independent audit verified {audit['fit_count']} checkpoints, {audit['receipts_verified']} fit/field/test receipts and {audit['source_label_role_checks']} source query roles. Maximum all-field/query replay error={audit['prediction_replay_max_abs']:.3g}; maximum independently recomputed metric difference={audit['metric_recompute_max_abs']:.3g}. The primary public inference entry point also reproduces its audited field exactly.", "",
        "Paired station-block bootstrap intervals (3000 draws) are exploratory within these fixed datasets; station correlation, overlapping source partitions and unadjusted multiple comparisons limit inference. Delta is selected model error minus comparator error:", "",
        "| Domain | Comparator | Endpoint | Delta | 95% interval |", "|---|---|---|---:|---|"]
    for interval in intervals:
        lo, hi = interval["ci95"]
        lines.append(f"| {interval['domain']} | {LABELS[interval['reference']]} | {interval['metric']} | {interval['delta']:+.5f} | [{lo:+.5f}, {hi:+.5f}] |")
    lines += ["", "Version1 failed at its first shared-scoring integration because its prediction table omitted high_doc; no completed validation selection or target score existed. Version2 fixes that schema only, retrains all36 fits, and preserves the failed frozen attempt. No tail-DOC endpoint is claimed.", "",
        "Machine-readable evidence: protocol.json, manifest.json, selected_model.json, research_decision.json, analysis/summary.csv, analysis/by_fit.csv, analysis/independent_audit.json, analysis/paired_intervals.json, analysis/selected_model_intervals.json, and per-fit prediction/provenance receipts.", ""]
    REPORT.write_text("\n".join(lines))
    receipt = {"report_sha256": sha256_file(REPORT), "decision_sha256": sha256_file(ROOT/"research_decision.json"),
        "report_generator_sha256": sha256_file(__file__), "inference_code_sha256": sha256_file("scripts/reconstruct_doc_unified_v2.py"),
        "audit_sha256": sha256_file(ROOT/"analysis/independent_audit.json"),
        "selected_intervals_sha256": sha256_file(ROOT/"analysis/selected_model_intervals.json"),
        "schema_inspection_sha256": sha256_file(ROOT/"schema_inspection.json"),
        "release_metadata_sha256": sha256_file(release/"primary_external_field_v2.meta.json")}
    (ROOT/"analysis/report_complete.json").write_text(json.dumps(receipt, indent=2)+"\n")
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
