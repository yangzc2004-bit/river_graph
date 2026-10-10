"""Independent diagnostic arithmetic, figures and final source receipts."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from river_graph.experiments.provenance import sha256_file

ROOT = Path("experiments/phase4_transfer/doc_fusion_component_comparison_v1")


def verify_diagnostic(directory, protocol, truth):
    record = json.loads((directory / "complete.json").read_text())
    for name, expected in record["files"].items():
        if sha256_file(directory / name) != expected:
            raise ValueError("changed input diagnostic artifact")
    design = json.loads((directory / "protocol.json").read_text())
    for name, expected in design["runtime_snapshot"].items():
        if sha256_file(name) != expected or sha256_file(directory / "code_snapshot" / name) != expected:
            raise ValueError("changed diagnostic runtime or source copy")
    meta = json.loads((directory / "predictions.meta.json").read_text())
    sources = meta.get("source_hashes", meta.get("checkpoint_and_selection_hashes"))
    for name, expected in sources.items():
        if sha256_file(name) != expected:
            raise ValueError("changed diagnostic checkpoint or stage selection")
    frame = pd.read_parquet(directory / "predictions.parquet")
    np.testing.assert_array_equal(frame.y_true, truth[frame.cell.to_numpy()])
    if not np.isfinite(frame[["y_true", "y_pred"]].to_numpy()).all():
        raise ValueError("nonfinite diagnostic predictions")
    table = pd.read_csv(directory / "summary.csv").set_index(["input_scenario", "model_name"])
    maximum, populations = 0., set()
    for (region, seed, scenario, name), part in frame.groupby(
            ["target_huc4", "seed", "input_scenario", "model_name"]):
        path = Path("experiments/phase4_transfer/doc_unmonitored_tasks_v1/geographical_masks") / f"huc4_{region}.npz"
        with np.load(path) as split:
            cells = split["test"]
        np.testing.assert_array_equal(np.sort(part.cell), np.sort(cells))
        if part.cell.duplicated().any() or seed not in protocol["seeds"]:
            raise ValueError("duplicate or invalid diagnostic queries")
        populations.add((scenario, name))
    for (scenario, name), part in frame.groupby(["input_scenario", "model_name"]):
        residual = part.y_pred.to_numpy(dtype=np.float64) - truth[part.cell.to_numpy()]
        independent_mae = float(np.abs(residual).sum() / len(residual))
        difference = abs(independent_mae - table.loc[(scenario, name), "mae"])
        maximum = max(maximum, difference)
        if difference > 1e-10 or len(part) != 7262 * 3:
            raise ValueError("diagnostic source-grid arithmetic or coverage disagrees")
    if len(populations) != len(design["scenarios"]) * len(design["models"]):
        raise ValueError("missing diagnostic cases")
    return {"maximum_arithmetic_difference": maximum, "queries_per_case": 7262,
            "cases": len(populations), "predictions_sha256": sha256_file(directory / "predictions.parquet")}


def plot_time(out, table):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scenarios = ["available_covariates", "query_current_temperature_hidden", "query_current_hydrology_hidden"]
    labels = {"time__current": "Current MLP", "time__lag_mlp": "History MLP", "time__gru": "GRU",
              "time__lstm": "LSTM", "time__transformer": "Time Transformer"}
    figure, ax = plt.subplots(figsize=(8.4, 5.0), layout="constrained")
    pivot = table.pivot(index="input_scenario", columns="model_name", values="mae")
    for name, label in labels.items():
        ax.plot(range(3), pivot.loc[scenarios, name], marker="o", linewidth=1.8, label=label)
    ax.set_xticks(range(3), ["Available inputs", "Query temperature\nhidden", "Query temperature + flow\nhidden"])
    ax.set_ylabel("DOC MAE (mg/L; lower is better)")
    ax.set_title("Time recipes under current-input missingness\n5 regions × 3 seeds; fixed checkpoints, historical non-query inputs retained", fontsize=11)
    ax.legend(fontsize=9, ncol=2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=.2)
    figure.savefig(out / "time_current_missing.png", dpi=180)
    figure.savefig(out / "time_current_missing.pdf")
    plt.close(figure)


def main():
    out = ROOT / "analysis"
    protocol = json.loads((ROOT / "protocol.json").read_text())
    audit = json.loads((out / "audit.json").read_text())
    arithmetic = json.loads((out / "independent_metrics_audit.json").read_text())
    if not audit["checkpoint_replay"] or audit["fits"] != 165 or audit["maximum_replay_difference"] > 1e-6:
        raise ValueError("all checkpoint replays required")
    if arithmetic["maximum_difference"] > 1e-10:
        raise ValueError("independent primary arithmetic required")
    truth = np.asarray(torch.load(protocol["dataset"], weights_only=False, map_location="cpu")["y"],
                       dtype=np.float64).ravel()
    diagnostics = {name: verify_diagnostic(ROOT / name, protocol, truth)
                   for name in ("input_robustness", "time_current_missing_v1")}
    (out / "diagnostic_arithmetic_audit.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
    pipeline = pd.read_csv(ROOT / "input_robustness/summary.csv")
    temporal = pd.read_csv(ROOT / "time_current_missing_v1/summary.csv")
    plot_time(out, temporal)
    marker = "## Additional frozen input diagnostics"
    report = out / "research_decision.md"
    base = report.read_text().split(marker)[0].rstrip()
    lines = ["", marker, "",
        ("These fixed-checkpoint diagnostics were declared before pooled test scores. They do not change "
        "the original primary endpoint, any model choices, or the 165-fit training protocol."), "",
        "Pipeline stress test: erase selected target channels for all months; source context remains:", "",
        pipeline.pivot(index="input_scenario", columns="model_name", values="mae").to_markdown(), "",
        ("Time-recipe stress test: erase query-month current inputs, retaining non-query historical measurements. "
        "Erased months also disappear from later windows, and observation ages are recomputed:"), "",
        temporal.pivot(index="input_scenario", columns="model_name", values="mae").to_markdown(), "",
        ("The latter compares time candidates with the same validation-selected environmental module and "
        "concatenation per fold. Operator and capacity differ; it does not causally isolate memory alone. "
        "Both interventions are stress tests of observed DOC queries, not direct measurements of naturally missing DOC cells."),
        "", "## Equal-shape serial inference", ""]
    timing_path = out / "serial_inference_timing.json"
    if timing_path.exists():
        timing = json.loads(timing_path.read_text())
        lines.extend([(f"One CPU thread; {timing['station_nodes']} station nodes × {timing['months']} months; "
                      "three passes after a warmup. One representative region's parent choices; "
                      "feature assembly, graph building and I/O excluded. This is not an all-reach field."), "",
                      pd.DataFrame(timing["records"])[["model_name", "parameters", "median_seconds"]].to_markdown(index=False), ""])
    lines.extend(["## Final verification", "",
        ("Disposition: Share with caveats within the internal development scope. All 165 checkpoints replay "
        "validation and test predictions; primary and diagnostic MAE are independently recalculated "
        "against the original DOC grid. Parent paper protocols and endpoints remain unchanged."), "",
        "Reproduction:", "", "```bash",
        "uv run python scripts/run_ladder.py --experiment doc-fusion-component-comparison-v1",
        "uv run python scripts/analyze_doc_fusion_component_comparison_v1.py --replay",
        "uv run python scripts/verify_doc_fusion_component_comparison_v1.py",
        "uv run python scripts/audit_doc_fusion_input_availability_v1.py",
        "uv run python scripts/evaluate_doc_fusion_input_robustness_v1.py",
        "uv run python scripts/evaluate_doc_time_current_missing_v1.py",
        "uv run python scripts/benchmark_doc_fusion_component_comparison_v1.py",
        "uv run python scripts/finalize_doc_fusion_component_comparison_v1.py", "```"])
    report.write_text(base + "\n" + "\n".join(lines) + "\n")
    source_record = {"script_sha256": sha256_file(__file__), "protocol_sha256": sha256_file(ROOT / "protocol.json"),
        "manifest_sha256": sha256_file(ROOT / "manifest.json"), "data_sha256": protocol["dataset_hash"],
        "regional_completions": audit["source_completions"],
        "diagnostic_completions": {name: sha256_file(ROOT / name / "complete.json") for name in diagnostics},
        "analysis_outputs": {path.name: sha256_file(path) for path in out.glob("*")
                             if path.is_file() and path.name != "sources.json"}}
    memo = Path("docs/doc_fusion_component_results_20261011.md")
    if memo.exists():
        source_record["decision_memo"] = {"path": str(memo), "sha256": sha256_file(memo)}
    (out / "sources.json").write_text(json.dumps(source_record, indent=2) + "\n")
    print(json.dumps({"fits_replayed": 165, "diagnostic_arithmetic": diagnostics}), flush=True)


if __name__ == "__main__":
    main()
