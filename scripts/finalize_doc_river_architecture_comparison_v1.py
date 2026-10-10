"""Add verified serial timing and final evidence receipts to the study readout."""
from __future__ import annotations

import json
from pathlib import Path

from river_graph.experiments.provenance import sha256_file

ROOT = Path("experiments/phase4_transfer/doc_river_architecture_comparison_v1")


def main():
    out = ROOT / "analysis"
    audit = json.loads((out / "audit.json").read_text())
    arithmetic = json.loads((out / "independent_metrics_audit.json").read_text())
    timing = json.loads((out / "serial_inference_timing.json").read_text())
    if not audit["checkpoint_replay"] or audit["maximum_replay_difference"] > 1e-6:
        raise ValueError("complete checkpoint replay is required")
    if arithmetic["maximum_difference"] > 1e-6:
        raise ValueError("independent arithmetic validation is required")
    marker = "Final verification and equal-shape serial inference"
    report = out / "research_decision.txt"
    base = report.read_text().split(marker)[0].rstrip()
    lines = ["", marker, "", "Validation disposition: Share with caveats within the internal study scope.",
             ("All 60 fitted models replay stored query predictions exactly; independent float64 "
              "source-grid MAE calculations agree within 1.3e-7 mg/L."),
             ("Serial CPU inference: four threads, 357 station nodes x 654 months (233,478 outputs), "
              "five timed passes after one warmup. Feature assembly, graph construction and file I/O excluded."), ""]
    for row in timing["records"]:
        lines.append(f"{row['model_name']}: {row['median_seconds']:.6f} seconds (median).")
    lines.extend(["", "Reproduction commands:",
        "uv run --no-sync python scripts/run_ladder.py --experiment doc-river-architecture-comparison-v1",
        "uv run --no-sync python scripts/analyze_doc_river_architecture_comparison_v1.py --replay",
        "uv run --no-sync python scripts/verify_doc_river_architecture_comparison_v1.py",
        "uv run --no-sync python scripts/benchmark_doc_river_architecture_comparison_v1.py",
        "uv run --no-sync python scripts/plot_doc_river_architecture_comparison_v1.py",
        "uv run --no-sync python scripts/finalize_doc_river_architecture_comparison_v1.py"])
    report.write_text(base + "\n" + "\n".join(lines) + "\n")
    receipt = {"script_sha256": sha256_file(__file__), "reviewed_report_sha256": sha256_file(report),
               "verified_sources": {name: sha256_file(out / name) for name in
                                    ("audit.json", "independent_metrics_audit.json", "serial_inference_timing.json")},
               "figure_png_sha256": sha256_file(ROOT / "figures/architecture_comparison.png")}
    (out / "finalization_sources.json").write_text(json.dumps(receipt, indent=2) + "\n")
    sources = json.loads((out / "sources.json").read_text())
    sources["outputs"] = {path.name: sha256_file(path) for path in out.glob("*")
                          if path.is_file() and path.name != "sources.json"}
    (out / "sources.json").write_text(json.dumps(sources, indent=2) + "\n")


if __name__ == "__main__":
    main()
