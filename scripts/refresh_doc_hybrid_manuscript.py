"""Refresh verified tables and figure assets in the DOC hybrid manuscript.

Run the two analysis scripts first if prediction evidence has changed. This
script edits marked table blocks only, preserving the scientific narrative.
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TEX = ROOT / "docs/paper/latex/spatial_transfer_draft_v1.tex"
HYBRID = ROOT / "experiments/phase4_transfer/kgml_local_transport_v1/manuscript_evidence_v1"
SPATIAL = ROOT / "experiments/phase4_transfer/spatial_adaptation/manuscript_evidence_v2"
SCENARIOS = [
    ("e1_r20_seed42", "Random gaps"),
    ("e2a_strict", "Unobserved periods"),
    ("e2b_partial", "Observation-assisted periods"),
    ("e3_spatial_seed42", "Unmonitored stations"),
]


def block(source: str, name: str, rows: list[str]) -> str:
    pattern = rf"(% BEGIN GENERATED {name}\n).*?(% END GENERATED {name})"
    source, count = re.subn(pattern, lambda m: m[1] + "\n".join(rows) + "\n" + m[2],
                            source, flags=re.DOTALL)
    if count != 1:
        raise ValueError(f"Expected one {name} block")
    return source


def main() -> None:
    source = TEX.read_text()
    if "DOC HYBRID MANUSCRIPT" not in source:
        raise ValueError("Expected the revised DOC hybrid manuscript")
    data = pd.read_csv(HYBRID / "family_metrics.csv").set_index(["mask", "model"])
    support = pd.read_csv(SPATIAL / "main_results.csv").sort_values("k")
    scenarios = [
        r"\begin{tabular}{>{\raggedright\arraybackslash}p{4.3cm}rr}\toprule",
        r"Scenario & Query cells & Stations\\\midrule",
    ]
    main_rows = [
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{>{\raggedright\arraybackslash}p{3.5cm}rrrrr}\toprule",
        (r"Scenario & \shortstack{RF\\MAE} & \shortstack{ET\\MAE} & "
        r"\shortstack{Hybrid\\MAE} & \shortstack{Hybrid\\RMSE} & "
        r"\shortstack{Hybrid\\$R^2$}\\\midrule"),
    ]
    tail = [
        r"\begin{tabular}{>{\raggedright\arraybackslash}p{3.8cm}rrrr}\toprule",
        r"Scenario & Threshold & Cells & ET MAE & Hybrid MAE\\\midrule",
    ]
    for mask, label in SCENARIOS:
        h, et, rf = (data.loc[(mask, model)] for model in
                     ("hybrid", "et_context", "rf_context"))
        scenarios.append(f"{label} & {int(h.n_unique_cells):,} & {int(h.n_stations)}" + r"\\")
        main_rows.append(
            f"{label} & {rf.mae:.3f} & {et.mae:.3f} & {h.mae:.3f} & "
            f"{h.rmse:.3f} & {h.r2:.3f}" + r"\\")
        mark = "*" if h.q90_n_unique_cells < 20 else ""
        tail.append(
            f"{label} & {h.q90_threshold_train:.2f} & "
            f"{int(h.q90_n_unique_cells)}{mark} & {et.q90_mae:.3f} & "
            f"{h.q90_mae:.3f}" + r"\\")
    for rows in (scenarios, main_rows, tail):
        rows.append(r"\bottomrule\end{tabular}")

    components = [
        r"\begin{tabular}{>{\raggedright\arraybackslash}p{3.7cm}rrrr}\toprule",
        r"Scenario & Local base & Context & Residual expert & Hybrid\\\midrule",
    ]
    for mask, label in SCENARIOS[1:3]:
        values = [data.loc[(mask, model), "mae"] for model in
                  ("local_base", "et_context", "temporal_residual", "hybrid")]
        components.append(label + " & " + " & ".join(f"{v:.3f}" for v in values) + r"\\")
    components.append(r"\bottomrule\end{tabular}")
    support_rows = [
        r"\begin{tabular}{rrrrr}\toprule",
        r"$K$ & MAE (mg/L) & Seed SD & RMSE (mg/L) & Reduction (\%)\\\midrule",
    ]
    for row in support.itertuples():
        support_rows.append(
            f"{row.k} & {row.mae:.3f} & {row.seed_mae_sd:.3f} & "
            f"{row.rmse:.3f} & {row.reduction_pct:.2f}" + r"\\")
    support_rows.append(r"\bottomrule\end{tabular}")

    for name, rows in [
        ("SCENARIOS", scenarios), ("MAIN RESULTS", main_rows),
        ("COMPONENTS", components), ("SUPPORT RESULTS", support_rows),
        ("TAIL RESULTS", tail),
    ]:
        source = block(source, name, rows)
    for original, destination in [
        (HYBRID / "doc_hybrid_performance.pdf", "doc_hybrid_performance.pdf"),
        (HYBRID / "doc_spatial_support_publication.pdf", "doc_spatial_support_v2.pdf"),
        (HYBRID / "doc_station_response_publication.pdf", "doc_station_response_v2.pdf"),
    ]:
        shutil.copyfile(original, TEX.parent / "figures" / destination)
    for asset in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", source):
        if not (TEX.parent / asset).is_file():
            raise FileNotFoundError(asset)
    if r"\end{document}" not in source:
        raise ValueError("Incomplete LaTeX document")
    TEX.write_text(source)
    print(f"Refreshed five tables and three empirical figure assets: {TEX}")


if __name__ == "__main__":
    main()
