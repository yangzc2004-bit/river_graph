"""Document stability checks after the primary active-structure results were inspected."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from river_graph.experiments.provenance import sha256_file

ROOT = Path("experiments/phase4_transfer/doc_river_active_structure_v1")


def main():
    path = ROOT/"analysis/network_hydro_changes.csv"
    frame = pd.read_csv(path, dtype={"target": str, "huc4": str})
    metrics = ["effective_fraction_adjusted_change", "effective_fraction_raw_change", "path_sd_reference_adjusted_change"]
    rows = []
    for omit in sorted(frame.component.unique()):
        selected = frame.loc[frame.component.ne(omit)]
        for group, sub in (("all", selected), ("class_1", selected.loc[selected.cluster.eq(1)]),
                           ("class_3", selected.loc[selected.cluster.eq(3)])):
            for metric in metrics:
                rows.append({"omitted_system": int(omit), "group": group, "metric": metric,
                    "estimate": sub[metric].mean(), "n_networks": len(sub), "n_systems": sub.component.nunique()})
    pd.DataFrame(rows).to_csv(ROOT/"leave_system_out.csv", index=False)
    coverage = []
    for cut in (0, .5, .8, .9):
        selected = frame.loc[frame.covered_area_fraction.ge(cut)]
        for metric in metrics:
            coverage.append({"minimum_area_coverage": cut, "metric": metric, "estimate": selected[metric].mean(),
                "n_networks": len(selected), "n_systems": selected.component.nunique(),
                "n_elongated": int(selected.cluster.eq(1).sum()), "n_broad": int(selected.cluster.eq(3).sum())})
    pd.DataFrame(coverage).to_csv(ROOT/"coverage_sensitivity.csv", index=False)
    report = {"status": "completed", "primary_results_known_before_checks": True,
        "checks": ["Omit each overlap system without changing within-network fits", "Retain fixed mapped-area coverage ladder"],
        "input_hash": sha256_file(path), "code_hash": sha256_file(Path(__file__)),
        "coverage_note": "Descriptive sensitivity; cuts change networks and cannot identify a within-network treatment effect",
        "primary_outputs_modified": False}
    (ROOT/"diagnostic_sources.json").write_text(json.dumps(report, indent=2)+"\n")
    print(pd.DataFrame(rows).query("group == 'all' and metric == 'effective_fraction_adjusted_change'").to_string(index=False))
    print(pd.DataFrame(coverage).query("metric == 'effective_fraction_adjusted_change'").to_string(index=False))


if __name__ == "__main__":
    main()
