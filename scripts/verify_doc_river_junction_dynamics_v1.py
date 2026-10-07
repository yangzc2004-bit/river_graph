"""Replay fixed geometry scenarios and check analysis/figure source bindings."""
from __future__ import annotations

import argparse
import io
import json

import pandas as pd
from analyze_doc_river_junction_dynamics_v1 import (
    CODE,
    DT,
    ROOT,
    SETTINGS,
    inputs,
    numerical_summary,
    simulate,
    summaries,
)

from river_graph.experiments.provenance import sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-replay", action="store_true")
    args = parser.parse_args()
    binding = json.loads((ROOT/"analysis_sources.json").read_text())
    for kind in ("input_hashes", "output_hashes"):
        for name, expected in binding[kind].items():
            if sha256_file(name) != expected:
                raise ValueError(f"Changed analysis source or output: {name}")
    for path in CODE:
        if sha256_file(path) != sha256_file(ROOT/"code_snapshot"/path):
            raise ValueError(f"Execution snapshot differs: {path}")
    cases, reps, examples, pairs, field, proxies = inputs()
    replayed = 0
    if args.full_replay:
        frame, curves = simulate(cases, examples)
        for name, actual in (("scenario_metrics", frame), ("example_curves", curves)):
            pd.testing.assert_frame_equal(pd.read_parquet(ROOT/f"analysis/{name}.parquet"), actual,
                                          check_dtype=False, rtol=1e-12, atol=1e-12)
        reports = summaries(frame, pairs, draws=binding["bootstrap_draws"])
        fine, _ = simulate(cases.loc[cases.case_id.isin(examples)], set(), dt=DT/2, keep_curves=False)
        keys = ["cohort", "case_id"]+SETTINGS
        refined = frame[keys+["pulse_peak", "duration_80"]].merge(fine[keys+["pulse_peak", "duration_80"]],
            on=keys, suffixes=("_coarse", "_fine"), validate="one_to_one")
        for metric in ("pulse_peak", "duration_80"):
            refined[metric+"_difference"] = refined[metric+"_fine"]-refined[metric+"_coarse"]
        tables = {"geometry": cases, "representatives": reps, "field_transit_proxies": field,
                  "field_transect_proxies": proxies, "numerical_refinement": refined} | reports
        for name, actual in tables.items():
            serialized = pd.read_csv(io.StringIO(actual.to_csv(index=False)))
            pd.testing.assert_frame_equal(pd.read_csv(ROOT/f"analysis/{name}.csv"), serialized,
                                          check_dtype=False, rtol=1e-12, atol=1e-12)
        if numerical_summary(frame, refined) != json.loads((ROOT/"analysis/summary.json").read_text()):
            raise ValueError("Summary differs from full replay")
        replayed = len(frame)
    for name in ("figure_sources.json", "figure_sources_cn.json"):
        path = ROOT/"figures"/name
        if path.exists():
            figure = json.loads(path.read_text())
            for kind in ("input_hashes", "output_hashes"):
                for file, expected in figure[kind].items():
                    if sha256_file(file) != expected:
                        raise ValueError(f"Changed figure source or output: {file}")
    result = {"status": "pass", "scenario_rows_replayed": replayed,
              "n_whole_junctions": int(cases.cohort.eq("whole_confluence").sum()),
              "n_monitored_footprints": int(cases.cohort.eq("monitored_footprint").sum()),
              "n_original_elongated_broad_pairs": len(pairs), "n_field_junctions": len(field),
              "original_morphology_labels_retained": True, "new_training": False}
    (ROOT/"verification.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
