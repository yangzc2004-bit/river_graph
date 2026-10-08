"""Check recovered sampling support and independent passive-pulse identities."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from analyze_doc_river_morphology_closure_v1 import ROOT, TYPES

from river_graph.analysis.river_cosampling_geometry import within_month_dates
from river_graph.experiments.provenance import sha256_file


def main():
    ledger = json.loads((ROOT / "analysis_sources.json").read_text())
    for category in ("source_hashes", "output_hashes"):
        for path, expected in ledger[category].items():
            if sha256_file(path) != expected:
                raise ValueError(f"changed {category}: {path}")
    folder = ROOT / "analysis"
    events = pd.read_parquet(folder / "recovered_activities.parquet")
    if events.duplicated(["site_no", "event_id"]).any():
        raise ValueError("activity duplicates inflate observations")
    dates = {s: set(g.date) for s, g in events.groupby("site_no")}
    pairs = pd.read_csv(folder / "recovered_source_pairs.csv", dtype={**TYPES, "source_a": str, "source_b": str})
    for r in pairs.itertuples():
        common = pd.DatetimeIndex(sorted(dates[r.target] & dates[r.source_a] & dates[r.source_b]))
        if (len(common) != r.n_common_days or len(within_month_dates(common)) != r.n_within_month_days
                or common.to_period("M").nunique() != r.n_common_year_months):
            raise ValueError("independent date intersection differs")
    added = pd.read_parquet(folder / "additional_activities.parquet")
    combined = pd.concat([events, added], ignore_index=True)
    if combined.duplicated(["site_no", "event_id"]).any():
        raise ValueError("additional public stations duplicated sampling activities")
    all_dates = {s: set(g.date) for s, g in combined.groupby("site_no")}
    added_pairs = pd.read_csv(folder / "additional_source_pairs.csv", dtype={**TYPES, "source_a": str, "source_b": str})
    for r in added_pairs.itertuples():
        common = pd.DatetimeIndex(sorted(all_dates[r.target] & all_dates[r.source_a] & all_dates[r.source_b]))
        if len(common) != r.n_common_days or len(within_month_dates(common)) != r.n_within_month_days:
            raise ValueError("expanded public-station date intersection differs")
    error = 0.
    for prefix in ("", "shortest_"):
        f = pd.read_csv(folder / f"{prefix}mechanism_scenarios.csv", dtype=TYPES)
        np.testing.assert_allclose(f.pulse_sd**2, f.input_variance+f.arrival_variance+f.corridor_variance, atol=1e-12)
        if f.pulse_peak.gt(1+1e-9).any() or f.minimum_translation.lt(-1e-9).any():
            raise ValueError("passive unit-gain peak or causality violated")
        if not np.isfinite(f.select_dtypes(include="number")).all().all():
            raise ValueError("nonfinite mechanism metric")
        a = f[f.scenario.eq("junction_translation_0.2")].set_index(["target", "sigma"])
        b = f[f.scenario.eq("junction_translation_0.8")].set_index(["target", "sigma"]).reindex(a.index)
        delta = abs(a[["pulse_peak", "pulse_sd", "pulse_central80"]]-b[["pulse_peak", "pulse_sd", "pulse_central80"]]).to_numpy().max()
        error = max(error, float(delta))
        if delta > 1e-8:
            raise ValueError("translation-only junction experiment changed the pulse")
        base = f[f.scenario.eq("actual_paths")].set_index(["target", "sigma"])
        spreading = f[f.scenario.eq("actual_shared_0.5")].set_index(["target", "sigma"]).reindex(base.index)
        np.testing.assert_allclose(base.pulse_centroid, spreading.pulse_centroid, atol=1e-12)
        if (spreading.pulse_peak > base.pulse_peak+1e-8).any():
            raise ValueError("positive common spreading increased mixture maximum")
    report = {"status": "verified", "date_intersections_checked": len(pairs),
        "activities_checked": len(events), "additional_pair_rows_checked": len(added_pairs),
        "additional_activities_checked": len(added), "junction_null_max_metric_error": error,
        "independent_variance_identity": True, "source_and_output_hashes": True,
        "training_performed": False}
    (ROOT / "verification.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
