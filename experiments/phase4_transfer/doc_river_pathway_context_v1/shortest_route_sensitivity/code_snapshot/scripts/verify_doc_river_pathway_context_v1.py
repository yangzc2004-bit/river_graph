"""Verify receiving variance identities, cropped routes and full replay."""

from __future__ import annotations

import argparse
import io
import json

import numpy as np
import pandas as pd
from analyze_doc_river_pathway_context_v1 import ROOT, TYPES, calculate

from river_graph.experiments.provenance import sha256_file


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--full", action="store_true")
    p.add_argument("--shortest-routes", action="store_true")
    args = p.parse_args()
    root = ROOT/"shortest_route_sensitivity" if args.shortest_routes else ROOT
    receipt = json.loads((root/"analysis_sources.json").read_text())
    for key in ("source_hashes", "output_hashes"):
        for path, expected in receipt[key].items():
            if sha256_file(path) != expected:
                raise ValueError(f"changed analysis source or result: {path}")
    data = root/"analysis"
    series = pd.read_parquet(data/"variance_series.parquet")
    budgets = pd.read_csv(data/"receiving_variance_budgets.csv", dtype=TYPES)
    for (target, subset, weighting), f in series.groupby(["target", "subset", "weighting"]):
        r = budgets[budgets.target.eq(target) & budgets.subset.eq(subset) & budgets.weighting.eq(weighting)].iloc[0]
        m, y = f.mixture_anomaly.to_numpy(), f.receiver_anomaly.to_numpy()
        # Independent direct check, not a call to the budget helper.
        discrepancy = y-m
        vm, vy, vd = np.var(m), np.var(y), np.var(discrepancy)
        cross = 2*np.cov(m, discrepancy, ddof=0)[0, 1]
        np.testing.assert_allclose([vm, vy, vd, cross],
            [r.mixture_variance, r.outlet_variance, r.mismatch_variance, r.twice_mix_mismatch_covariance], atol=1e-10)
        np.testing.assert_allclose(vy, vm+vd+cross, atol=1e-10)
        np.testing.assert_allclose([m.mean(), y.mean()], 0., atol=1e-9)
    receiver = pd.read_csv(data/"receiver_pathway_panel.csv", dtype=TYPES)
    if receiver.target.duplicated().any() or receiver.comid.duplicated().any():
        raise ValueError("physical receivers must not be duplicated")
    # Independently resample complete systems and concatenate their rows.
    # This checks the consequential class contrast without the analysis helper.
    high = receiver[receiver.covered_area_fraction.ge(.8) & receiver.cluster.isin([1, 3])]
    blocks = [g for _, g in high.groupby("component", sort=True)]
    rng = np.random.default_rng(42)
    repetitions = rng.multinomial(len(blocks), np.full(len(blocks), 1/len(blocks)), size=5000)
    differences = []
    block_values = [(g.cluster.to_numpy(), g.outlet_mix_log_sd_ratio.to_numpy()) for g in blocks]
    for counts in repetitions:
        classes = np.concatenate([np.tile(c, n) for n, (c, v) in zip(counts, block_values, strict=True)])
        values = np.concatenate([np.tile(v, n) for n, (c, v) in zip(counts, block_values, strict=True)])
        if np.any(classes == 1) and np.any(classes == 3):
            differences.append(values[classes == 3].mean()-values[classes == 1].mean())
    saved = pd.read_csv(data/"form_contrasts.csv")
    saved = saved[saved.minimum_coverage.eq(.8) & saved.metric.eq("outlet_mix_log_sd_ratio")].iloc[0]
    point = high[high.cluster.eq(3)].outlet_mix_log_sd_ratio.mean()-high[high.cluster.eq(1)].outlet_mix_log_sd_ratio.mean()
    np.testing.assert_allclose(point, saved.estimate, atol=1e-12)
    if json.loads((root/"config.json").read_text())["bootstrap_draws"] == 5000:
        np.testing.assert_allclose(np.quantile(differences, [.025, .975]), [saved.ci_low, saved.ci_high], atol=1e-10)
    pairs = pd.read_csv(data/"same_region_form_pairs.csv")
    expected = pairs.area_comparable & pairs.both_observed & pairs.both_high_coverage
    np.testing.assert_array_equal(expected, pairs.usable_form_pair)
    reaches = pd.read_csv(data/"source_corridor_reaches.csv", dtype=TYPES)
    if reaches.duplicated(["target", "source_station", "comid"]).any() or reaches.length_km.lt(0).any():
        raise ValueError("unique nonnegative cropped reaches required")
    if args.full:
        config = json.loads((root/"config.json").read_text())
        tables, summary, _ = calculate(draws=config["bootstrap_draws"], shortest=args.shortest_routes)
        for name, expected in tables.items():
            path = data/f"{name}.parquet"
            if path.exists():
                actual = pd.read_parquet(path)
            else:
                actual = pd.read_csv(data/f"{name}.csv", dtype=TYPES)
                expected = pd.read_csv(io.StringIO(expected.to_csv(index=False)), dtype=TYPES)
            pd.testing.assert_frame_equal(actual, expected, check_dtype=False, atol=1e-10, rtol=1e-9)
        if json.loads((data/"summary.json").read_text()) != summary:
            raise ValueError("summary replay differs")
    for path in (root/"figures").glob("figure_sources*.json"):
        f = json.loads(path.read_text())
        for key in ("input_hashes", "output_hashes"):
            for source, expected in f[key].items():
                if sha256_file(source) != expected:
                    raise ValueError(f"changed figure or figure source: {source}")
    message = f"Verified {len(receiver)} physical receiver pathways and {len(budgets)} exact observed variance budgets."
    (root/"verification.md").write_text("# Verification\n\n"+message+"\n\n"+
        ("Full calculation, 5,000 system-bootstrap results and source replay agree.\n" if args.full else "Direct variance and source checks completed.\n")+
        "Shared cropped suffixes, fixed station roles, same-date flow comparisons, missing waterbody context "
        "and metadata-only form comparison opportunities remain explicit.\n")
    print(message, flush=True)


if __name__ == "__main__":
    main()
