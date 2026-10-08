"""Replay paths/scenarios and independently check spectral variance and intervals."""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from analyze_doc_river_signal_timescale_v1 import CODE, ROOT, TYPES, calculate
from scipy.integrate import quad

from river_graph.experiments.provenance import sha256_file


def independent_spectral_variance(d, w, time, coherence, memory):
    """Integrate each distinct path-difference covariance, not its closed form."""
    result = 0.
    for i, a in enumerate(d):
        for j, b in enumerate(d):
            scale = coherence if i != j else 1.
            def density(omega):
                return 2*time/(np.pi*(1+(time*omega)**2)*(1+(memory*omega)**2))
            difference = abs(a-b)
            if difference:
                value = quad(density, 0, np.inf, weight="cos", wvar=difference, epsabs=1e-9)[0]
            else:
                value = quad(density, 0, np.inf, epsabs=1e-9)[0]
            result += w[i]*w[j]*scale*value
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shortest-routes", action="store_true")
    p.add_argument("--full-replay", action="store_true")
    args = p.parse_args()
    root = ROOT/"shortest_route_sensitivity" if args.shortest_routes else ROOT
    ledger = json.loads((root/"analysis_sources.json").read_text())
    for part in ("source_hashes", "output_hashes"):
        for name, expected in ledger[part].items():
            if sha256_file(name) != expected:
                raise ValueError(f"changed analysis artifact: {name}")
    if sha256_file(root/"config.json") != ledger["config_sha256"]:
        raise ValueError("changed scenario configuration")
    for path in CODE:
        if sha256_file(path) != sha256_file(root/"code_snapshot"/path):
            raise ValueError(f"snapshot disagrees with executable source: {path}")
    for suffix in ("", "_cn"):
        receipt = json.loads((root/"figures"/f"figure_sources{suffix}.json").read_text())
        for part in ("input_hashes", "output_hashes"):
            for name, expected in receipt[part].items():
                if sha256_file(name) != expected:
                    raise ValueError(f"changed figure or supporting data: {name}")
    data = root/"analysis"
    source = pd.read_csv(data/"source_paths.csv", dtype=TYPES)
    scenario = pd.read_parquet(data/"scenario_metrics.parquet")
    working = scenario[scenario.arrangement.eq("actual_paths") & scenario.correlation_time.eq(.5)
        & scenario.coherence.eq(.5) & scenario.memory_allocation.eq(.5)]
    errors = []
    for r in working.itertuples():
        f = source[source.target.eq(r.target)].sort_values("source_order")
        d, w = f.normalized_delay.to_numpy(), f.area_weight.to_numpy()
        value = independent_spectral_variance(d, w, .5, .5, r.memory_time)
        errors.append(abs(value-r.outlet_variance))
    if max(errors) > 2e-8:
        raise ValueError("frequency integral disagrees with closed-form variance")
    draws = json.loads((root/"config.json").read_text())["bootstrap_draws"]
    f = working[working.covered_area_fraction.ge(.8) & working.cluster.isin([1, 3])]
    groups = [g for _, g in f.groupby("component", sort=True)]
    rng = np.random.default_rng(42)
    boot = []
    # Concatenate complete independently sampled systems; do not call cluster_mean.
    for counts in rng.multinomial(len(groups), np.full(len(groups), 1/len(groups)), size=draws):
        sample = pd.concat([group for group, count in zip(groups, counts, strict=True) for _ in range(count)], ignore_index=True)
        if sample.cluster.eq(1).any() and sample.cluster.eq(3).any():
            boot.append(sample.loc[sample.cluster.eq(3), "sd_ratio"].mean()-sample.loc[sample.cluster.eq(1), "sd_ratio"].mean())
    # Concatenating complete blocks uses the same fixed draws but independently
    # computes the estimand; it does not reuse the analysis bootstrap helper.
    point = f.loc[f.cluster.eq(3), "sd_ratio"].mean()-f.loc[f.cluster.eq(1), "sd_ratio"].mean()
    contrast = pd.read_csv(data/"form_contrasts.csv").query(
        "minimum_coverage == .8 and correlation_time == .5 and coherence == .5 and memory_allocation == .5 and metric == 'sd_ratio'").iloc[0]
    np.testing.assert_allclose(point, contrast.estimate, atol=1e-12)
    independent_interval = np.quantile(boot, [.025, .975])
    np.testing.assert_allclose(independent_interval, [contrast.ci_low, contrast.ci_high], atol=1e-12)
    if args.full_replay:
        tables, summary, _ = calculate(draws=draws, shortest=args.shortest_routes)
        for name, actual in tables.items():
            path = data/f"{name}.{'parquet' if name in ('scenario_metrics', 'illustration_waves') else 'csv'}"
            stored = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path, dtype=TYPES)
            pd.testing.assert_frame_equal(actual, stored, check_dtype=False, check_exact=False, atol=1e-11, rtol=1e-10)
        if summary != json.loads((data/"summary.json").read_text()):
            raise ValueError("summary replay mismatch")
    note = ("# Verification\n\n"
        f"- Source, configuration, execution snapshots and both figure receipts agree.\n"
        f"- Independent frequency integration checks all {len(working)} working-point receiving networks; maximum absolute error {max(errors):.3g}.\n"
        f"- Independent complete-system resampling reproduces the high-coverage form point {point:.8f} and 95% interval [{independent_interval[0]:.8f}, {independent_interval[1]:.8f}].\n"
        f"- Full scenario, geometry, illustration and bootstrap replay: {'performed' if args.full_replay else 'not requested'}.\n")
    (root/"verification.md").write_text(note)
    print(note)


if __name__ == "__main__":
    main()
