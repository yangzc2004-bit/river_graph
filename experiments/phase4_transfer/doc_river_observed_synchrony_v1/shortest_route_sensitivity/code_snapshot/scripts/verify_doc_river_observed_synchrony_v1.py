"""Reconcile actual observations and independently check synchrony calculations."""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from analyze_doc_river_observed_synchrony_v1 import ARRIVALS, ROOT, TYPES, calculate

from river_graph.experiments.provenance import sha256_file


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shortest-routes", action="store_true")
    p.add_argument("--full", action="store_true")
    args = p.parse_args()
    root = ROOT/"shortest_route_sensitivity" if args.shortest_routes else ROOT
    arrivals = ARRIVALS/"shortest_route_sensitivity" if args.shortest_routes else ARRIVALS
    ledger = json.loads((root/"analysis_sources.json").read_text())
    for name in ("source_hashes", "output_hashes"):
        for path, expected in ledger[name].items():
            if sha256_file(path) != expected:
                raise ValueError(f"changed content: {path}")
    for suffix in ("", "_cn"):
        receipt = root/"figures"/f"figure_sources{suffix}.json"
        record = json.loads(receipt.read_text())
        for name in ("input_hashes", "output_hashes"):
            for path, expected in record[name].items():
                if sha256_file(path) != expected:
                    raise ValueError(f"figure content changed: {path}")
    selected = pd.read_parquet(arrivals/"analysis/selected_activities.parquet")
    point = pd.read_csv(root/"analysis/network_metrics.csv", dtype=TYPES)
    primary = point.query("version == 'monthly' and adjustment == 'calendar_year' and excursion_quantile == .75")
    maximum = 0.
    for row in primary.itertuples():
        g = selected[selected.target.eq(row.target)]
        pivot = g.pivot(index="month", columns="source_order", values="doc_monthly").sort_index()
        time = pd.DatetimeIndex(pivot.index)
        orders = sorted(c for c in pivot.columns if c >= 0)
        w = g[g.source_order.ge(0)].groupby("source_order").area_weight.first().reindex(orders).to_numpy(float)
        w = w/w.sum()
        values = pivot[[*orders, -1]].to_numpy(float)
        year = time.year.to_numpy(float)
        design = np.column_stack([np.ones(len(time)), np.sin(2*np.pi*time.month/12),
            np.cos(2*np.pi*time.month/12), year-year.mean()])
        anomalies = values-design@np.linalg.lstsq(design, values, rcond=None)[0]
        anomalies -= anomalies.mean(axis=0)
        a, y = anomalies[:, :-1], anomalies[:, -1]
        sd = a.std(axis=0)
        covariance = np.cov(a, rowvar=False, bias=True)
        numerator = denominator = 0.
        for i in range(len(w)):
            for j in range(i+1, len(w)):
                numerator += w[i]*w[j]*covariance[i, j]
                denominator += w[i]*w[j]*sd[i]*sd[j]
        expected = numerator/denominator
        maximum = max(maximum, abs(row.source_coherence-expected))
        np.testing.assert_allclose(row.source_coherence, expected, atol=1e-10)
        np.testing.assert_allclose(row.outlet_sync_log_sd_ratio, np.log(y.std()/(w@sd)), atol=1e-10)
        count = (a > np.quantile(a, .75, axis=0)).sum(axis=1)
        high = y > np.quantile(y, .75)
        c, s = count >= 2, count == 1
        assert row.n_coincident == c.sum() and row.n_solo == s.sum()
        assert row.n_receiver_high_coincident == (c & high).sum()
        assert row.n_receiver_high_solo == (s & high).sum()
        if min(c.sum(), s.sum()) >= 5:
            np.testing.assert_allclose(row.peak_risk_difference, high[c].mean()-high[s].mean())
    summaries = pd.read_csv(root/"analysis/signal_summary.csv")
    summary = summaries.query("version == 'monthly' and adjustment == 'calendar_year' and excursion_quantile == .75 and minimum_coverage == 0 and group == 'all' and metric == 'peak_risk_difference'").iloc[0]
    eligible = primary.dropna(subset=["peak_risk_difference"])
    blocks = sorted(eligible.component.unique())
    multiplicities = np.random.default_rng(42).multinomial(len(blocks), np.full(len(blocks), 1/len(blocks)), size=5000)
    # Explicitly assemble complete-system draws instead of calling cluster_mean.
    values = [eligible.loc[eligible.component.eq(b), "peak_risk_difference"].to_numpy() for b in blocks]
    resampled = [np.concatenate([value for value, n in zip(values, count, strict=True) for _ in range(n)]).mean()
                 for count in multiplicities]
    np.testing.assert_allclose([summary.estimate, summary.ci_low, summary.ci_high],
        [eligible.peak_risk_difference.mean(), *np.quantile(resampled, [.025, .975])], atol=1e-10)
    if args.full:
        tables, expected, _ = calculate(draws=5000, shortest=args.shortest_routes)
        for name, frame in tables.items():
            path = root/"analysis"/f"{name}.{'parquet' if name.endswith('series') else 'csv'}"
            saved = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path, dtype={k: v for k, v in TYPES.items() if k in frame})
            pd.testing.assert_frame_equal(saved, frame.reset_index(drop=True), check_dtype=False, atol=1e-9, rtol=1e-8)
        if json.loads((root/"analysis/summary.json").read_text()) != expected:
            raise ValueError("summary replay differs")
    text = ("# Verification\n\n"
        "- Source, saved configuration, analysis products and both figure receipts agree.\n"
        f"- Independently reconstructed calendar projection, pair covariance and peak counts for all {len(primary)} monthly networks; maximum coherence error {maximum:.3g}.\n"
        "- Independent complete-system resampling reproduces the primary excursion-risk mean and 95% interval.\n"
        f"- Full observed-panel and sensitivity replay: {'performed' if args.full else 'not requested'}.\n")
    (root/"verification.md").write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
