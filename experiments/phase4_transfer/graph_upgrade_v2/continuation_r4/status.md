# R4 status: complete

The observation-aware multi-scale pilot is complete: 18/18 configurations
finished with finite predictions and complete V5 sidecars. The causal contract
and four-path gradient tests pass. Compact paired tables are in:

- `m13_vs_m1/paired_summary.csv`
- `m13_vs_m3/paired_summary.csv`
- `m13_vs_rf/paired_summary.csv`

## Pilot result

| analyte | holdout | M13 MAE | M1 MAE | M13 vs M1 |
|---|---|---:|---:|---:|
| DOC | temporal | 1.307 | 1.143 | -14.3% |
| DOC | spatial | 3.404 | 3.490 | +2.5% |
| pH | temporal | 0.268 | 0.272 | +1.1% |
| pH | spatial | 0.274 | 0.280 | +2.1% |
| specific conductance | temporal | 195.4 | 191.0 | -2.3% |
| specific conductance | spatial | 532.5 | 489.5 | -8.8% |

The combination does not beat M1 in temporal DOC or conductance. It gives a
small spatial gain for DOC and pH, but the spatial conductance result is worse.
It remains behind the temporal RF on DOC and conductance. The useful scientific
conclusion is conditional complementarity: multi-scale paths help some spatial
transfer tasks, while observation-aware memory remains the stronger general
default. No full formal training expansion is justified by this pilot alone.

Q90 cells are shown separately in the paired tables; temporal DOC has only 12
high-value query cells and is flagged unstable. Seed variation is retained in
the raw run metrics rather than selecting a best seed.
