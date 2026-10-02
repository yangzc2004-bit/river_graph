# Spatial transfer: unified manuscript evidence

This revision reconciles all primary tables and figures to the frozen
120-tree regional ExtraTrees product (source pool 40; seeds 42–46). It
supersedes the mixed-product summaries in v1. No model was retrained and no
prediction was changed.

## Main result

The evaluation contains 43 held-out DOC stations. Five candidate observations
per station are reserved at every support level, leaving the same 2,316 query
cells. Losses are averaged over seeds and query cells, not calculated from an
ensemble-mean prediction.

| K per station | MAE (mg/L) | Reduction | Paired ΔMAE | Station-resampled 95% CI |
|---:|---:|---:|---:|---:|
| 0 | 2.466 | 0.00% | 0.000 | — |
| 1 | 2.320 | 5.90% | −0.146 | [−0.427, 0.033] |
| 3 | 2.160 | 12.41% | −0.306 | [−0.753, −0.037] |
| 5 | 2.023 | 17.93% | −0.442 | [−1.026, −0.089] |

Each bootstrap resamples whole stations and recomputes the cell-weighted loss.
The previously quoted −0.419 mg/L is the **station-equal** difference and is
retained separately; it was not the point estimate corresponding to the main
cell-weighted table.

## Station response and shuffle

- 36/43 stations improve; median station-level reduction is 3.39%.
- Bias/gain Pearson r is 0.921, Spearman r is 0.495. Excluding the largest-bias
  station reduces Pearson r to 0.575. Because both axes share baseline errors,
  this is a descriptive relationship, not independent mechanism evidence.
- Matching support residuals gives K5 MAE 2.023; permuting corrections across
  stations gives 2.736. One predetermined permutation per seed is used, five
  in total at K5. This is a descriptive diagnostic, not a significance test.
- Old values 34/43 and 2.5% came from the separate 300-tree experiment. Those
  original artifacts remain unchanged but are not the source of this draft's
  main figures.

## Method interpretation

The regional expert uses 39 RF-context features, including visible global,
upstream and downstream DOC summaries. Its 40-station fitting pool is selected
from label-free hydroclimatic, ecological, geographic and graph descriptors.
All known station descriptors enter their standardization. This is a known-
cohort, retrospective spatial reconstruction setting, not an independent
external-basin test.

The adapter adds a station-constant log1p residual. Support candidates are the
first, middle, last, first-quarter and third-quarter observed positions, used
in that nested order. Thus support can occur after query months. Alpha values
0.25/0.50/0.75 are inherited from internal validation in the earlier 300-tree
adapter experiment, then applied unchanged to this 120-tree product.

The regional model plus local calibration is an operational decomposition.
This result has not isolated seasonal or hydroclimatic transfer through feature
ablation. The outer split has also been reused during model development.

## Other comparisons

The later three-seed learner comparison gives K5 MAE 2.024 (ExtraTrees), 2.048
(random forest), and 2.086 (histogram gradient boosting). Internal validation
selected random forest; its outer result did not improve the existing product.
The identical gradient-boosting predictions across seeds add no independent
stochastic evidence.

The earlier KGML numbers 2.823 → 2.544 are not included in this manuscript's
quantitative comparisons: the review could trace them to progress prose but
not yet to matching prediction products. This removes an unsupported numerical
comparison; it does not establish that KGML adaptation fails.

## Files and reproduction

- Analysis: `scripts/analyze_spatial_manuscript_evidence.py`
- Tables, figures and correction note:
  `experiments/phase4_transfer/spatial_adaptation/manuscript_evidence_v2/`
- Figure/number injection: `scripts/refresh_spatial_manuscript.py`
- Manuscript: `docs/paper/latex/spatial_transfer_draft_v1.tex`

```bash
PYTHONPATH=.:scripts uv run --no-sync python scripts/analyze_spatial_manuscript_evidence.py
uv run --no-sync python scripts/refresh_spatial_manuscript.py
```

The three empirical plots embed vector coordinates directly in the manuscript.
Figure 1 is the image-generated architecture schematic in
`docs/paper/latex/figures/regional_calibration_modular_architecture_v4.png`; its generation
and revision prompts are saved alongside it. Compile with the adjacent `figures/`
directory present. The desktop single-file compiler cannot load that asset, so
the complete PDF was exported with the bundled local Tectonic engine
and all pages were visually checked. The numerical results are unchanged.
