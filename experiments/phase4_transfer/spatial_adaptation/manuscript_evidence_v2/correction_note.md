# Unified manuscript evidence

All tables and figures in this directory use the same frozen 120-tree, five-seed regional product and fixed 2,316-cell query. No predictions were changed and no model was trained.

## What was reconciled

The previous main table was cell-weighted, whereas its bootstrap point was station-equal. The primary bootstrap now recomputes the cell-weighted statistic after sampling whole stations; station-equal results are a separately labelled secondary summary. The previous map, bias plot and shuffle diagnostic came from the separate 300-tree experiment. They are recomputed here from the 120-tree product.

K=5 MAE = 2.023486 mg/L; reduction = 17.9306%; delta = -0.442093, 95% CI [-1.026121, -0.088648].

36/43 stations improve; median station reduction = 3.3890%. Bias/gain Pearson r = 0.9211. Bias and gain share query errors and this association is descriptive. Seasonal or hydroclimatic transfer is a modeling rationale rather than a mechanism established by these comparisons.

Support candidates are the first, middle, last, first-quarter and third-quarter observed months, used as nested prefixes. Their labels only determine the log1p residual correction. Because support spans the record, this is retrospective reconstruction.
