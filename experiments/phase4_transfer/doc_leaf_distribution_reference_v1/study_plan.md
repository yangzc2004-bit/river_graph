# Source-only conditional-distribution reference for DOC

## Question and fixed design

The completed concentration-objective study worsens overall K0 reconstruction;
its native forest helps high concentrations at a large central-error cost.
This follow-up asks whether the environmental reference can preserve its useful
log-fitted partition geometry while changing how source concentrations are read
from those partitions. It continues the existing environmental-plus-neural
residual model; it does not replace the GNN/GRU with a new model family.

Use source station partitions142/143/144 and seeds42/43/44, the same47 station-
hidden covariates, source train labels and all validation DOC cells at K0.
Receiving-site DOC/pH/conductance remain absent. Keep all older predictions.

The empirical distribution follows the leaf-weight idea of
[Meinshausen, Quantile Regression Forests (2006)](https://www.jmlr.org/papers/volume7/meinshausen06a/meinshausen06a.pdf):
each tree assigns equal mass to source fit observations in the query leaf;
average those distributions over the forest. Here the partitions are the already
selected log-fitted ExtraTrees, rather than claiming to reproduce its original
random-forest training algorithm. This is an established distributional readout,
not a newly invented algorithm or an uncertainty calibration result.

Two fixed arms use exactly the same source distribution and partitions:

1. Native leaf-distribution mean (control for concentration aggregation).
2. Leaf-distribution median: the first native DOC value whose normalized
   weighted cumulative probability reaches0.5.

Do not average individual leaf medians and call that the mixture median. Do not
search quantiles, blend arms, change leaf sizes or rebuild splits. Source labels
are the only labels used to populate leaves. The source fold-hiding feature
construction remains identical to the retained forest fit. Compute query weights
in chunks to support arbitrary new station counts without a dense full-grid ×
source-observation matrix. A training concentration contributes through source
experience, never through a receiving-site water-quality input.

## Evaluation and next step

Compare both arms with retained log ExtraTrees and the actual retained complete
model; compare the median with its matched native mean. Report native MAE, Q90
MAE, signed bias, station-equal error, each partition/seed and5,000 paired station
draws, with saved-state replay. Verify the empirical weights normalize and that
the conditional median minimizes the weighted absolute error on a hand-checkable
example. Preserve parent trees/predictions and execution sources.

If the median improves the source reference reliably, generate new nested
station-hidden OOF distributional predictions and refit the SAME retained
ecology/GRU/native residual and ecological-memory fusion. Neither source targets
nor held-station feature histories may enter their own OOF reference. That
integration requires a separate saved experiment after this comparison, not
in-sample residual subtraction. Existing geography/external outcomes do not
select the new readout. If the readout fails, retain the full model and record
the finding before the next source mechanism.
