# T8 synthesis analysis specification

This analysis is defined after viewing T2--T7 point estimates. It is descriptive
synthesis under the existing 10-epoch / patience-3 budget, not a new preregistered
confirmation test. No model selection or new training is performed here.

## Estimands and uncertainty

- Compare temporal versus snapshot on exactly matched test station-month cells.
- Average cell-wise absolute errors across the five training seeds before
  resampling; seeds do not multiply the ecological sample size.
- Delta MAE = temporal MAE minus snapshot MAE. Negative favors temporal.
- Relative reduction = 100 * (1 - temporal MAE / snapshot MAE).
- Report each analyte and each mask separately. E2 summaries weight E2a/E2b
  equally; never pool raw-scale errors across analytes.
- Use 2,000 paired cluster bootstrap replicates, RNG seed 20260927, percentile
  95% intervals. Report station clusters AND month clusters as separate
  sensitivity analyses; neither is a joint space-time dependence correction.
- Preserve the cell-weighted MAE estimand in each replicate by summing sampled
  cluster errors and dividing by sampled cell counts. No seed resampling.
- Include RMSE, R2, log MAE and train-derived Q90 tail MAE from audited products;
  n<20 tails are flagged unstable. pH Q90 is an upper-tail diagnostic, not a
  general adverse-water-quality threshold.
- Historical ablations and window curves use the matched seeds 42--44 only.

## Interpretation

The comparison measures performance at this budget, not convergence. A longer
GRU unroll also changes recurrent computation; a window-length result alone
does not isolate information in past months. The reverse-history arm retains
ordered information under a fixed permutation. The hydro-only arm removes
target channels at ALL months, including the current month. It does not isolate
historical target values alone, nor does it isolate ecological variables from
hydrology or seasonality. These definitions govern the report.

## Provenance

Re-audit all six source directories using the existing evaluator. Record input
sidecar/product hashes and training metadata. Report historical declarative
metadata errors separately; do not rewrite sidecars or their hashes. A hash
integrity pass does not prove the declared config matches runtime semantics.
