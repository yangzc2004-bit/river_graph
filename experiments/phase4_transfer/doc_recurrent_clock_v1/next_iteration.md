# Next focused experiment: conditional distribution at the residual head

## Motivation

Readout learning and recurrent-clock changes do not improve the integrated
main candidate. High-DOC underprediction remains more pronounced than ordinary
error. Existing source-validation diagnostics also find missed high observations
when flow is available; raw-label checks verified the audited extremes.
Simple tail weighting and ordinary-error penalties have already been tested.
The next experiment should test information compression at the prediction head,
not repeat another tail-weight sweep or change the graph stack.

## Minimal change

Retain the daily-head/legacy-support integrated expert. Freeze its encoder,
GRU, decay, forest, ecological profile and existing native point head.
Use the same hidden64, current38 head features and seven hidden×feature
interactions (550 head features). Compare:

1. Existing point prediction, reused unchanged.
2. A single-Gaussian log1p residual density head.
3. Two ordered Gaussian log1p residual components with a learned mixture
   probability. Interpret components as conditional statistical states,
   not identified ecological processes.

The baseline is the frozen native prediction. Source training uses the same
station-fold-hidden features and OOF forest plus frozen source-trained neural
correction; only the forest is OOF. Reconstruct that exact source baseline
rather than using the full-source forest at source loss cells.

Output the density's **median**, followed by the monotone nonnegative native
transform, rather than its mean. This keeps the point estimator aligned with
native MAE. A density may represent a high-concentration tail without forcing
every ordinary prediction upward. It may also fail to improve the point median;
that is an informative result.

## Bounded experiment

DOC, the same three spatial partitions and three seeds. Two new head-only
fits per package,18 fits. No encoder, recurrent or forest optimization.
Fit input normalization and initial density summaries from source loss cells
only. Optimize unweighted source likelihood; do not introduce a new Q90 weight
grid or graph-trust gate. Select checkpoints using unweighted held
source-validation fixed-query native MAE, retaining the original point
prediction as a selectable reference. Fix numerical scale floors and the
median solver before training. Report head parameter counts: the mixture
changes expressiveness as well as the loss, so the single-component control
matters.

Carry unchanged constant/legacy support bases and the existing source-validation
alpha/ridge/ecological mixture selection. Use the same fixed K0/1/3/5 query
population. Report ordinary and Q90 MAE, bias, false-high rate, all-K MAE and
station/partition consistency. Likelihood, scale and component diagnostics are
secondary; density fit or wider tail intervals alone do not justify point-model
promotion. Do not introduce new uncertainty or active-sampling claims.

This is a prepared research direction, not a completed fit. Implement in a new
versioned directory, preserve all previous comparisons, and promote it only
from an actual point-prediction improvement over the current main candidate.
