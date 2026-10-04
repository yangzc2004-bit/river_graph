# Conditional distributional residual heads for DOC reconstruction

## Question

Can the current model's features separate ordinary and high-concentration
conditional states better than one scalar residual head? Prior source-validation
diagnostics show missed high DOC even with observed flow. Readout and recurrent
clock changes did not improve the integrated main candidate.

## Fixed model and information

DOC, existing spatial partitions142/143/144 and seeds42/43/44. Retain each
selected daily-head (`off`) expert from `doc_daily_hydro_memory_v1`: forest,
ecological/self encoder, GRU, decay and native head are frozen. Use its same
550 head features: hidden64, current extras38 and hidden-major interactions
with existing indices{0,2,4,28,30,31,32}. No new covariate or graph is added.

Source views hide all DOC at each receiving station fold. Source native base
is max(0, forest OOF + selected native scale × frozen neural residual), with
the actual38-channel source extras. Only the forest component is OOF; the
neural expert has trained on source labels. Validation/test use the full-source
forest and unchanged frozen native prediction. Feature mean/std are fitted
only at source loss cells; std below1e-6 is replaced with1.

## Three heads

1. **point**: original native prediction, reused exactly.
2. **single**: linear550→2 Gaussian log1p residual location/scale.
3. **mixture**: linear550→5 two-Gaussian log1p residual density: low location,
   positive ordered location gap, two scales and mixture logit.

Residual r=log1p(y)−log1p(native base). Scale=.03+softplus(raw scale).
No upper scale cap is introduced. Mixture high location=low+softplus(gap).
Initialize all linear weights to0. Source-only intercept initialization:
single mean(r), max(std(r),.05); mixture q25/q75 with gap at least.05,
equal weights.5 and both scales max(std(r),.05).

Optimize unweighted source Gaussian/mixture likelihood with stable logsumexp.
The point estimate is the conditional density **median**, not its mean or a
weighted average of component medians. Solve mixture CDF=.5 by64 bisection
steps between min(mu−12sigma) and max(mu+12sigma). Single median equals its
location. Native output=max(0,expm1(log1p(base)+scale×median residual)).
Correction scale0 returns the original base exactly.

The single/mixture head has1102/2755 trained parameters. The mixture changes
capacity and nonlinear conditional expressiveness as well as the likelihood;
the single-component control is retained. Statistical components are not
identified ecological processes or physical concentration regimes.

## Training and selection fixed before fitting

Two new heads ×nine packages=18 fits. Adam LR.001, at most100 epochs,
patience10, source batch512, gradient norm1, torch threads2. Every source
observed loss cell has equal weight; no tail-weight or station-weight sweep.
Checkpoint/global correction scale{0,.25,.5,1} use held source-validation
fixed K0 query native MAE. Epoch0 and the exact original prediction are
selectable; ties favor the smaller scale and earlier checkpoint. NLL cannot
select or promote the point model. Record both NLL and MAE traces.

Carry the unchanged constant and legacy `gru_tuned_anchor` support bases and
frozen ecological residual memory. Refit the existing direct alpha/ridge and
ecological gamma/alpha/ridge grids on source validation. Each K uses the same
reserved-five support schedule and fixed query population as the parent.
Target labels enter only reserved support adaptation and final evaluation.

## Comparisons

Report context and all three heads, direct/integrated where applicable,
K0/1/3/5. Eight overall comparisons: single/mixture versus point, K0/K5,
direct/integrated. Four mechanism comparisons: mixture versus single, the
same K/pipeline combinations. All twelve are specified here before fitting.

Native/log MAE, RMSE, R2, Q90/ordinary error and bias, recall/false-high rate,
partition/seed/station consistency and gain concentration are reported.
Use5000 paired whole-station bootstrap draws; seeds average within partition,
then partitions equally. Joint station identities across overlapping
partitions receive the same bootstrap multiplicity. No target-selected K,
route, threshold or component is used. Source density scale/weight/gap/NLL
diagnostics are separate from target point-prediction evidence.

## Interpretation and products

Promote a head from actual point-prediction improvement over the current main
candidate, checking high DOC and ordinary errors together. Better likelihood
or a broader tail component alone is not a point-performance improvement and
does not establish calibrated uncertainty. If density heads fail, retain the
point model and use their source diagnostics to choose the next hypothesis.

Save trained heads, source normalization/initialization, all selection traces,
source baseline/feature identities, full-grid distribution/median components,
adapted query products and independent replay. Old results remain unchanged.
Previously seen station partitions remain development data; positive-K support
and inherited calendar-anchor adaptation are retrospective reconstruction.
