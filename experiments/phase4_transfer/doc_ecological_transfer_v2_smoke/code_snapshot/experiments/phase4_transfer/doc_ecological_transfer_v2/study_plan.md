# Support-aware update of the ecological residual prior

The v1 ecological affine procedure improves reconstruction without target-station
observations. Its fixed K0-selected mixture does not improve the five-support
product. This integration is designed after those v1 results and keeps the v1
profiles and products as the fixed-mixture comparison.

Freeze every source ecological profile, donor selection, ridge, ecological
normalizer, forest, recurrent expert and temporal support basis. Do not refit
source residual memories. The only new choice is how much regional memory to
retain after target support becomes available.

For each of four residual families and both support adapters, retain gamma
{0,0.25,0.5,1}. At K0 lock gamma to v1 and require bitwise equal output. At
K1/K3/K5 select gamma jointly with the existing support-adapter alpha/ridge
using adapted native query MAE on the same fixed source-validation episodes.
For each gamma, both support and query predictions must come from that same
context/temporal/memory mixture. Gamma zero reproduces the interaction model
and its existing support adapter. Ties prefer zero, the prior gamma, then a
smaller gamma. No target query label enters either selection or adaptation.

The interpretation is a learned update of a regional prior with local
observations. It does not imply that the regional prior should always decline
monotonically with K; report the validation-selected weights at every K.
The original K0 gain is carried over, not another independent confirmation.

Use all nine existing station-partition/seed packages. Report all four modes,
two support heads and K0/1/3/5. Primary comparisons are support-aware versus
fixed mixing at K5, ecology versus interaction at K1/K3/K5, and matched affine
versus bias/global controls. Report MAE, RMSE, R2, Q90, ordinary error, bias,
false Q90 rates, station directions and 5,000 paired whole-station bootstrap
draws with the existing partition-equal estimator.

Do not select a model, K or profile family from outer query scores. Preserve
both versions. These remain same-cohort development experiments with
retrospective target support.
