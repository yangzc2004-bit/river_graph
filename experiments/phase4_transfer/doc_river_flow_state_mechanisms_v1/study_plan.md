# Explain observed changes in DOC buffering between water states

Date: 2026-10-08.

## Scientific question

At the same actual confluence, why does the outlet-to-branch-mixture DOC SD
ratio change between low and high flow? Separate changes in the incoming signals,
the branch water balance, and the remaining outlet departure before assigning
the ratio change to shared-corridor smoothing.

The preceding joint-calendar results are known. An initial diagnostic of those
saved summaries also shows that a rising ratio can coincide with decreasing
mixture SD. This is an exploratory explanation of existing observations, not
another independent confirmation or an intervention on river geometry.

## Fixed observations

Use the exact all-calendar and 47-date joint-calendar populations, four real
mapped arrangements, laboratory measurements, daily flow, source clocks and
hydro-only state thresholds from `doc_river_joint_campaigns_v1`. Preserve its
low/middle/high definitions and annual harmonic plus linear time projection.
All four configurations are reported; C16 and C7 remain the previously observed
positive high-minus-low ratio cases. Do not change sampling or river-form labels.

## Three explanatory calculations

1. **Separate numerator and denominator.** Decompose the high-minus-low change
   in log(outlet/mixture SD) into the change in log outlet SD minus the change
   in log mixture SD. Report absolute SD in mg C/L alongside ratios. This is an
   exact statistical decomposition, not a physical transport attribution.

2. **Separate branch coordination and contribution.** On the same adjusted
   dates, calculate fixed-weight mixture variance with each state's mean branch
   flow fraction. Describe its reduction relative to the share-weighted branch
   variances. Decompose the low-to-high change in this mixing potential using
   all-order Shapley contributions for three observed parameter groups: branch
   correlation, the two branch SDs, and mean water fraction. Also retain the
   preceding population-wide fixed weight as a separate sensitivity. The
   constructed parameter substitutions hold the other measured summaries fixed;
   they are not measured counterfactual floods or estimated causal effects.

3. **Separate a mixture departure from a measured channel effect.** For
   D=outlet−partial mixture, use the exact identity
   Var(outlet)=Var(mixture)+Var(D)+2Cov(mixture,D). Keep the departure variance and
   covariance visible; neither is a DOC removal coefficient. As a separate
   instantaneous mixing-feasibility diagnostic, on dates with 0<f≤0.95 for
   f=(Q_A+Q_B)/Q_outlet, compute the unmonitored-water concentration needed to
   reproduce outlet DOC without net processing. Retain negative solutions and
   out-of-range water shares in the audit. Feasibility demonstrates ambiguity,
   not that the unmonitored concentration was actually measured or that channel
   processing is absent. Daily flow is not a contemporaneous complete load.

## Repeatability and interpretation

Use 5,000 whole-year bootstrap draws, jointly across the common calendar, and
refit the same seasonal projection in every draw. The flow-state thresholds
remain the saved hydro-only references. Bootstrap intervals reflect temporal
repeatability at the actual configurations; nested sites do not become
independent river systems. Preserve unavailable estimates and valid-draw counts.

Save the complete state and high-minus-low summaries, all eight parameter
substitutions, departure identities and the no-processing feasibility ledger.
Plot the actual SD changes, incoming synchrony/contribution decomposition and
partial water budget; inspect the figures. Write an English research decision
that distinguishes measured signal changes from still-unidentified channel
operations. Connect the explanation to the same branch paths and common corridor.
No model is trained and no preceding result is overwritten.

This study has been made possible by data provided by the Swedish Infrastructure
for Ecosystem Science (SITES).
