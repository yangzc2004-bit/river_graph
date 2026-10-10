# Conditional station-held readout errors for river learning

## Research question and interpretation

Does learning the errors of a scalar readout that did not fit the receiving
station improve the existing upstream environmental-state correction? The
preceding absolute/contrast state variants did not establish a sizeable river
advantage. This experiment tests a training-target explanation before adding
more propagation parameters.

This is **conditional readout cross-fitting**, not complete-model OOF. The
existing encoder, GRU, donor attention allocation/values, ecological coefficients
and observed-DOC branch retain their historical source fits, including source
labels from the held readout folds. Queries retain their original complete-model
features too. Only the freshly fitted scalar head's LOSS excludes the whole
receiving fold. A negative or positive result cannot identify a full station-OOF
model's performance or prove the cause of previous river failures.

## Controlled change

Export exactly the retained native readout's hidden, extra, interaction and donor
state design. Refit this scalar readout with fixed encoder and fixed ecological
offset/scaling: either all source rows, or four of five station folds. Both use
zero initialization, Adam .001, batch512, source-Q90 weight2 and **fixed30 epochs**;
neither held-fold nor validation labels select these readout fits. Memory and
observed-river corrections remain frozen. Both refitted variants are TRAINING
anchors only. Validation inference always starts from the identical stored
complete model plus observed-DOC river correction.

The primary comparison is crossfit_state versus refitted_state. This controls
for the effects of refitting the readout and using a frozen representation.
Comparison against expanded_upstream measures the actual gain over the preceding
river version. Do not attribute differences from a refit control alone to a
gain over the current model.

## Arms, roles and scope

ST357 source-development roles142/143/144 x seeds42/43/44. No receiving DOC,
pH or conductance inputs; no geographic/external role used to tune this version.
54 scalar-readout fits and45 river fits:

- refitted_state: all-source readout errors, real upstream states;
- crossfit_state: station-held readout errors, same real upstream states;
- crossfit_uniform: same crossfit errors, uniform allocation;
- crossfit_matched_upstream / crossfit_matched_nonupstream: common candidate/lag
  support with non-ancestor identities matched as in the preceding version.

All river operators retain the exact absolute-state banks, paths, queries,
normalization,2heads x32, lags0/1/3 and30epoch/patience5 budget. Epoch0 remains a
candidate. Preserve complete model, strong trees, observed-DOC, expanded dynamic,
uniform and matched arms. The covariate-known atlas remains transductive;
conditional folds do not turn this into independent or external evaluation.

## Analysis and scientific completion

Report source anchor error distributions before/after readout crossfit and
validation-anchor distributions, native/log errors and Q90. Do not assume source
errors are universally smaller: the current source-Q90 errors exceed validation
in partitions142/143, while partition144 has a much larger validation tail.
Report all nine packages, station-equal losses, coverage/path strata, and5000
paired whole-station draws, seed losses averaged first, then partitions equal.
Verify exact fixed products/banks, checkpoint replay and held-readout-label
perturbation conditional on the fixed feature snapshot. Inspect plots and record
whether to adopt the river head, conduct genuine complete-model fold refits, or
advance to an explicit confluence/storage operator. Preserve old experiments.

## Explicit repair from the partial v1 execution

v1 completed partition142 only; partition143 failed a replay assertion before
new fits because export omitted the local nonnegative clipping BEFORE memory
blending. Two of15,241 source rows differed (up to0.140793mg/L). Keep v1 code
snapshot, three packages and failure log unchanged. v2 restores the retained
two-stage clipping both in readout prediction AND readout fitting loss, then
runs all nine packages from scratch. No model or arm was selected from v1
validation scores; same predeclared arms and30epoch budget.
