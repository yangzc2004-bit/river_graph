# Research decision: source water-quality supervision

## Result

All nine source packages (station partitions142/143/144, seeds42/43/44)
completed18 auxiliary pretraining fits and18 ordinary DOC residual fits.
The source chemistry labels supervise the existing ecological encoder and GRU;
receiving sites still have no DOC, pH or conductance inputs. The matched shuffle
arm has the same30 auxiliary epochs and subsequent DOC training settings.

| Complete procedure | Source K0 MAE, mg/L | Q90 MAE, mg/L |
|---|---:|---:|
| Retained model |1.769053|9.075420|
| Source pH/EC supervision |1.788135|9.112441|
| Source-label shuffle |1.769833|9.060217|
| Strong station-hidden trees |1.866362|9.394458|

The chemistry-supervised complete model regresses1.079% against the actual
retained model: paired gain CI[-1.886%,-0.331%], zero of three partition
means improving. It also regresses1.034% against the matched shuffle control
[-1.878%,-0.276%]. Its Q90 gain is-0.408%[-1.101%,0.012%] versus retained
and-0.576%[-1.411%,-0.096%] versus shuffle. The native-only variant also has
no established improvement. All intervals use5,000 paired station draws,
seed means within partition and equal partition weight.

It remains4.19% better than strong trees and2.27% better than the preceding
complete model, but those advantages do not establish an upgrade over the
retained model. Retain the current portable release and stop this sequential
chemistry-pretraining variant; do not run a geographical matrix for it.

## What the auxiliary task learned

True supervision improves internal source-fold standardized pH/log-conductance
MSE in every package, with best epochs11–30. The GRU parameter distance is
3.89–6.74, compared with0–0.95 for the shuffled control. Auxiliary prediction
is therefore learnable, but its learned representation does not improve DOC
residual reconstruction in this training design. This is evidence about the
sequential pretraining procedure, not a claim that chemistry is unrelated to DOC
or that no form of shared supervision can work. No receiving chemistry is used.

## Next model mechanism

Return representation learning to the DOC task. Test whether the existing GRU
can use the environmental tree's causal concentration trajectory, rather than
seeing that reference only through the current-month residual head. Compare a
current-reference projection with a matched full-history projection, keeping
the same tree, original ecological/GRU initialization, native DOC objective and
memory fusion. The source trajectories require dense station-blocked OOF
prediction: the old cache contains predictions only at observed source DOC
cells, and must not be filled with in-sample tree predictions or future labels.
Develop the new version on the same source roles; old geography and external
results remain outside development.

## Reproduction and inspection

Source-only scaler/count recomputation, source/receiving station exclusion and
internal auxiliary checkpoint selection passed for all18 fits. All native and
complete products replay bitwise from saved states; original predictions and
DOC input features are unchanged, and DOC initial weights equal their selected
warm backbones. Three new tests verify receiving-label isolation, held-source
normalization, label shuffling, warm-weight continuation, saved replay and
causal input windows. The full suite passed950 tests, two skipped; Ruff and
historical artifact audit passed. The four-panel scientific figure was actually
viewed and has no clipped labels or overlapping data annotations. Large fits
remain local; the related pending-file list is saved because Git writes are
restricted by the current filesystem policy.
