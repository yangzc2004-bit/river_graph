# Upstream environmental-state messages for DOC reconstruction

## Question and architecture

The preceding observed-river branch improved native DOC MAE by 0.34% overall
and 1.25% where upstream DOC was usable. It could do nothing where DOC support
was absent. This experiment asks whether upstream environmental dynamics can
add information in those gaps. It stays on the same complete DOC model.

Freeze the environmental tree base, ecology/self encoder, observation-aware
GRU, source-similarity attention, ecological integration and the preceding
observed-river branch. Reuse the encoder/GRU to encode every source station's
12-month environmental history. Remove all DOC values, masks, history counts,
upstream/downstream observation support and contextual target predictions from
this encoding. No receiving DOC/pH/conductance input is opened. Source chemistry
has historically supervised the retained weights; that is learned source
experience, not receiving chemistry input or a fully OOF neural representation.

State values contain GRU64, ecology9, daily descriptors8, their preceding-month
change8 and monthly hydro/masks4. At least one measured hydro input within the
causal twelve-month window makes a state usable. Missing DOC does not invalidate
it. Use at most twenty actual upstream stations over mapped paths <=3000 km,
lag slots 0/1/3, the preceding 90-dimensional query, two heads and dimensions32.
A learned value projection and bias-free zero-start output send state messages.
No eligible state returns the anchor exactly. Calendar lags denote information
history rather than physical travel times. Current-month hydro is retrospective
reconstruction information, not a forecast before that month has ended.

## Fixed comparison before results

Partitions142/143/144 x seeds42/43/44; all source-validation DOC cells, K0.
Retain unchanged complete model, environmental trees and observed-river results.
Five newly fitted arms per package (45 neural fits):

1. State upstream: state correction on the complete anchor.
2. Observed + state (primary): additional state correction on the frozen
   observed-river anchor. Two corrections are added sequentially in log1p space;
   no cell/region/seed winner is chosen.
3. Uniform upstream state: same allocated state network with uniform allocation
   and unit reliability, on the complete anchor.
4. Matched state upstream, on the observed-river anchor.
5. Matched state non-upstream, on that same observed-river anchor.

Training banks exclude the receiving fold. Control sources exclude complete
physical ancestry (including paths beyond the primary cap) and COMID aliases.
Drainage area and label-free hydro availability determine controls. Matched arms
retain identical real path slots and intersect hydro validity, using the same
state operator and observed anchor. DOC values never determine state matching.
The primary unrestricted arm tests performance; matched arms isolate real
connectivity versus environmental similarity under common hydro support.

Adam .001, batch512, at most30 epochs, patience5, source-Q90 tail weight2.
Source-validation native MAE selects the epoch, including the zero-start option.
Complete and observed-anchor training outputs are fitted, not full-model OOF.
Old donor DOC residuals retain their double-held environmental OOF provenance.
No old geographical or external labels are used. These source-development
results require subsequent independent confirmation if the candidate is useful.

## Evaluation and interpretation

MAE, RMSE, R2, log1p MAE, source-Q90 error and bias. Seed means within partition,
then equal three partitions; 5,000 paired whole-station bootstrap draws jointly
across partitions. Compare the primary with BOTH complete and observed anchors.
Report every arm, station-equal error, per-package/partition directions, observed
DOC support versus state-only support, path bands, correction and lag diagnostics.
Separately report no observed DOC / no usable environmental state: silence of
one channel does not imply silence of the other. State-only messages are latent
environmental representations, never fabricated DOC observations.

Rebuild states and matched banks, replay checkpoints, check hidden-label/future
contracts and inspect figures before a research decision. All historical models
and results remain unchanged. No post-result architecture or endpoint changes
within this version.
