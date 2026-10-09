# Current donor availability: source-development decision

## Result and decision

Opening permitted current source DOC observations without requiring an earlier
same-season observation improves the existing attention model. Carry this fixed
version into a separate five-region geographical replication. Retain the
complete procedure as the candidate and native-only as its ablation; do not
choose a different model for each region or support count.

All nine source-role packages (142/143/144 × 42/43/44) completed: 18 neural
fits, 90 reused double-held reference fits and no new forest. Every package,
candidate array, source exclusion, initial state, neural prediction, ecological
fusion and diagnostic replayed exactly. Analysis uses 5,000 paired station
draws, seeds averaged within partition and equal partition weights. The PNG
was inspected. These are source-validation development results.

## Matched complete-model comparison

| Procedure | K0 MAE, mg/L | Q90 MAE, mg/L |
|---|---:|---:|
| Retained complete | 1.769053 | 9.075420 |
| Matched anomaly-only source attention | 1.748102 | 8.932044 |
| Matched full source attention | 1.736968 | 8.927563 |
| Expanded current availability, seasonal values only | 1.742044 | 8.945071 |
| Expanded current availability, actual source values | **1.723002** | **8.895024** |
| Strong station-hidden trees | 1.866362 | 9.394458 |

The isolated availability increment versus matched full-source attention is
**0.804% [0.277%, 1.335%]**: 8/9 packages, all three partitions and all three
seed means improve. Its Q90 increment is 0.364% [-0.158%, 1.232%], so this
comparison does not establish a tail improvement. Against anomaly-only
attention, the complete-model gain is 1.436% [0.702%, 2.261%]; that comparison
also includes the seasonal source-value change.

Actual current source values beat individual seasonal means under the same
expanded validity by **1.093% [0.582%, 1.610%]**, all three partitions and
seed means positive. Q90 improves 0.560% [0.089%, 1.413%]. The seasonal
control retains the original matched aggregate source descriptors, so this
contrast isolates the individual donor values, not removal of every source
observation from the network.

The accumulated gain versus the retained complete model is
**2.603% [1.642%, 3.681%]**, all nine packages improving; Q90 improves
1.988% [0.850%, 3.544%]. Against strong station-hidden trees, MAE improves
7.681% [4.958%, 10.785%]. These accumulated gains are not the isolated
availability contribution. Station-equal MAE is 2.175422, a separate estimate
from the partition-weighted cell MAE above. Signed bias remains -0.607553 mg/L.

## Native-only check

The same availability change improves native-only MAE from 1.740125 to
1.726375: **0.790% [0.177%, 1.434%]**, all three partitions and seed means
positive. Q90 improves from 8.898214 to 8.840746, 0.646% [0.015%, 1.750%].
Thus the positive overall increment exists before ecological fusion. Do not
substitute this native-only tail result for the complete-model tail contrast.

## What changed scientifically

The 20 candidate identities, ecological prior, daily-flow keys, receiving GRU
query, original matched aggregate readout, 37,900 parameters and training
budget are unchanged. Only individual donor availability is expanded. Mean
usable donors rise from 3.883 to 5.017, 3.093 to 4.007 and 2.709 to 3.477 in
the three partitions. Fractions with at least one donor rise from
85.06/83.67/83.97% to 92.90/92.50/91.93%. There are 179/271/279 newly
supported station-month occurrences across partitions; repeating training
seeds does not create additional ecological observations.

This supports using current source experience when old source history is
missing. It does not establish that a current donor is a physical upstream
driver. The retrieval relationship is ecological similarity. Receiving K0
DOC, pH and conductivity remain absent. Query folds and donor reference folds
remain excluded; saved seasonal statistics use permitted source training only.

## Next experiment

Freeze the same candidate and expanded seasonal control for HUC4
1013/1019/0708/1030/1101 × seeds 42–46, reusing all earlier reference fits.
Carry matched full-source, anomaly-only, retained complete and tree predictions
unchanged. Report all-observation K0, a separate fixed-query K curve, Q90,
bias, station-equal scores, support/input-availability strata and 5,000 paired
station intervals. This is retrospective ST357 geographical replication,
not independent external-basin validation. The old portable release and scored
external products remain unchanged. A positive development result warrants
replication; it is not yet the geographic performance claim.

## Reproduction

Use `scripts/analyze_doc_current_availability_attention_v1.py
--bootstrap-draws 5000` (includes exact replay) followed by
`scripts/plot_doc_current_availability_attention_v1.py`. Fitting-version
validation: 1,002 tests passed, two skipped; Ruff and the historical artifact
audit passed. Large fitting caches stay local; the related-file whitelist
records deliverables pending the already documented Git write limitation.
