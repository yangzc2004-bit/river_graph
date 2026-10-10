# Research decision: confluence mixing and storage in DOC river messages

## Outcome

All nine source-development packages and 45 river fits are complete. Explicit
branch mixing yields a very small improvement over ordinary upstream attention;
adding storage buffering does not increase it. Keep the preceding upstream-state
candidate and released model. Do not replace either with this new operator.

This round advances the existing GNN river branch: fixed local ecology/hydro/GRU
prediction plus a learned correction from actual directed upstream paths. It
changes propagation rather than increasing backbone depth. It provides a
reproducible mechanism comparison, not a new confirmed performance record.

## Method

Operate on source DOC departures from separately station-blocked environmental
predictions. Along a nested branch, retain the closest usable donor for each lag;
an older ancestor is a backup when the closer station has no usable observation.
Independent represented branches can mix. Normalize measured nonnegative monthly
NWIS discharge when every retained source is measured and the group sum is
positive. Otherwise use drainage area for the entire group. Reversing or all-zero
flow invokes the same explicit fallback. Never normalize mixed discharge/area
units or count nested catchments as independent inflows.

The storage arm adds positive distance/confluence/storage attenuation and a
storage-dependent older-memory bias. All encoders retain physical path features;
this tests their explicit use in the operator, not their mere availability.
Real/nonancestor matched controls copy the same real-slot branch frontier,
flow/area prior, path, ages and common observation mask. Only source identity,
source innovation and source hydro descriptors differ.

The quantities being mixed are signed log1p innovations. This is an
information-propagation prior, not concentration/load conservation. Monthly
lags 0/1/3 are memory slots, not measured river travel times. Missing tributaries,
intervening sources and sinks are not represented as observed fluxes.

## Fixed evaluation

ST357 source-development station roles 142/143/144 x seeds 42/43/44. Receiving
DOC/pH/conductance are absent. Source query folds are excluded from donor labels
and their double-held environmental tree references. The complete base's own
source predictions remain fitted rather than full-model OOF. No new forests,
backbone or conditional scalar readouts were fitted. All arms use 30epochs,
patience 5,2 heads x 32, source-Q90 weight 2 and an epoch 0 zero-correction candidate.

The new branch replaces ordinary observed-DOC messages. It does not add the
same upstream innovations twice. The environmental-state branch is a retained
comparator, not part of the new inference anchor. Queries are identical for all
models:140 unique receiving stations,7,897 unique station-months and8,857
partition-cell occurrences. Seeds repeatedly predict these same ecological
samples. Metrics average seed losses within each partition and then weight the
three partitions equally. Intervals use 5,000 paired whole-station draws jointly
across partitions. Development/early-stopping selection optimism remains.
These are neither geographic confirmation nor external-basin validation.

## Complete comparison

| Model | MAE mg/L | Q90 MAE mg/L | log1p MAE |
|---|---:|---:|---:|
| Complete current model |1.723002|8.895024|0.248207|
| Strong environmental trees |1.866362|9.394458|0.275922|
| Previous ordinary observed-DOC branch |1.717226|8.852561|0.247090|
| Previous environmental-state branch |1.714729|8.841157|0.246694|
| Plain upstream attention, replay control |1.717226|8.852561|0.247090|
| Confluence mixing |1.716458|8.853940|0.246987|
| Confluence mixing + storage, primary |1.716517|8.855133|0.246977|
| Matched real upstream + storage |1.718084|8.868784|0.247295|
| Matched non-upstream + storage |1.722024|8.892089|0.248103|

Plain attention reproduces the retained dynamic branch. Every arm is reported;
the mixing-only arm is not retrospectively renamed the primary model.

| Comparison | Native MAE reduction % [95% station CI] | Positive packages | Positive partitions |
|---|---:|---:|---:|
| Storage versus plain |0.041267 [-0.057871,0.156336]|6/9|3/3|
| Storage versus mixing |−0.003432 [-0.055448,0.045775]|7/9|2/3|
| Storage versus previous environmental-state branch |−0.104301 [-0.257014,0.045573]|3/9|0/3|
| Storage versus complete current model |0.376385 [0.042714,0.707061]|9/9|3/3|
| Mixing versus plain |0.044697 [-0.063075,0.172537]|5/9|3/3|
| Matched real versus non-upstream |0.228815 [-0.061811,0.518141]|6/9|2/3|

The0.376% accumulated improvement over the complete model is not the effect of
adding storage: ordinary upstream messages already account for most of it. The
new operator's actual increment is0.041%, with an interval crossing zero. Its
Q90 error increases0.029% versus plain and0.158% versus the preceding
state branch; both intervals cross zero. Storage versus mixing also slightly
increases Q90 error. More positive packages can coexist with a negative mean
because package effects have different magnitudes.

## What the available river data let the operator learn

| Partition | Usable upstream DOC query cells | Multiple represented branches | Nested candidate–lag slots removed | Measured-flow share of retained frontier slots |
|---|---:|---:|---:|---:|
|142|22.733%|8.235%|38.805%|50.104%|
|143|21.741%|7.432%|40.674%|68.880%|
|144|52.481%|29.122%|39.709%|29.537%|

Percentages in the first two columns use all query cells as denominator. The
last column counts valid retained source–lag slots, not receiving months or
attention mass. These fractions repeat across training seeds. Fifty unique
receiving stations have some usable upstream observations. Unsupported cells
retain the original base exactly, including stations with both supported and
unsupported months. The nonredundant frontier removes about40% of otherwise
valid slots without reducing whether a query has any upstream support.

Among supported rows, the storage arm's mean allocation is40.915% current,
24.808% previous-month,18.881% three-month and 15.396% zero-message. Its weighted
attenuation is 0.8702. Despite an explicit older-memory bias, the learned net
allocation moves toward the current month relative to mixing-only. These
weights are model diagnostics, not a measured storage response.

Descriptive storage-versus-plain gains: observed-support rows 0.162%, multiple
represented branches 0.361%, mostly-area-proxy rows 1.281%, and mapped storage
above 0.1 rows 0.532%. Every corresponding station interval crosses zero. The
storage subgroup has only 15 unique receiving stations. Do not turn these
selected, overlapping groups into separate positive mechanism claims.

## Scientific interpretation and next work

Removing nested duplicates and supplying a mixture prior produces a weak,
consistent partition-average direction. The current data do not establish an
additional benefit of buffering, nor a real-upstream advantage over the matched
nonancestor operator. They also do not show that river mixing or retention is
unimportant in nature: observed support, sampling alignment and monthly
aggregation limit what this particular reconstruction task can identify.

The most useful next mechanism work is observation-time alignment: audit actual
source DOC sampling dates against daily hydro changes, then represent whether
an upstream innovation belongs to the receiver's hydro event or to an older,
unrelated event. Develop this on source roles with fixed ordinary/complete
comparators, before another depth/head/attenuation sweep. Preserve the monthly
reconstruction task and receiving-label isolation; use finer timing as source
message metadata. A subsequent complete-model station-held training comparison
would require actual component refits, not the already negative conditional
scalar-readout experiment. This round has not diagnosed either factor as the
unique cause of weak river gains.

## Verification and products

All 45 checkpoints replay bitwise; all nine mixing-prior caches reconstruct
exactly; all fixed products are bitwise unchanged; matched priors/availability
are identical. Independent calculations verify every model's MAE/log/Q90 and 15
primary station intervals. The full suite passes 1,474 tests, with 3 documented
skips; Ruff and historical artifact verification pass. Two PNG/PDF/SVG figures
were inspected; legends overlapping stacked bars were moved outside the plot.
The initial analysis object-dtype boolean mask after panel concatenation was
corrected before reporting. Neither repair changed training or model selection.
Large local fitting caches/checkpoints are retained separately from committed
small predictions, configuration, execution snapshot and analyses.

Reproduce via uv-managed Python:

```bash
uv run python scripts/verify_doc_confluence_storage_v1.py --rebuild-inputs
uv run python scripts/analyze_doc_confluence_storage_v1.py --bootstrap-draws 5000
uv run python scripts/verify_analysis_doc_confluence_storage_v1.py
uv run python scripts/plot_doc_confluence_storage_v1.py
```
