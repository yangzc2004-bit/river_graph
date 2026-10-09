# Research decision: concentration-relative source attention across regions

## Completed experiment

All25 fixed geographical packages are complete: HUC4 regions1013/1019/0708/
1030/1101, each with training seeds42–46. The current learned, current fixed-
prior and earlier-season relative-value arms comprise75 neural fits. All250
double-held-fold forest references were reused; no new forest was fitted.
K0 receiving stations provide no DOC, pH or conductance, including historical
values or their derived visibility features.

The proposed change expresses individual source departures in dimensionless,
season-centered OOF log1p units, then scales the transferred state by one plus
the frozen receiving environmental reference. Native-concentration loss,
local GRU readout, candidate pool, initialization,37,900 parameters and the
30-epoch/patience5 budget remain unchanged. This tests the units in which source
experience should transfer, within the existing model.

These are already evaluated ST357 geographical roles. The experiment is a
retrospective geographical replication, not independent external-basin
validation. All fixed arms and regions were completed without choosing a
different model for a region or K.

## Main result

The primary K0 population includes all valid DOC cells in each test region.
Seeds are averaged within region and the five regions are weighted equally.
Intervals use5,000 paired station resamples; all months and procedure outputs
for a station stay together. Station-equal MAE is a separate estimand.

| Procedure | MAE, mg/L | Station-equal MAE | Q90 MAE, mg/L |
|---|---:|---:|---:|
| Relative current-source complete |2.258278|2.486052|10.409544|
| Retained complete |2.282266|2.504735|10.519953|
| Native-value current attention complete |2.266609|2.493343|10.422181|
| Relative fixed-prior complete |2.268010|2.491390|10.451457|
| Relative earlier-season complete |2.276485|2.498913|10.501850|
| Preceding aggregate-source complete |2.272858|2.498975|10.471173|
| Strong station-hidden trees |2.344885|2.547670|10.520142|
| Trees with aggregate source features |2.314529|2.516639|10.481605|
| Older complete model |2.360283|2.556920|10.632267|

The complete procedure improves the actual retained model by **1.051%
[0.361%,1.754%]**, with four of five region averages improving. Q90 MAE
improves **1.050% [0.542%,1.602%]**, with all five regions improving.
Station-equal MAE improves0.746%;66 of104 station averages improve and38
worsen. This is a reproducible overall and tail increment in these roles.

The isolated increment over the immediately preceding native-value attention
procedure is **0.368% [0.151%,0.566%]**, with four positive regions. Its
Q90 increment is0.121% [-0.070%,0.281%], also four positive directions;
the additional relative-value change does not establish a new tail gain over
that preceding attention model.

Gain over bare strong trees is3.693% [0.722%,6.506%], positive in all five
regions. Gain over aggregate-source trees is2.430% [-0.854%,5.510%], positive
in four. Those trees share the source information origin but do not have the
attention branch's individual hydrological keys; this is not an identical-raw-
input architecture contrast. The4.322% gain over the older complete model
includes previous model improvements. Neither comparison is the isolated
relative-value increment. The5% working improvement target remains unmet.

## Learned allocation and source timing

The complete current-source model improves its matched fixed-prior arm by
0.429% [0.112%,0.725%], with three positive regions, and its earlier-season
arm by0.800% [0.092%,1.507%], with four. Complete Q90 gains are0.401%
[0.187%,0.644%] over fixed prior, positive in all five regions, and0.879%
[0.352%,1.431%] over earlier-season, positive in four.

Native neural-only MAE is2.259501, versus retained neural-only2.265595:
0.269% [-0.176%,0.620%], with three positive regions. Its Q90 gain over
retained neural-only is1.030% [0.573%,1.440%], positive in all five.
Relative neural-only improves native-value neural-only by0.273%
[0.052%,0.474%], four positive regions.

The overall difference between relative learned and fixed-prior neural-only
is not established: -0.006% [-0.247%,0.228%], two positive regions.
Their Q90 difference supports a0.436% [0.212%,0.683%] gain in all five.
Thus the complete overall allocation benefit includes source-validation-
selected ecological-memory fusion. It should not be assigned entirely to
the attention operator. Both native and complete results remain in the report.

## Regional behaviour and local adaptation

| Held-out HUC4 | Relative current MAE | Retained MAE | Gain |
|---|---:|---:|---:|
|1013|5.586720|5.626815|0.713%|
|1019|1.637584|1.619163|-1.138%|
|0708|1.728603|1.745890|0.990%|
|1030|1.249417|1.260627|0.889%|
|1101|1.089066|1.158835|6.021%|

Region1101 contributes the largest relative improvement. Region1019 still
worsens relative to retained complete;0708 improves over retained but slightly
worsens versus native-value attention. All results are retained. Overall signed
bias is-0.655575mg/L, compared with-0.649449 for retained complete and
-0.665207 for native-value attention. Reduced MAE does not eliminate the
remaining concentration bias.

The K curve uses identical query cells at every K, reserving all five support
candidates even at K0; its K0 differs from the all-observed primary population.

| K | Relative current complete MAE | Retained complete MAE | Gain |
|---|---:|---:|---:|
|0|2.271570|2.296495|1.085%|
|1|2.148383|2.160459|0.559%|
|3|1.966872|1.987946|1.060%|
|5|1.887012|1.909790|1.193%|

K0/K3/K5 intervals are[0.377%,1.795%],[0.510%,1.500%] and
[0.451%,1.780%]. K1's[-0.893%,1.665%] crosses zero. Every K has
positive Q90 gain with a supported paired interval. Aggregate-source trees
still have lower K3/K5 point estimates1.953828/1.871110; the candidate is
not replaced with a per-K tree winner.

Q90 thresholds are source-derived. Region1013 has904 unique tail cells and
mean recall0.9861. Regions0708/1019 have66/43 tail cells and zero recall;
1030/1101 have15/11 cells, zero recall and unstable flags. Repeated seeds
do not increase these ecological sample counts. Lower tail MAE is distinct
from reliable extreme-event detection.

## Information allocation and environmental strata

Mean zero-innovation prior mass is0.514 for learned current relative attention,
0.761 for fixed prior and0.697 for earlier-season attention. Mean per-head
Shannon entropy is0.659,0.659 and0.623 in natural-log units, respectively;
it is not divided by the number of valid candidates. This is allocation among
ecologically similar source stations, not measured river transport or causal
effects. The candidate pool and true river edges remain different relations.

Source-support, hydrological availability, ecological novelty, source distance
and receiving-reference concentration summaries are saved with contributing-
region counts. The no-current-donor stratum also improves, while the zero-
hydrological-channel stratum remains worse than retained. Stratified point
estimates describe where the complete procedure works; they do not isolate
the donor effect or select new region-specific adjustments.

## Verification and next research

All25 source libraries, double-held exclusions, candidate arrays, initial
weights, predictions, ecological-memory components, attention diagnostics and
support products replay bitwise. All primary, fixed-query and stratified tables,
5,000 station intervals and vector figures are complete. The actual PNG was
viewed and its labels, legend, intervals and region ordering were checked.
The fitted version passed988 tests with two skips; after adding the next
source-only mechanism the workspace passes992 with two skips. Ruff and the
historical artifact audit pass. Executed source copies and old products remain.

Retain this relative-value procedure as the strongest tested complete candidate
in this geographical series. Keep its modest incremental magnitude visible.
The current portable release and previously evaluated external predictions
remain the deployed references during further development.

The next source experiment, `doc_ratio_source_attention_v1`, was specified
before opening these geographical numerical results. It changes only the
relative source branch from first-order native conversion to the exact inverse:
`(1+B)*expm1(u)`, retaining the native local readout, loss and parameter count.
Its zero correction and derivative match the old model. Development again
uses142/143/144 ×42/43/44, with matched fixed/earlier-season controls and all
first-order comparators. One technical package has replayed and all nine source
packages are proceeding. No geographical or external concentration selects
the new correction, and no new forest is fitted.
