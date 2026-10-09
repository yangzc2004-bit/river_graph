# Nonlinear DOC readout: five-region replication decision

## Result and retained model

All25 fixed packages are complete: five HUC4 regions × seeds42–46. Keep the
delivered linear-head complete model. The nonlinear head's small source signal
does not establish a useful additional geographical gain.

| Procedure | Region-equal K0 MAE, mg/L | Station-equal MAE, mg/L | Q90 MAE, mg/L |
|---|---:|---:|---:|
| Retained linear complete |2.282266|2.504735|10.519953|
| Nonlinear complete |2.278136|2.504015|10.544403|
| Retained linear neural-only |2.265595|2.485495|10.494278|
| Nonlinear neural-only |2.277652|2.504087|10.531189|
| Strong station-hidden trees |2.344885|2.547670|10.520142|
| Preceding complete |2.360283|2.556920|10.632267|

Incremental complete MAE reduction is0.181% [-1.320%,1.630%], with three
of five positive region directions. Matched neural-only gain is -0.532%
[-1.923%,0.714%], with two of five positive directions. Complete Q90 gain
is -0.232% [-0.741%,0.367%]. Station-equal MAE is nearly unchanged and
aggregate bias becomes more negative (-0.674033 vs -0.649449 mg/L).

The new complete procedure improves over the older complete model by3.480%
[1.093%,6.023%], and over strong trees by2.847% [-0.843%,6.243%]. These
include the previously established residual-model gains. They do not show that
the nonlinear head provides a significant new increment, and the5% working
goal against both principal comparators remains unmet.

## Adaptation and regional consistency

| Fixed-query K | Retained complete MAE | Nonlinear complete MAE | Strong tree MAE |
|---|---:|---:|---:|
|0|2.296495|2.291941|2.359473|
|1|2.160459|2.165389|2.181676|
|3|1.987946|2.000691|1.989754|
|5|1.909790|1.917713|1.905463|

K curves use an identical query with all five candidate support cells reserved.
The nonlinear complete model remains slightly worse at K1/3/5. It improves in
1013,1030,1101 and worsens in1019,0708 at primary K0. Every region and K
remains in the analysis; no model winner is selected for a region or K.
The source-fixed full procedure remains the object of comparison. The better
observed neural-only score is reported as an ablation rather than substituted
post hoc into this complete-model evaluation.

## Implementation and evidence

Each parent region/seed supplies its already saved source-only backbone, strong
tree, nested station OOF reference and input normalization. The nonlinear fit
uses the linear candidate's initial backbone states, verified bitwise, with
the same30-epoch budget and source-validation selection. The added32-unit
head, objective and all dimensions were fixed from source development before
these geographical queries were scored. Ecological memory and simple support
adaptation retain their original validation-selected families.

All parent primary predictions and K curves remain exactly unchanged. New
complete prediction and support products replay bitwise; neural replay from
the subset input uses the existing1e-6 tolerance for different inference batch
composition. The complete source exclusion checks reside in verified parent
stages, which were checked before reuse. Label-free point/state components were
saved before scoring and opening prescribed test support.

The analyzer initially imported the task-directory constant through a runner
whose unused import had been removed by lint. Importing it directly from its
defining module repaired analysis without changing fitting or predictions.
The failed log, execution-time source copy and `analysis_repair.json` are saved.

`scripts/analyze_doc_nonlinear_geographical_replication_v1.py
--bootstrap-draws 5000` verifies all25 packages and produces region/seed
metrics, station-equal error, Q90 bias/recall, fixed-query curves and
source-defined ecological/hydro/similarity strata.5,000 paired station draws
preserve each station's months; seeds are averaged within region and regions
receive equal weight. The plotted vector/PDF/PNG comparison was inspected.
The full suite passed938 tests with two skips; Ruff and historical audit passed.

This is retrospective replication on already evaluated ST357 geographical
tasks, not a new independent holdout or external basin. External predictions,
the portable release and manuscript numerical claims remain unchanged.

## Next research direction

Output-unit changes, global calibration, region-hidden training and a larger
readout have now been tested. Their additional benefit over the retained model
is absent or small. Investigate predictive information before another size
increase. The local StreamCat table contains distinct forest, wetland, farming
and urban composition fields that the current dataset aggregates into broad
totals. Audit those label-independent fields and their COMID alignment, then
test detailed composition using source roles only. Select all predefined
categories rather than choosing fields from these geographical outcomes.
