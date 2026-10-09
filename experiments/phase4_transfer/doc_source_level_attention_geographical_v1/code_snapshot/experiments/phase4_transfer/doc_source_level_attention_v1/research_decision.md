# Research decision: preserve source seasonal bias during DOC transfer

## Completed source experiment

All nine source packages142/143/144 ×42/43/44 are complete:36 neural fits,
90 reused double-held-fold references and no new forest fits. The four new
arms use full current relative residual, full earlier-season residual, full
fixed-prior allocation and seasonal mean alone. The old first-order anomaly-
only procedure and every retained comparator remain in the prediction panel.

Only individual donor values change. The same37,900-parameter ecology/GRU/
attention model, native objective, initialization,30epochs/patience5,
candidate availability, source normalization and ecological-memory fusion are
retained. Seasonal means are from allowed donor stations; query fold A is
excluded and each donor B's reference omits both A and B. K0 receivers have
no DOC/pH/conductance. This is source-validation development.

## Overall performance

| Complete procedure | MAE, mg/L | Station-equal MAE | Q90 MAE, mg/L |
|---|---:|---:|---:|
| Full current relative source residual |1.736968|2.190777|8.927563|
| First-order anomaly-only source |1.748102|2.205626|8.932044|
| Full current fixed prior |1.742909|—|8.948794|
| Full earlier-season source |1.763675|—|9.061016|
| Seasonal individual values only |1.747950|2.204194|8.958080|
| Retained complete |1.769053|2.232407|9.075420|
| Strong station-hidden trees |1.866362|—|9.394458|

With5,000 paired station draws and equal source-partition weights after seed
means, full current source values improve retained complete by **1.814%
[1.031%,2.750%]**, positive in all nine packages, all three partitions and
all three training seeds. Its Q90 gain is1.629% [0.536%,3.072%], with
eight positive packages and all three positive partitions.

The isolated increment over the anomaly-only complete procedure is
**0.637% [0.084%,1.284%]**, with eight positive packages, two positive
partitions and three positive seed averages. Native-only improves its
anomaly-only counterpart by0.659% [0.033%,1.384%], with the same directions.
This improves the strongest preceding source candidate through information
preservation, without enlarging the neural architecture.

Q90 changes relative to anomaly-only are0.050% [-0.662%,0.736%] complete
and0.043% [-0.751%,0.840%] native-only. Do not describe this mechanism as
a new established tail improvement. Its benefit is in overall reconstruction.

Complete gain over strong trees is6.933% [4.276%,9.950%]. This includes
the retained model's existing ability; it is not the seasonal-value increment.
Station-equal MAE improves1.865% over retained and0.673% over anomaly-only.

## Matched information controls

Full current complete improves the earlier-season arm by1.514%
[0.852%,2.297%], positive in all packages and partitions. The gain over the
seasonal-only arm is0.628% [0.256%,1.034%], positive in two partitions;
its Q90 gain is0.341% [0.031%,0.830%]. Native-only overall gain over
seasonal-only is0.677% [0.266%,1.139%]. Preserving seasonal information
and using current individual departures both contribute to the measured
complete comparison.

Seasonal-only changes individual donor values but retains the original current
aggregate readout features. It is not a control removing all contemporaneous
source information. The head and ecological-memory component are fitted jointly
with their respective inputs; these contrasts are method comparisons, not
causal error partitions.

Against full fixed-prior complete, gain is0.341% [-0.101%,0.869%], despite
three positive partition averages. Native-only gain is0.397%
[-0.074%,0.959%]. Adaptive allocation is not established on overall source
validation in this experiment. Keep that matched control in confirmation.

| Source partition | Full-current MAE | Anomaly-only MAE | Retained MAE |
|---|---:|---:|---:|
|142|1.841283|1.840147|1.877208|
|143|1.637581|1.657299|1.665676|
|144|1.732039|1.746861|1.764274|

Partition142 is slightly worse than anomaly-only;143/144 provide the isolated
improvement. All three improve retained. Complete signed bias is-0.620238mg/L,
versus anomaly-only-0.605140 and retained-0.638608. Improved overall MAE
does not guarantee a smaller signed concentration bias in every comparison.

## Allocation, verification and next experiment

Mean zero-source prior mass is0.451 for full current,0.662 fixed,
0.609 historical and0.472 seasonal-only. Mean per-head Shannon entropy is
0.804/0.897/0.778/0.744 in natural-log units. Matched current donor support
covers84.2% of source-validation cells, with mean3.23 donors. These are
ecological-source information weights, not river transport coefficients.

All fitted candidates, source exclusions, initial weights, neural/memory
predictions and diagnostics replay bitwise. Reconstruction of full residuals,
unchanged availability, future-input isolation, zero head, new node count and
save/reload contracts pass.995 tests pass with two skips; Ruff and historical
audit pass. The actual PNG was inspected. Existing deployment, already scored
external cases and original geographical products remain.

Carry the fixed full-current procedure and all three matched controls to the
five established HUC4 roles, each at seeds42–46. Include the immediately
preceding anomaly-only geographical model, actual retained complete and strong
trees in all-observed K0, native ablation, fixed-query K0/1/3/5, Q90, bias,
station-equal and environmental-stratum comparisons. No different region or
K winner is selected. Reuse the250 existing pair references; fit no new forest.
These geographical roles have been evaluated before, so the next experiment
is retrospective ST357 replication, not independent external-basin validation.
Keep this source decision fixed during confirmation.
