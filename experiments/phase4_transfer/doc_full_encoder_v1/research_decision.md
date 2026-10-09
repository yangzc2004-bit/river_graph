# Research decision: adapting the whole local self encoder

## Results

Nine source-validation fits (142/143/144 ×42/43/44) are complete. Both
existing self layers and the ecological encoder were updated at1e-5, instead
of updating only the last self layer and ecology. Parameter allocation, initial
weights,12-month GRU, reference/OOF arrays, inputs, native tail objective and
30-epoch/patience5 budget were retained. The local expert uses empty edges.

| Procedure | K0 MAE (mg/L) | Q90 MAE (mg/L) | Signed bias (mg/L) |
|---|---:|---:|---:|
| Retained complete model |1.769053|9.075420|-0.638608|
| Full-encoder complete |1.769569|9.073926|-0.633568|
| Retained neural-only model |1.772035|9.053055|-0.625464|
| Full-encoder neural-only |1.772145|9.042726|-0.620907|
| Strong station-hidden trees |1.866362|9.394458|-0.611014|

Complete-model gain over the actual retained complete model is-0.029%
[-0.302%,0.245%], with one of three partition averages and four of nine
packages improving. Matched neural-only gain is-0.006% [-0.380%,0.348%].
Complete Q90 gain is0.016% [-0.174%,0.225%]; matched neural-only Q90 gain
is0.114% [-0.169%,0.491%]. These5,000-draw paired station comparisons show
no established performance upgrade. The apparent5.19% advantage over trees
largely belongs to the already retained procedure.

## Interpretation and decision

The first self layer changes in every fit: selected-state distance0.033–0.150.
Trainable parameters increase from31,559 to35,143, with unchanged allocated
architecture and frozen message parameters. The new scope is working, but
freezing the first layer is not shown to be the remaining DOC bottleneck.
Keep the retained complete model and its deployed products. Do not combine
this change with failed auxiliary/window variants or initiate a learning-rate
scan or another geographical/external matrix for it.

The latest three source studies answer different questions: joint chemistry
supervision, a second year of memory and full self-encoder tuning. None improves
the actual complete model. Further iteration should investigate additional
predictive information instead of repeating capacity/optimization changes.
One locally testable question is whether contemporaneous source DOC departures
contain information beyond station/climatological residual profiles. Audit
source-only, station-OOF residual synchrony against ecological-distance and
calendar-preserving shuffle comparisons before developing such a branch. Keep
the receiving sites entirely water-quality-free. Monthly weather inputs remain
a parallel data-access task, not an existing model feature.

## Verification

All nine native/integrated predictions replay bitwise. Parent predictions,
initial weights, inputs, OOF population and query cells are unchanged; message
weights remain exactly at initialization. Hidden-label/future-input/initial
forward/save-load contracts pass.971 tests pass, two skip; Ruff and historical
audit pass. The source-generated figures were actually inspected. Large fits
remain local; related files are listed for submission, with Git index writes
unavailable in this environment. Previous geographical and independent-basin
results remain unchanged and were not used for development.
