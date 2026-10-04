# DOC recurrent clock comparison

## Question and model change

The selected model attenuates its shared recurrent state using DOC observation
age. At receiving stations whose entire DOC history is hidden, that scalar
grows from the beginning of the record. Test whether a clock grounded in
available hydrology gives better cross-station reconstruction.

Keep the same ecological/self encoder, 12-month GRU, daily38-channel native
head, concentration/flow interactions, forest and old v4 support basis.
Only the scalar supplied to the four-input decay layer changes. The original
M1 age feature in the spatial input remains unchanged.

## Three versions

1. **legacy**: unchanged DOC-age clock; reuse the nine verified `off` models
   from `doc_daily_hydro_memory_v1`, trained with the identical budget below.
2. **unseen_neutral**: the decay clock is zero wherever last-observation-valid
   is zero. For receiving stations this removes the advancing calendar scalar.
   The allocated parameters stay the same, but the64 age coefficients receive
   no age signal there; report this reduced effective input capacity.
3. **flow_window**: clock=log1p(flow age)/log1p(12), where flow age is computed
   only from monthly discharge visibility within the current rolling window.
   Initialize/cap at12; reset to0 on visible discharge and otherwise increment
   at valid steps. Padding does not advance the clock. No information outside
   the window or from future months is added.

These are shared-state decay interventions. They are not independent hydraulic
storage paths, physical residence times, or evidence of river-message value.

## Training fixed before new fits

DOC, partitions142/143/144, seeds42/43/44. Two new versions ×nine packages=
18 neural fits; reuse nine matched legacy controls and all fixed forests.
Initialize each new model from the same original source encoder/GRU/decay
used in the legacy experiment, with a zero native residual head.

Use the same source station-fold-hidden DOC inputs and forest OOF baseline;
native source MAE, Q90 weight2; uniform observed-cell shuffle; batch512.
Encoder last_self_ecology LR1e-5, GRU/decay LR1e-4, head LR1e-3; max120
epochs, patience5, clip1, torch threads2. Select unweighted source-validation
fixed K0 query MAE with epoch0 and residual scales{0,.25,.5,1}. Never use
target query labels for clock, checkpoint or adaptation selection.

Refit the existing direct alpha/ridge and ecological gamma/alpha/ridge grids
on source validation. All versions use the unchanged constant and v4
gru_tuned_anchor support bases. This holds support geometry fixed while
testing recurrent storage. Report K0/1/3/5, all fixed target query cells.

## Analysis

Eight specified contrasts: each new clock versus legacy, K0/K5, direct and
integrated. Report native/log MAE, RMSE, R2, Q90/ordinary error and bias,
recall/false-high rates, station/partition consistency and concentration.
Use5000 whole-station paired bootstrap replicates with seed means within
partition and equal partition weights. Show every version and K; do not
choose a K-specific or target-selected station route.

Record source-validation traces, selected scales, clock/decay summaries,
unchanged-basis checks and prediction components. Independently replay the
visibility views, clocks, native head and adaptation. If no improvement
appears, retain the daily-head/legacy-support candidate and shift effort
toward the remaining high-DOC underprediction rather than repeat clock fits.

The station partitions have been seen in development. Positive-K support and
the inherited feature-anchor normalization are retrospective reconstruction.
