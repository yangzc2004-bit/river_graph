# Chemical-state DOC reconstruction on new station assignments

Status: fixed next-stage recipe; training has not started.

## Question

The current chemical decoder improves zero-support DOC reconstruction, and
chemical-state station calibration improves K3. Refit the accepted model and
fixed comparators on new station assignments to assess whether these gains
survive a new training/validation/test partition. Do not expand the architecture
or retain the unsuccessful residual kernel in this comparison.

Use station-partition seeds242/243/244 and training seeds42/43/44. The new
partition seeds are the preceding seeds plus100, fixed without inspecting DOC
values. Preserve ST357, its node/month order, observation-only partitioning,
232 source/54 validation/71 test station counts, nested K0/1/3/5 support and
fixed-query rules. Support remains retrospective calibration.

These are new role assignments on the existing Mississippi cohort, not new
external measurements or an independent basin. All fitted states, source
normalizers, source OOF forests and calibration parameters must be refitted;
old models trained on these new target stations cannot be reused as predictors.
Fixed covariate packs and their station/month alignment may be reused.

## Model curves

1. Retained general daily-hydrology/ecological model with legacy support basis.
2. Nonlinear chemical neural model with legacy support basis.
3. Nonlinear chemical neural model with validation-selected legacy/mask/chemical
   support coordinates: the accepted whole-K procedure.
4. Chemical tree with the same validation-selected representation policy.

Keep the decoder's matched no-auxiliary and availability-only neural controls
for information attribution. They use the same architecture, initialization,
optimization budget and absent-chemistry fallback as the chemical-value head.
All curves use the same source roles, held-station query cells and DOC readings.

## Minimal training recipe

Reconstruct only retained branches; historical model-search matrices are not
required.

- Initial expert: `UnifiedDOCReconstructor`, original20-epoch/5-patience
  initialization, selected-context forest and five station-blocked source OOF
  forests. Preserve the source-only inference/visibility definitions.
- Legacy support basis: retained `EpisodicStationProjector` (100/15) followed
  by retained `EpisodicTemporalAdapter` (30/5). This retains the actual earlier
  GRU support representation, rather than substituting a new PCA or projector.
- Ecological residual profile: refit the single30-epoch `interaction_tuned`
  `NativeTemporalResidual` initializer used to choose ecological-affine donor
  count/ridge, then retain that profile. Changing profile selection to the final
  neural expert would define a different recipe and is not part of this run.
- Native neural correction: retained `EncoderNativeResidual` off arm,
  `last_self_ecology`, extra38, concentration interactions and tail weight2,
 120 epochs/patience5. Use the executed parent configuration for other fixed
  optimizer/feature settings.
- Explicit feature models: current-daily tree and chemistry tree only.
- Chemical decoder: matched nonlinear no-auxiliary, masks and chemistry heads,
 120 epochs/patience10; retain source-only scaling and learned embedding form.
- Calibration: source-only two-dimensional mask/chemical PCA, existing
  alpha/ridge and ecological-mixing rules, then representation choice by final
  active source-validation MAE. Preserve K0 mixing and all fallback definitions.

Source forest predictions used to train native corrections are OOF; the neural
correction is source-trained. Do not describe the complete source pipeline as
OOF. Fixed class settings come from the executed parent archives, not from
retuning against results of the new partitions.

## Results

Report whole K curves and fixed paired comparisons: accepted chemical model
versus the general model at K0/K3/K5; selected coordinates versus legacy at
K3/K5; chemical coordinates versus mask coordinates at K3/K5; chemical-value
head versus its matched input controls; and accepted neural versus chemical tree.
Keep all partition and seed results.

Use the existing seed-mean/partition-equal estimator and joint whole-station
bootstrap with5000 draws. Report MAE/RMSE/R2/log error, Q90 error/bias/detection,
ordinary errors, chemistry availability and station response concentration.
Separate the new-assignment results from prior development results. No
target-selected K splice, station route or post-result model changes enter this
comparison.

## Outputs and expected work

New fitted states, support basis, native components, full-grid products,
observed-query products, configurations and analysis belong to this directory.
Existing Phase0–3 and development experiments remain available.

The original nine initial experts took47.6 summed minutes; retaining only the
necessary subsequent branches suggests roughly50–70 minutes of serial fitting
for nine new packages on the current machine, plus export/replay. This is a
planning estimate. A compact orchestration script should refit retained branches
directly instead of invoking every historical experiment runner.
