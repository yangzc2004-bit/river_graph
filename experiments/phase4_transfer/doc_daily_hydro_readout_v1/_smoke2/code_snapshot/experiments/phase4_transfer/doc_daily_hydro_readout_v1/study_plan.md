# DOC support readout learning from the selected recurrent state

## Scientific question

Can source support episodes teach the current 64-dimensional recurrent state
a better two-dimensional station adaptation subspace? Fixed readout refresh
helped K3 but not K5. This experiment learns the readout before changing the
recurrent clock or adding another temporal operator.

## Design fixed before new training

- DOC only; existing spatial partitions 142/143/144 and seeds 42/43/44.
- Nine fits of 128 parameters. Retain the selected daily-head (`off`) expert.
- Freeze its forest, ecology encoder, GRU, decay, native scalar head and
  ecological residual profile. Copy the native full-grid product unchanged.
- Initialize P exactly at the saved v4 readout P0. Optimize
  P=P0+||P0||F D, D=0 initially. No PCA, whitening or QR initialization.
- Adam LR0.001, at most30 epochs, patience5, gradient norm1, station batch32.
- Hidden export/projection batch2048, torch threads2. Use the same32 fixed
  calendar anchors, station scalar RMS and scale floor0.0001.
- Source inputs use the original receiving-station-fold-hidden DOC views.
  Source native baseline=max(0, expm1(context OOF)+selected_scale*native_delta),
  with the unchanged true 38-channel source scalar-head features.
- Only the forest part is OOF. The frozen neural expert already trained on
  source labels; these are source-training residuals, not fully OOF residuals.
- Source episodes use K3/K5 and ridge1/10, alpha1; optimize native query MAE
  equally over stations and the four combinations. Five value-blind support
  months are sampled per station/epoch and excluded from every query set.
- Validation uses held source-validation stations, their full-source native
  baseline and the existing reserved-five support schedule over all validation
  observed cells. Select pooled query MAE averaged over the same four losses.
  Epoch0 remains a selectable checkpoint. No target-query labels are used.

## Products and comparisons

Four bases: constant, legacy, refreshed_fixed and refreshed_learned. Each has
direct and ecological-integrated pipelines and fixed K0/1/3/5 query curves.
Refit the unchanged downstream alpha/ridge and positive-K ecological mixture
grids on source validation; lock K0 gamma to the parent choice.

Primary mechanism contrasts: learned versus fixed refresh, K3 and K5,
direct and integrated (four). Overall reference: learned versus legacy,
the same K/pipeline combinations (four). Report native/log MAE, RMSE, R2,
Q90 and ordinary errors, bias, partition/station consistency and concentration.
Use5000 paired whole-station bootstrap replicates, averaging seeds within
partition and weighting the three partitions equally. Preserve all curves;
do not select a K-specific route from target results.

K0/K1, constant/legacy/fixed products and the native grid must reproduce their
parents exactly. Save initial/selected readouts, training traces, episode
schedules, source baselines and identities, normalized/raw bases and anchor
statistics. Independently reconstruct source/full states and replay adaptation.

## Interpretation

These previously seen station partitions support model development. A positive
result demonstrates better support geometry for the frozen expert, not a new
river-message effect or independent external confirmation. Rolling hidden
states are causal; full-record anchor normalization and positive-K adaptation
are retrospective reconstruction. Readout failure retains the existing main
candidate and motivates a subsequent recurrent-clock comparison.
