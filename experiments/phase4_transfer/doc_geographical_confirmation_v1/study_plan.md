# DOC reconstruction in entirely held-out HUC4 regions

## Question and fixed candidate

Does matching source training inputs to unmonitored stations improve the existing
ecology/observation-GRU residual predictor in geographically separated regions?
The candidate is the complete station-hidden tree + native neural residual +
ecological memory recipe. It was selected using source-validation roles of
partitions 142/143/144, without evaluating their target roles. Nine development
packages gave 3.31% lower K0 MAE than the current complete predictor and 5.21%
lower MAE than the station-hidden tree comparator. These are development results.

## Geographical tasks

Use the previously saved masks in `doc_unmonitored_tasks_v1/geographical_masks`.
Test HUC4 regions rotate in the order 1013, 1019, 0708, 1030, 1101. The next
region supplies validation, and all remaining stations supply training. Start
with seeds 42/43/44; retain complete models for subsequent 45/46 confirmation.
Every backbone, forest, OOF fit, profile and adapter is refitted using this
geographical task. Historical fitted weights are not reused.

This is retrospective geographical replication inside ST357. The cohort has
been studied before; it is not an external river-basin validation.

## Matched information and budgets

No target-station DOC, pH or specific-conductance input is available. Source DOC
context remains available. Temperature, discharge, ecology, season and the
existing eight daily-flow descriptors are retained. The historical backbone
recipe uses 20 epochs, hidden64, two spatial layers, a 12-month observation GRU,
and no-message self encoding. The current daily native readout has a 120-epoch
cap. The new residual inherits that freshly fitted encoder/GRU, resets its
readout to zero, and receives 30 additional epochs with patience5. The comparison
therefore tests the complete upgraded recipe, not equal total optimization
steps. A resource-matched refinement can be reported separately if needed.

The current model retains its 39-feature environmental context forest plus the
daily neural residual and ecological memory. Two strong daily-input tree
comparators have 47 features: ordinary source inputs and station-fold-hidden
inputs. Both use the same four 300-tree candidates selected on validation MAE.
The new residual uses station-hidden tree predictions and genuinely station-OOF
residual targets. In each outer OOF fit, training feature folds are rebuilt
inside the remaining source stations. Local target values and visibility are
absent from every pseudo-target input.

## Predictions and support curves

Primary K0 includes every valid DOC cell in the test region, including sparse
stations. Five designated candidate support cells are excluded from the separate
K0/1/3/5 curve query at every K. Identical validation-selected log1p mean-residual
adapters are applied to each complete point predictor. Support is used only in
this explicit retrospective calibration; it does not refit the backbone.

The five reported procedures are current_model, matched_daily_trees,
station_hidden_trees, unmonitored_residual and unmonitored_integrated. The fixed
primary candidate is unmonitored_integrated. Do not select a different winner
for each test region or K.

## Analysis and next action

First average seeds within region, then weight the five regions equally. Report
paired MAE, relative gain, RMSE, bias, log error, source-derived Q90 error/recall,
station-weighted MAE and fixed-query K curves. Use 5,000 paired station bootstrap
draws, carrying all months and seeds of each sampled station together. Include
ecological novelty, hydro availability and source similarity strata, and the
distribution of improved/worsened stations. Retain every region in the result.

The working performance aim is at least 5% lower K0 MAE than both current-model
and strong-tree comparators, at least three improving regions, and an overall
paired interval supporting improvement. Smaller consistent gains remain useful.
Test scores do not change this version's model, losses, query cells or selection
policy. Future mechanism development returns to source roles. The selected
external DOC case 02040104 is prepared but remains unevaluated.
