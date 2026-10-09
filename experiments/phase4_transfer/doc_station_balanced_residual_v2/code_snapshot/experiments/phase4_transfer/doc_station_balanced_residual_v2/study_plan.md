# Station-balanced DOC residual development

## Question

Can equal-station optimization of the current neural residual improve
reconstruction at new stations? Sparse monitoring gives stations very different
numbers of DOC observations. The current cell-weighted loss gives long records
greater influence. Change this one property before changing architecture.

This mechanism was considered during source development before the external
case was scored. Its execution is specified after the complete external v1
result, which also shows different cell-weighted and station-equal summaries.
Develop only on the previously authorized142/143/144 source train/validation
roles, seeds42/43/44. No geographical or external query selects this version.
Preserve those completed tests as results of their fixed models.

## Matched experiment

Reuse the saved station-hidden ExtraTrees selection and nested station-OOF
residuals from `doc_unmonitored_residual_v1`. The environment base, raw input
scales, ecology/self encoder, GRU, observation decay,38 readout features,
interactions, zero head, learning rates,30-epoch cap and patience5 are unchanged.
Initialize from the saved cell-weighted candidate's *initial* backbone states,
not its final weights. Refit both cell-equal and station-equal residuals.
The cell-equal replay must reproduce the saved source-validation prediction.

For station i with train cells S_i and original tail weight w_it in {1,2}, use

    new_weight_it = w_it / sum_{s in S_i}(w_is) * n_cells / n_stations.

Each station's total loss weight is exactly equal, while a Q90 cell keeps twice
the within-station weight of an ordinary cell. Keep absolute-error loss and
uniform cell batching; no tail threshold, weight scan or smooth-loss change.
Checkpoint/scale selection still uses the original pooled source-validation
MAE. This permits a matched comparison on the same selection objective.

Compare cell-equal and station-equal native residuals and their same-family
source-validation-selected ecological-memory fusion. Retain the original full
model and strong station-hidden trees. The complete station-balanced recipe
is the development candidate; native-only remains a mechanism ablation.

## Research outputs

Report primary all-observation K0 MAE, station-equal MAE, Q90 error/bias,
three-partition directions, seed variation and5,000 paired station-bootstrap
intervals. K0 query cells, source labels and validation roles stay identical
to the parent. No new target labels or auxiliary chemistry enter fitting.

If station balancing has a useful consistent source-validation effect, retain
that candidate for a later explicitly versioned geographical experiment. If it
has no effect, preserve the outcome and proceed to temporal compatibility and
manuscript/product completion. Source validation is development evidence,
not a new independent test. Do not silently rerun or retune the completed
external v1 case.

## Execution repair before the first training epoch

The initial v1 attempt stopped before training because a recomputed parallel
ExtraTrees validation prediction differed from the saved base by at most
2.85e-14 mg/L. Raw neural/readout inputs were bitwise identical. Preserve
v1's source snapshot, configuration and failed startup log. This v2 uses
1e-12 numerical replay tolerance for the forest base, then restores the saved
validation base exactly before fitting. The actual cell-weighted training replay
still must agree with the parent's predictions to1e-10. No labels, loss weights,
architecture, optimizer, budget or scientific comparison changed.
