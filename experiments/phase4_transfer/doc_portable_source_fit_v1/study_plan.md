# Fixed-source DOC deployment for the independent external case

## Scientific role

Complete the portable version of the internally selected integrated DOC recipe.
The five-region confirmation is complete. Deployment retains that full recipe
and all four comparison/ablation procedures; it does not replace memory fusion
with the geographically better native-only ablation. External02040104 outcomes
have not been examined. This source refit supplies no additional internal test.

## Source fitting

Retain the validation station identities in the previously specified ST357
split142. Fit on all other ST357 stations with valid DOC, including historical
internal test stations as source observations for the independent deployment.
There are303 source stations with20,288 cells and54 validation stations with
2,283 cells. Test/context arrays are empty. The membership choice uses cell
availability and previously fixed roles. Every seed uses exactly these roles.

Use seeds42–46 and the unchanged geographical recipe: fresh20-epoch
observation-GRU backbone, current daily native residual with120-epoch cap,
and station-hidden integrated candidate with30 extra epochs; patience5.
Fit four300-tree candidates on source inputs and select on source validation.
Rebuild nested station-OOF tree residuals. The upgraded native head starts at
zero and inherits the newly fitted current ecology/GRU weights. Source-only
OOF errors supply ecological memory. The no-message neural path is retained.

The explicit `PortableDOCReconstructor.fit` facade consumes a source dataset,
station-disjoint source train/val roles and the existing eight daily features.
It reuses the geographical fitting helpers rather than developing another
model. `predict`, `predict_components`, `save` and `load` use fitted scales
and saved source experience for arbitrary new site IDs and calendar grids.
New target DOC/pH/conductance tables are never fitting inputs.

## External deployment and comparison

Freeze the arithmetic mean of the five native-scale predictions as the
point ensemble for every procedure. Report individual-seed metrics as a
stability diagnostic in addition to ensemble performance. No procedure,
region or K-specific winner is selected from external scores.

K0 scores all6,514 external DOC cells. K curves reserve the same five candidate
supports for each of130 stations and score5,864 fixed queries at every K.
Use the existing value-blind first/middle/last/quarter/three-quarter ordering.
Choose ensemble support shrinkage from {0,.25,.5,.75,1} on the matching source
validation query, independently by procedure and K, before external DOC is
read. K0 uses zero support. Support is explicit retrospective calibration,
never a backbone input. No external DOC calibrates K0 or its interval.

Report native MAE, station-equal MAE, RMSE, bias, log error, source-derived Q90
error/recall, K curves and hydro/ecology/source-distance strata. Use5,000 paired
station bootstrap draws, carrying months together. Compare against both the
current full model and the strong station-hidden trees. External covariates
are constructed already; input scales and donor profiles come from ST sources.
Source observations align by calendar month. No external river graph is
required for this no-message recipe, and directional context is zero without
named source-to-target river links.

Source validation supplies90% empirical interval calibration in log1p space.
Report coverage jointly with width and high-DOC diagnostics. Calibration
shares the existing model-selection validation role, so it is empirical;
do not claim distribution-free guarantees. External outcomes are reported
for the fixed procedure, without retraining or threshold changes.

## Execution and reproducibility

Run through `run_ladder.py --experiment doc-portable-source-fit-v1`.
Save completed source backbone/tree/current stages for exact resumption.
Keep fitted forests, input grids and full source caches local; retain code,
small model states, manifests, observed-cell predictions and research results.
Test source role isolation, portable replay, new nodes, future information,
support-only adaptation and saved-state restoration. Historical experiments
remain unchanged. After source fitting, verify the portable source-validation
replay before any external scoring.
