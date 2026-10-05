# Execution progress: DOC reconstruction at unmonitored stations

## Research objective

Improve the existing full DOC model for stations with no DOC, pH or conductivity
history. Retain environmental trees, ecology encoding, observation-aware GRU,
daily hydrology and residual decomposition. K0 is the primary task; K1/3/5 are
separate fixed-query support diagnostics. Continue the already authorized plan
through geography, external replication, a portable predictor and manuscript.

## Completed work

1. `doc_unmonitored_tasks_v1`: fixed five HUC4 tasks, input-condition definitions,
   all-observation K0 populations and separate support-query schedules.
2. `doc_source_retrieval_v1`: nine development packages. Donor attention adds no
   benefit; hydro pretraining gives -0.34% relative gain. Keep both findings.
3. `doc_source_retrieval_v2`: donor-minus-local contrast, nine packages; 0.06%
   gain, with an interval crossing zero. Uniform/removed-donor controls agree.
4. `doc_source_retrieval_v3`: station-balanced native-error donor profiles;
   nine packages; 0.04% gain, interval crosses zero. No useful new retrieval
   component was selected from these experiments.
5. `doc_unmonitored_trees_v1`: training source rows now hide the whole station
   fold's water-quality history. Nine packages; 4.97% lower MAE than ordinary
   matched trees, but still 2.01% worse than the complete current model.
6. `doc_unmonitored_residual_v1`: retrain the existing ecology/GRU native residual
   on the new tree's station-OOF errors. Nine packages, saved-state replay, 5,000
   paired station bootstrap draws and inspected vector figures complete.
   Integrated K0 MAE1.769053 vs current1.829579:3.31% improvement
   [1.72%,5.03%]; vs station-hidden trees1.866362:5.21% [2.69%,8.02%].
   All three development partitions improve. These are source-validation scores,
   not geographical or external confirmation.
7. External input preparation: 02040104 selected by data conditions,130 stations,
   6,514 DOC monthly observations,520 calendar months,5,864 reserved-support
   curve queries. All stations have adequate ecology features. At DOC cells,
   temperature availability is96.5% and discharge34.1%. Label-free input grid
   and scoring-only dataset are saved separately. No external prediction has
   been evaluated.

## Active geographical experiment

Directory: `experiments/phase4_transfer/doc_geographical_confirmation_v1`.
Read its `study_plan.md` and execution source snapshot before recovery.
Entry point: `uv run python scripts/run_ladder.py --experiment
doc-geographical-confirmation-v1`. HUC4 order1013/1019/0708/1030/1101;
next region supplies validation; seeds42/43/44 initially. The first package
1013×42 completed in710seconds; identity, station-OOF exclusion, saved-state
native replay and integrated bitwise replay passed. The full15-package batch
is now running, reusing that completed package. Analysis and plotting scripts
are prepared for the complete batch. Expected remaining duration is roughly
2–3hours on this machine; source early stopping can change that estimate.

The current backbone and current-model recipe are fitted afresh for each
geographical role assignment. No development checkpoint trained on a test
region is reused. The fixed candidate inherits that task's current encoder/GRU
and refits its new station-hidden residual. Keep both ordinary and station-
hidden strong trees. Large tree and input caches remain local.

## Next work in order

1. Finish the fixed15-package geographical experiment. Implement and run the
   saved-state verifier, analysis with5,000 paired station draws, and figures.
   Evaluate equal-region K0, station-equal MAE, high-DOC error/recall/bias, fixed
   K curves, novelty/hydrology/source-similarity strata and station directions.
2. Supplement fixed candidates and controls to seeds45/46. Report the complete
   result; do not pick a winner per test region or per K. If geography fails,
   preserve that version and continue mechanism development on source roles.
3. Package arbitrary-new-node/month inference using saved source preprocessing,
   model state and source experience. Add hidden-label, causal-input, new-N/T
   and save/load tests. Do not recompute normalization on the external cohort.
4. Run the fixed external02040104 case with the internal-selected model. K0
   uses no external DOC calibration; K1/3/5 use only designated supports.
   Diagnose performance and hydro limitations without tuning on its query.
5. Check the two existing time-reconstruction scenarios, then val-only interval
   calibration jointly with width. Retain past uncertainty-ranking findings.
6. Integrate the method, geographical results, support curves, high-DOC panels
   and external results into the existing manuscript using the user's
   importance → prior-method gap → new model → experiments → future-work story.

The experiment scripts save finished stages for recovery. If executing code
needs repair, retain its source snapshot and explicitly document a new version.
Run pytest, Ruff and the historical artifact audit for each completed version.
Commit only related source, tests, small records, predictions and recomputable
analyses; do not stage unrelated historical untracked files or large local fits.
The existing30-minute DOC heartbeat now continues this plan and stays quiet
when training state has no actionable change.

## Software verification at this checkpoint

Full pytest:907passed,2skipped; Ruff:all checks passed; historical artifact
audit:exit0. A subsequent dedicated five-region metric test also checks that
source-validation analysis helpers cannot accidentally substitute three-region
averaging for the geographical estimand. Run that test and the final full suite
when finalizing the geographical version.
