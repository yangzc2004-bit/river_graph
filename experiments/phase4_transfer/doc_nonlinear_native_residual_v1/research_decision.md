# Small nonlinear DOC readout: source-development decision

## Result and next evaluation

All nine source packages are complete. The nonlinear complete procedure has
MAE1.760777 mg/L versus the retained linear complete model's1.769053:
0.468% reduction [-0.505%,1.581%], with seven of nine package directions
and two of three partition directions positive. This is a small promising
development signal; the station interval does not establish an incremental
gain over the actual retained model. Do not replace the deployed release yet.

| Procedure | K0 MAE, mg/L | Mean bias, mg/L | Q90 MAE, mg/L |
|---|---:|---:|---:|
| Retained complete |1.769053|-0.638608|9.075420|
| Nonlinear complete |1.760777|-0.659018|9.039926|
| Nonlinear neural residual alone |1.763886|-0.647295|9.011321|
| Retained neural residual alone |1.772035|-0.625464|9.053055|
| Strong station-hidden trees |1.866362|-0.611014|9.394458|
| Preceding complete |1.829579|-0.667535|9.225863|

Matched neural-only gain is0.460% [-0.804%,1.922%]. Nonlinear complete gain
over strong trees is5.657% [2.604%,9.152%]; over the older complete model it
is3.761% [2.034%,5.611%]. Those larger comparisons include the existing
station-hidden native residual improvements. The current head's incremental
gain is the0.468% contrast. Q90 incremental gain is0.391%
[-0.344%,1.420%], and mean underprediction grows slightly.

Fix this32-unit head and evaluate it across all five pre-existing HUC4 tasks,
with seeds42–46 and the region-specific saved source trees/backbone/OOF
references. That replication asks whether the readout's conditional correction
is more useful in geographical transfer than in station-role development.
Keep all five regions and K values, the original complete model and strong trees.
Do not change head width, objective or training settings after seeing those
queries. These are already evaluated ST357 geographical tasks, so the follow-up
is retrospective replication of a source-fixed candidate, not a new independent
geographical test or an external basin. The external product remains unchanged.

## Model and verification

The original linear head remains, with a parallel LayerNorm/Linear(32)/GELU/
zero-Linear branch. This adds a small nonlinear readout on the existing
ecological encoder, observation-aware GRU and hydro/regime features. Source
input views, starting backbone weights, OOF reference,30 epochs, patience5,
native-MAE/tail objective and fusion method are matched to the retained model.

Three new tests verify exact initial linear behavior with usable gradients,
hidden-label isolation, repeated-fit/reset reproducibility, saved-state replay
and future-input isolation. All nine source products replay bitwise and leave
the five parent procedures unchanged.5,000 paired station draws average seeds
within partition, weight partitions equally and preserve shared-station/month
structure. Numerical sources and actual inspected figures are saved.

The first analyzer completed replay and all numerical files, then its console
display requested a renamed metric column. Restoring the established
`relative_gain_pct` display column repairs this reporting error. Its original
log and source snapshot remain, with `analysis_repair.json`; no training code,
prediction, bootstrap calculation or endpoint changed.

The full suite passed938 tests, with two skips; Ruff and historical audit passed.
The new mechanism remains a candidate awaiting fixed geographical evaluation.
