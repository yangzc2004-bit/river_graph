# DOC sampling-aware river GNN: completed source-development experiment

All54 fits completed (partitions142/143/144 x seeds42/43/44 x six arms) in
`experiments/phase4_transfer/doc_sampling_river_v1`. No new forests or full local
backbone fits.22,571 frozen DOC monthly cells reconcile with original dates,
without changing DOC values; receiving water-quality dates/values remain absent.
The existing river GNN received result-weighted source ages, sampling support and
measured source/receiving fixed-month-end flow phase. Controls included dates only,
ordinary monthly attention, shuffled source phase and matched nonancestor donors.

Primary MAE1.717288mg/L versus ordinary1.717226 gives−0.003607% gain, station
interval[−0.044268,0.033723]; real source-flow correspondence does not beat shuffled.
Previous environmental-state candidate1.714729 is better. Dates only improve by
0.003954%, interval spanning zero. Q90 and station-equal metrics do not rescue the
primary model. Do not replace the retained best candidate or released model.
Detailed comparison and input coverage: experiment `research_decision.md`.

Observed-source availability covers21.7–52.5% of query cells,50 of140 stations;
both measured phases cover17.4–20.9%. Supported/phase/young-age subgroups do not
establish incremental gains. These are source-validation development outcomes,
not independent regional or external validation, and not evidence that river
transport lacks ecological relevance.

All9 input reconstructions,54 checkpoint replays, fixed products, independent
metrics and18 primary station intervals verified. Two figures viewed; legend
repair only.9 new tests; full suite1,483passed,3skipped; Ruff/history audit pass.
Large caches/checkpoints stay local. No automation restarted.

Next scientific task: jointly train upstream temporal states and receiving local
correction on genuinely station-held local-predictor episodes. Stop minor metadata
extensions of the same frozen scalar readout. The complete source fitting base
is in-sample, although donor innovations are environmental OOF; isolate this
training/evaluation mismatch in a new version. Existing geographical/external
results remain results of their previously evaluated model.
