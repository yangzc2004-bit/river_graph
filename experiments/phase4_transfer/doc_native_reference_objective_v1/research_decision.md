# Research decision: concentration-aligned reference objectives

## Result and interpretation

All nine source-validation packages are complete (partitions142/143/144,
training seeds42/43/44). Each fits the three fixed reference arms on the same
47 station-hidden covariates. No geographical/external queries are evaluated.
Source receiving stations have no local DOC, pH or conductivity inputs.

| Procedure | Native DOC MAE | Q90 MAE | Signed bias |
|---|---:|---:|---:|
| Actual retained complete model |1.769053|9.075420|-0.638608|
| Retained log ExtraTrees |1.866362|9.394458|-0.611014|
| Native-target ExtraTrees |2.161084|8.760702|+0.264612|
| Log L1 histogram boosting |1.891568|9.676523|-0.830566|
| Native L1 histogram boosting |1.924264|9.471566|-0.644433|

MAE is mg/L, averaged over seeds within each partition and then equally over
the three partitions. Paired intervals use5,000 station-cluster draws with the
same station multiplicity across overlapping partitions.

Changing the forest target to native concentration worsens MAE15.79%
[8.15%,25.09%] against the retained forest and22.16%[12.62%,34.01%] against
the complete model. Every partition worsens. It does improve Q90 MAE6.75%
[2.94%,12.07%] against the forest and3.47%[0.40%,7.69%] against the complete
model. Thus recovering more high concentrations comes with substantially worse
central reconstruction; tail improvement alone is not a complete-model upgrade.

Both L1 boosting references underperform the retained complete model: log
MAE is6.93% worse and native MAE8.77% worse, with paired intervals excluding
zero. Native L1 boosting also loses1.73% overall to its matched log counterpart,
while improving that counterpart's Q90 error2.12%. Station-equal errors tell the
same overall story. Native training is not automatically better merely because
the evaluation uses native MAE. Forest splits, leaf aggregation and concentration
scale are distinct parts of this learning procedure.

## Decision and next experiment

Keep the existing portable complete model, log-space environmental reference,
native-concentration neural residual and memory fusion. Do not refit the neural
branch on any of these three unsuccessful references. Do not blend them using
validation outcomes or run another geographical matrix for them.

The next source experiment separates environmental partition geometry from its
leaf readout: retain the existing log-fitted ExtraTrees splits, reconstruct the
source-only empirical concentration distribution in each leaf, and compare its
native mean with a fixed conditional median. This tests a native-MAE readout
without rebuilding the partitions around large squared concentration errors.
It has a separate plan and directory; results here remain unchanged.

## Verification and products

All retained parent predictions are exact copies. Saved reference replay differs
by at most1.78e-15; every new model has47 inputs and its frozen parameters,
source fit cells and target transform. Analysis is reproducible via
`uv run python scripts/analyze_doc_native_reference_objective_v1.py --bootstrap-draws 5000`.
Plotting uses `scripts/plot_doc_native_reference_objective_v1.py`; PNG/PDF/SVG
were inspected, and the partition legend was moved below its panel to avoid
covering points. The saved execution source remains unchanged.

959 tests pass, two explicitly skip; Ruff and historical artifact verification
pass. The first new isolation test exposed parallel forest accumulation at
1.78e-15. Its correction retains exact feature/tree-state equality and uses a
numerical tolerance only for the reduction; see the repair record and preserved
initial test log. Training and prediction products were not altered.

Large fitted models and feature arrays stay local. Related small products and
code are recorded in the pending-commit whitelist; Git index writes are not
permitted by the current workspace policy, so this cycle is saved but uncommitted.
