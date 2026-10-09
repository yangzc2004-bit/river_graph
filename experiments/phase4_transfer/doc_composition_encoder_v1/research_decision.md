# Research decision: detailed ecology in the retained DOC encoder

## Results

All9 matched source-role packages completed, with unchanged trees/OOF reference,
the existing32-unit ecological embedding and64-unit observation-aware GRU.
The new22 input columns start at zero; source-only native residual fitting uses
the retained30-epoch budget, patience5, tail weight2 and linear prediction head.
The same-dimension aggregate encoder is retained as a matched control.

| Procedure | Source K0 MAE, mg/L | Q90 MAE, mg/L |
|---|---:|---:|
| Retained complete model |1.769053|9.075420|
| Expanded aggregate encoder, complete |1.769011|9.080510|
| Detailed-composition encoder, complete |1.769340|9.067969|
| Detailed-composition encoder, native-only |1.772445|9.046094|
| Retained strong trees |1.866362|9.394458|

The detailed complete model's gain over the actual retained complete model is
-0.016% [-0.096%,0.061%]. Gain over the aggregate complete control is-0.019%
[-0.170%,0.145%]. No overall upgrade is established. Its5.20% benefit over
the strong tree is inherited from the retained neural procedure, not a5.20%
gain from new ecological categories. Its Q90 reduction is0.082% [0.016%,0.198%]
over the retained full procedure: a small tail effect, insufficient to compensate
for the lack of overall improvement. Retain the current portable release.

## What this narrows down

The preceding tree experiment found a1.32% improvement over its expanded
aggregate control, but not over the original strong tree. The current neural
test likewise does not improve the actual complete model. Detailed categories
are useful to retain as data, but adding them under this fitting setup has not
resolved the remaining new-site error. This does not prove that those land-cover
categories are intrinsically uninformative or that all ways of learning them fail.

The retained-model source error decomposition now provides a more direct next
research priority. Approximately9.93% of source-validation cells are above the
source training Q90 threshold; they carry43.76% of absolute error and84.67%
of squared error. Squared station mean bias accounts for20.84% of MSE, with
79.16% remaining within stations. The highest-error20% of stations contribute
55.49% of absolute error. These are equal-partition averages after seed means,
not new ecological samples or a causal decomposition. Station-specific observed
bias is unavailable at K0 and is not proposed as an inference input.

Next prioritize dynamic hydro-climate states that might distinguish high-DOC
conditions. Keep the retained architecture and source roles, rather than add
another larger readout or repeat the completed pure tail-weight/Gaussian-mixture
scans. First test whether new causal exogenous state descriptors provide useful
information, with a matched availability-only control, before retraining the
neural reference. Source validation alone drives this development.

## Verification

All native and complete products replay bitwise; the prior predictions, tree
context, OOF target construction and initial ecological/GRU/decay weights are
preserved. Initialization/gradient, hidden-label, future-input, save/load and
new-node-count tests passed.944 tests passed, two skipped; Ruff and historical
artifact verification passed. Four-panel figures were actually inspected.

A synchronous progress callback's loop name was explicitly bound after all9
fits to satisfy Ruff B023. It had executed before advancing each loop, so the
repair changes no training values or logged arm identities. Original execution
sources remain in `code_snapshot/`; `post_training_lint_repair.json` records
both source hashes and the completed state. No saved training code was replaced.

The analysis also corrects overall/Q90 direction counts to use each selected
subset. Original tables/code are retained in `analysis_original_direction_counts/`;
predictions, MAE and bootstrap intervals are unchanged. The repair is recorded
in `analysis_direction_repair.json`.
