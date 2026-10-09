# Research decision: antecedent states in the native DOC residual

## Completed experiment

All9 source packages (142/143/144 ×42/43/44) completed. This version keeps
the retained station-hidden tree/OOF reference, ecological encoder, GRU,
native-MAE objective and memory fusion. It adds8 causal hydro-state value/validity
features to the existing linear head, with an availability-only matched control.
No unconfirmed nonlinear or detailed-ecology branch is combined with it.

| Procedure | Source K0 MAE, mg/L | Q90 MAE, mg/L |
|---|---:|---:|
| Retained complete model |1.769053|9.075420|
| Availability residual, complete |1.768244|9.064593|
| Physical-state residual, complete |1.767694|9.066212|
| Physical-state residual, native-only |1.770755|9.045172|
| Retained strong trees |1.866362|9.394458|

Physical-state complete gain over the retained complete model is0.077%
[-0.053%,0.217%],3/3 positive partition directions and8/9 packages.
Gain over the matched availability complete control is0.031%
[-0.029%,0.095%]. Q90 gain over the retained full model is0.101%
[-0.077%,0.357%]; Q90 is slightly worse than the availability control.
These are5,000 paired station draws and equal partitions after seed averaging.

The new descriptors give a small coherent direction but no meaningful overall
performance upgrade. Its5.29% advantage over strong trees includes the existing
neural benefit and must not be attributed to antecedent states. Keep the current
portable release; this candidate does not warrant another geographical matrix.

## Research direction

The four recent source experiments tested detailed ecological classes and
antecedent hydro-climate states in both tree and neural implementations. Each
completed9 packages. None materially improves the actual retained complete
model. Retain their matched-control evidence, without stacking the small changes
or selecting a different winner for each partition or error stratum.

Next test richer source supervision rather than another set of input transforms.
The source-only pH/EC audit is complete in `doc_source_auxiliary_states_v1`:
34,296–36,671 pH and39,467–41,719 conductance labels per partition align
to the DOC station/month grid. Use them to pretrain the existing ecology/GRU
representation, then fit the same native DOC residual. A label-shuffle control
distinguishes useful supervision from extra optimization. New-site K0 still has
no DOC/pH/conductance inputs. The older chemistry-profile retrieval experiment
used receiving chemistry and does not answer this task; it is not reused.

## Verification

All native and integrated products replay bitwise, and prior predictions,
initial backbones, unchanged input channels and tree context are verified.
Causal state construction, missing/zero/negative flow handling and hidden-label
independence passed meaningful tests.947 tests passed, two skipped; Ruff and
historical artifact audit passed. The four-panel figure was actually inspected.

An analysis repair computes direction counts within each overall/Q90 subset.
The initial table reused overall directions for Q90 rows. Predictions, MAE and
bootstrap intervals are unchanged; original tables/code are retained alongside
`analysis_direction_repair.json`. Initial launcher failure before script creation
is preserved in `startup_missing_scripts.log`; no training state existed then.
The fitted version's execution sources remain unchanged in `code_snapshot/`.
