# Fixed source-information geographical replication

## Source-selected question and scope

The completed source-development study selects the same-month source innovation
readout in the existing ecology/GRU residual. Its overall gain over the retained
complete model is0.451% [-0.045%,1.032%]; Q90 gain is0.840%
[0.150%,1.906%]. This small development signal motivates a finite geographical
replication, not a neighbour/lag/architecture search.

Use established ST357 HUC4 test roles1013/1019/0708/1030/1101, with the next
region in that cyclic order as validation and remaining stations as sources.
These tasks were evaluated in earlier model versions. Report this as
retrospective geographical replication, not new independent external validation.
No receiving-site DOC/pH/conductance enters K0.

## Fixed candidate and references

Retain the parent geographic environmental reference, station-OOF residual
targets, source preprocessing, initial neural weights, ecological self encoder,
observation-aware GRU,12-month history,30-epoch/patience5 optimization and
tail weight2. Append the three features selected in source development:
current native source innovation divided by the fixed1mg/L unit, matched donor
count/20 and weight mass/(1+mass). Preserve all seven old hidden-feature
interactions and add innovation×GRU state. Dimensions38→41; +67 parameters.
No new fusion gate, graph depth or loss search.

For each source query fold A, donor references exclude both A and donor fold B.
Ten unordered pair forests per package supply both directions. The library
never includes query-fold labels. Validation/test use source-only observations
and the parent's source station-OOF reference. Frozen seasonal statistics use
permitted source training records, consistent with this spatial task; prediction
reads only current or earlier donor observations. This is not temporal-holdout
fitting. Twenty ecological candidates and the original matched earlier-season
availability restriction remain fixed.

Compare:

- Retained complete model and retained neural-only model.
- Fixed real-source neural-only and complete ecological-memory procedure.
- Strong station-hidden ExtraTrees and ExtraTrees with identical source inputs.
- Preserve the parent's older complete/matched-tree references for context.

Each package reuses verified parent geography preprocessing/backbone/OOF and
fitted reference states; it does not refit them or select a new forest family.
New pair forests, source library, neural and enriched-tree states are saved.
All candidate checkpoint, residual scale, memory fusion and simple support
adapter choices use the source validation region only.

## Execution and evaluation

Complete five regions ×seeds42/43/44 first, then the SAME candidate and
comparators at45/46. This is25 fixed packages,250 pair forests,25 neural
fits and25 enriched-tree fits. The initial package checks input/replay and
role isolation; numerical test scores do not change continuation.

Primary K0 scores all valid DOC cells in each region. The separate K0/1/3/5
curve has fixed queries excluding five reserved support cells. Support labels
are used only by the saved simple adapter, never by the backbone or library.
Save point predictors before accessing designated test scoring/support.

Main estimator: equally weight five regions after averaging training seeds.
Use5,000 paired station-bootstrap draws, keeping station months together.
Report paired MAE/reduction, RMSE, bias, station-equal MAE and source-derived
Q90 error/recall. Show region results and fixed-query K curves. Source support,
ecological novelty and hydro completeness strata use source-derived definitions.
Keep both improvements and regressions. A comparison with an equally informed
tree separates source information from architecture; improvements over older
models do not establish improvements over the retained complete model.

The working5% goal against both primary comparators remains. This study can
establish a smaller reliable incremental benefit or close this candidate.
No per-region/K winner, geographic retuning or external DOC calibration.
Portable deployment and manuscript remain the preceding evaluated release
until this fixed replication is analyzed.
