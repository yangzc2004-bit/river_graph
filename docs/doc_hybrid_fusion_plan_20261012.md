# RiverDOCNet explicit-history and direct-readout development comparison

The authorized change retains the environmental MLP, twelve-month GRU, and
upstream/downstream/global graph Transformer. A small MLP encodes all 12 by 8
original monthly driver/mask/age/season features. Its 12-dimensional output
joins the local state before graph interaction. A jointly trained readout also
receives the environmental, recurrent and lag representations directly,
alongside the graph representation. Local and graph normalization are separate.
No DOC context, station identity, daily-flow additions, new data, tree residual,
gate, or change-loss term is introduced.

This is a fixed architecture candidate, not a search selected using test scores.
The control is the original-input/log-loss directional RiverDOCNet from the
independently audited dynamic study. History random forest and XGBoost from
that same study are the strong controls. Their frozen checkpoints, selections,
validation and query predictions are reused as immutable dependencies and
independently replayed again. Only the new architecture requires 24 new fits.

Data: the frozen ST357 Mississippi monthly observed-sample DOC cohort. Exact
parent masks are copied without regeneration: five HUC4 whole-station tasks
times three seeds, and 3/6/12-month gaps times three fixed mask/optimization
seeds. Time gaps share their fixed three-month scoring cores; they are not
nine independent repetitions or complete twelve-month gap trajectories.

The new model keeps the original normalization, log1p DOC MSE, month batches,
current/previous-month forward schedule, AdamW settings, clipping, maximum 60
epochs, minimum 15 epochs, patience 10 and native validation-MAE checkpoint
selection. This isolates the architecture bundle at the existing budget;
it does not settle the separate question of longer training. Shared original
environmental, recurrent, combine and graph tensors have the same seeded
initialization. The added branches/readout increase parameters and effective
capacity; any gain cannot be assigned uniquely to lag preservation versus
the direct readout or additional capacity.

The fixed new architecture is scored only after all its validation checkpoint
choices are sealed. Native MAE and log-MAE remain joint primary endpoints.
RMSE, predictive R-squared, station-centered native/log scores, variability,
and consecutive-month change error/correlation remain supplementary. Spatial
metrics pool the five regions within seed before averaging three seeds. Time
metrics average the three fixed repetitions per gap; the equal-gap overview
averages their nine scores, not repeated query cells pooled into a new sample.

Compute uses six independent single-thread workers, with at most three
simultaneous full-cohort temporal fits to limit memory. An isolated run_ladder
executes the frozen sources. A process-scoped macOS sleep inhibitor keeps the
authorized local batch awake; it does not change system preferences.

Freeze includes source hashes, dataset/nodes/mask hashes, baseline-package
hashes, Git commit and dirty-tree status. Every new prediction has a sidecar.
Independent verification checks labels, query equality, feature equivalence,
validation-only selection, checkpoint replay and metric recomputation. Paired
regional intervals are exploratory on previously inspected development
regions, with no multiple-comparison correction or global SOTA inference.
Old paper endpoints and frozen results are unchanged.
