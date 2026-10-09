# Train the retained DOC model with region-hidden source observations

Two source-only concentration experiments are complete. Environmental bias
calibration and a log-concentration neural readout do not improve the actual
retained complete model. Keep their numerical results. This round instead tests
the observation regime used to construct training examples.

## Scientific change

The existing source station folds scatter held sites among monitored neighbors.
Geographical deployment removes an entire water-region role. Reconstruct source
training views by hiding complete source HUC4 groups, using graph-node `huc_cd`
aligned by site identity. Whole regions remain intact; source DOC observation
counts balance up to five folds. Seed changes only tie ordering. No target
concentration, validation error or historical geographical outcome sets a fold.

Fit the environmental reference on these region-hidden training feature views,
using the preceding strong tree's selected300-tree hyperparameters. Build new
region-blocked OOF predictions: the held regional fold's DOC labels and input
context are excluded from its forest; remaining fitting rows themselves use
inner region-hidden views. Rebuild existing raw M1 source features using exactly
the same outer regional folds. These are source observation views and forest
OOF predictions, not an independently cross-fitted neural ensemble.

Train the retained native mg/L residual with its existing ecological encoder,
GRU, daily hydro features, zero head, initial backbone weights,30 epochs,
patience5 and tail weight2. Do not incorporate either failed concentration
correction. Keep the full source-validation inference view and all local
water-quality hiding unchanged. Regions are not added as predictive inputs.

## Source-development panel

Partitions142/143/144 × seeds42/43/44. Retain all five preceding arms and add
regional-view trees, their native neural residual and the unchanged ecological
memory fusion. Report5,000 paired station intervals, MAE, bias, Q90, hydro
availability and regional fold exclusions. The main comparison is against the
retained full model; tree-only changes and neural gain beyond the new tree are
separate comparisons. These station-validation panels diagnose development;
they are not the previous HUC4 geographical confirmation or external basin.

Persist completed forest, OOF and neural stages before memory/product export.
Keep old results and execution snapshots. A useful source candidate can proceed
to a separately fixed geographical replication; no old geographical/external
query chooses this round's groups, model dimensions, coefficients or epochs.
