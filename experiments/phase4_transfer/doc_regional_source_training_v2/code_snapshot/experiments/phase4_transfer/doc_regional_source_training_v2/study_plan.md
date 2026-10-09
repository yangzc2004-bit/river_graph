# Region-hidden source training for new-station DOC

## Question and model

Does training on whole-region observation outages improve the retained DOC
model's new-site performance? Keep its ecological encoder, observation-aware
GRU, native mg/L residual head, daily hydro inputs and ecological-memory fusion.
The only scientific change is the source observation regime and its matched
environmental reference. Neither preceding concentration correction is added.

Use source-development partitions142/143/144 × seeds42/43/44. Receiving sites
have no local DOC, pH, conductance, chemical history or availability channels.
Their DOC is used only for source-validation checkpoint selection and scoring.
No previous geographical or external query selects this experiment.

## Regional training views

Read HUC identity from graph-node `huc_cd`, aligned by `site_no`. The existing
table contains HUC8 and HUC12 codes, sometimes with a dropped leading zero.
Restore the applicable eight/twelve-digit width before reading HUC4. This
normalizes the source table's representation; it changes no catchment definition.

Keep every source HUC4 intact. Greedily balance observed source row counts across
up to five folds, with seed-dependent tie ordering. Concentrations and prior
geographical outcomes do not determine folds. Hide each fold's target chemistry
and context when constructing its source examples. Fit the environmental tree
with the preceding selected300-tree hyperparameters on these regional views.

For every outer regional fold, remove all its fitting labels and context.
Create inner region-hidden examples among remaining stations and fit an OOF
forest that predicts the held source cells. The new OOF reference covers exactly
source-fitting labels. Construct M1 source observation views using these same
outer folds. These are feature views and forest cross-fitting; the neural
model itself is not a cross-fitted ensemble.

Start the neural fit from the preceding candidate's initial spatial encoder,
GRU and decay states, with a zero head. Retain30 epochs, patience5, tail weight2
and the original native residual-scale grid. Full source-validation raw inputs
remain identical to the retained model. The environmental prediction and its
residual normalization are regenerated consistently with regional training.

## Comparisons and outputs

Preserve the parent's five procedures and add `regional_trees`,
`regional_residual`, `regional_integrated`. Compare the new complete procedure
against the actual retained complete model and strong station-hidden trees.
Separate tree-only changes from neural value beyond the new regional tree.
Report MAE, RMSE, bias, Q90 and hydrological-availability strata, averaging
seeds within partition and weighting the three partitions equally. Use5,000
paired station-bootstrap draws; retain all months and shared-station identities.

Save fitted forest, OOF and neural stages before memory/product export. Preserve
parent predictions and old experiments. Successful source evidence can identify
a candidate for separately fixed geographical replication. This development
panel is neither geographical confirmation nor independent-basin validation.

## V1 startup repair

V1 stopped before fitting because its node reader required HUC8, while45 of
357 graph-node entries use HUC12 (three lack a leading zero). All codes are
numeric. V1's code snapshot, configuration and failed log are preserved.
V2 accepts the two existing coding widths and records their correct HUC4.
Runner and analyzer share this normalization. Scientific settings are unchanged;
no V1 fitted model or prediction package exists.
