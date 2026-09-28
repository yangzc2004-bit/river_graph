# Model-upgrade route decision

U1 passes: use 30 epochs for any new confirmatory training. U2 remains a
diagnostic branch because its corrected partial results are mixed and its
unmonitored-station arm was not completed. U3 is complete and shows that a
temporal random forest outperforms the current EcoHydroGraph in all four tested
DOC/electrical-conductance families.

The next model work, if continued, must be a targeted hybrid or representation
study that explains this gap. Do not start Transformer, deeper-GNN, or broad
hyperparameter matrices. The existing EcoHydroGraph remains the spatially
structured model, while Temporal RF becomes the strong performance baseline.
The paper should report this boundary explicitly unless a later hybrid closes
it under the same masks and provenance rules.

U4 tested a validation-selected convex blend of Temporal RF and the temporal
graph model. On strict temporal extrapolation, the blend improved all three
seeds for both DOC and specific conductance (mean reduction 2.9% and 1.9%).
The station-clustered intervals crossed zero, so this is directional
complementarity evidence rather than a general superiority claim. The next
discriminating run is one spatial holdout family with the same two analytes
and three seeds. Its first completed DOC configuration selected pure RF and
showed no blend gain, but the remaining configurations were not completed;
this is an exploratory partial result, not a spatial gate. Deeper GNNs and
broad hyperparameter search remain out of scope.
