# U5 spatial blend pilot verdict

This is an exploratory early-stop record, not a complete confirmatory
experiment. The first frozen configuration (DOC, `e3_spatial_seed42`, seed 42)
completed with validation-selected `alpha_rf = 1.0`: the blend exactly equals
Temporal RF on the hidden test cells (MAE 2.7144), while the temporal graph
alone has MAE 3.0880. The result therefore provides no evidence that the graph
adds complementary information for this spatial holdout.

DOC seed 43 was started and interrupted before export; four configurations
were not started. The batch was stopped because the spatial GNN configuration
was slower than anticipated and the first completed configuration selected
the RF endpoint exactly. This stopping decision was made after seeing that
result, was not specified in the frozen protocol, and cannot establish the
absence of spatial complementarity across analytes or seeds. The
partial product is retained under `runs/` with its U5 protocol hash and is
excluded from pooled U4/U5 claims. No model or endpoint was changed.

Decision: the six-run spatial protocol remains incomplete. Keep Temporal RF
as the available performance baseline and reserve any spatial blend verdict
until the remaining configurations are completed. Do not interpret the early
stop as a failed six-run scientific gate.
