# U5 spatial blend pilot verdict

This is an exploratory early-stop record, not a complete confirmatory
experiment. The first frozen configuration (DOC, `e3_spatial_seed42`, seed 42)
completed with validation-selected `alpha_rf = 1.0`: the blend exactly equals
Temporal RF on the hidden test cells (MAE 2.7144), while the temporal graph
alone has MAE 3.0880. The result therefore provides no evidence that the graph
adds complementary information for this spatial holdout.

The remaining five configurations were not run because the spatial GNN
configuration was substantially slower than the strict temporal family and
the first completed configuration selected the RF endpoint exactly. The
partial product is retained under `runs/` with its U5 protocol hash and is
excluded from pooled U4/U5 claims. No model or endpoint was changed.

Decision: do not expand the blend to a broad spatial matrix. Treat the blend
as a targeted temporal-extrapolation diagnostic and keep Temporal RF as the
strong performance baseline for spatial missingness.
