# R9 observation-gated transport pilot

The `msggate2` variant keeps the M1 two-layer spatial encoder but multiplies
the directed message sum by a learned node/month gate before the local path
and message are added. DOC temporal holdout, seeds 42--44, 30 epochs,
patience 5.

| arm | MAE |
| --- | ---: |
| M1 baseline | 1.142702 |
| observation-gated messages | 1.182934 |

The gate arm is 3.40% worse than M1 in the seed-mean comparison; the paired
station bootstrap CI for the MAE increase is `[-0.046, 0.158]`. All three
seeds are worse in point estimates, but the interval crosses zero. This pilot
does not support expanding the gate to the spatial holdout.

Together with R5, R6, R7 and R8, the evidence favors a composite temporal
predictor (RF-like local features plus a river-message GNN) over additional
GNN depth, scalar message residuals, or a free node/month message gate. The
river-message blend remains the useful result to carry forward; its spatial
holdout behavior is still RF-dominated.
