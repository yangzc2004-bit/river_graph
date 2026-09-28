# R5 status: complete

All four spatial variants completed 6/6 runs (24/24 total). The compact paired
summaries are stored under each `*_vs_m1/` directory.

| variant | temporal MAE | spatial MAE | temporal vs M1 | spatial vs M1 |
|---|---:|---:|---:|---:|
| M1 baseline | 1.143 | 3.490 | — | — |
| Res2 | 1.258 | 3.507 | -10.1% | -0.5% |
| Res3 | 1.257 | 3.666 | -10.0% | -5.0% |
| Res4 | 1.260 | 3.660 | -10.3% | -4.9% |
| Res3-JK | 1.218 | 3.456 | -6.6% | +1.0% |

The layer-wise fusion variant is the best residual model, but its spatial
interval still crosses zero and its temporal error is higher than M1. Extra
layers therefore do not solve the current GNN gap. The next model change
should target how messages are weighted or how the RF-like local baseline is
combined with graph residuals, rather than adding more spatial depth.

Q90 values are included in each paired table; the temporal DOC tail has only 12
query cells and is marked unstable.
