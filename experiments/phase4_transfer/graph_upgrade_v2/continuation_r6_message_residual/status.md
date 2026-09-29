# Message-only residual pilot (R6)

This pilot tests a two-layer spatial trunk in the M1 observation-aware temporal
model. The self path is evaluated once per layer; directed upstream and
downstream messages are multiplied by a trainable scalar initialized at 0.1.
The variant is named `msgres2` and does not alter the released H2X or the
ordinary residual variants.

Configuration: DOC, `e2a_strict` (temporal holdout) and `e3_spatial_seed42`
(spatial holdout), seeds 42--44, 30 epoch budget, patience 5. The comparison
uses the same station-month query cells and the same paired station-cluster
bootstrap as the existing graph-upgrade analysis.

| holdout | M1 MAE | msgres2 MAE | msgres2 change | station bootstrap CI for change |
| --- | ---: | ---: | ---: | ---: |
| temporal | 1.142702 | 1.395947 | +22.16% | [+0.1287, +0.3974] |
| spatial | 3.490092 | 3.757086 | +7.65% | [+0.1330, +0.4648] |

Correction: percentage changes use M1 as the denominator, i.e.
100 * (msgres2 / M1 - 1). The original text used msgres2 as denominator;
the underlying predictions and absolute-error confidence intervals are unchanged.

The learned message scales stayed close to their 0.1 initialization (roughly
0.10--0.12 across layers and seeds). In this pilot, a small message residual
does not recover the gap to the observation-aware baseline; ordinary residual
depth is also not an improvement in the earlier R5 pilot. Scales remaining
near initialization do not demonstrate that the model learned to reject
messages: optimization and initialization are alternative explanations.
This result motivates testing RF/GNN complementarity before investing in
another message-weighting architecture.

The six run directories, sidecars, full grids, and the paired comparison are
under this directory. The comparison was generated with
`scripts/compare_graph_mechanisms.py`.
