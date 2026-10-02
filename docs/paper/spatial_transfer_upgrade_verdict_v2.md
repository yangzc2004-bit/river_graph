# Spatial transfer upgrade verdict (development extension)

This note records the current spatial-transfer development results. It does
not replace the frozen Phase 0--3 paper endpoints.

## Current best result

The strongest improvement is a target-station support residual adapter on top
of the source-similarity ExtraTrees expert. The source expert is fitted with
the target stations held out. Five target DOC observations estimate a
station-specific log1p residual, which is shrunk by an alpha selected on the
internal spatial validation split. The outer query is fixed across K.

| adapter | K=0 MAE | K=1 MAE | K=3 MAE | K=5 MAE |
| --- | ---: | ---: | ---: | ---: |
| source expert | 2.4556 | 2.4556 | 2.4556 | 2.4556 |
| level calibration | 2.4556 | 2.3449 | 2.1879 | 2.1353 |
| support residual | 2.4556 | 2.3145 | 2.1590 | **2.0234** |

At K=5 the residual adapter reduces pooled MAE by 17.6% relative to the
paired K=0 query. It is 5.2% better than the earlier fixed-level adapter. A
station-clustered paired bootstrap gives a positive K=5 improvement interval
in the adapter comparison output. The support-shuffle diagnostic removes the
gain, showing that the improvement uses the target station observations rather
than only the number of support cells.

A nested source-pool sweep selected 40 source stations and alpha 0.75 for K=5;
its three-seed outer mean was 2.024, effectively identical to the existing
40-station residual adapter. Enlarging or shrinking the source pool is not the
remaining bottleneck.

## KGML integration check

The same support correction was applied to the existing five-seed KGML
full-grid predictions, with alpha chosen from the internal held-out stations
before the outer evaluation. KGML improved from an outer E3 MAE of about 2.823
at K=0 to about 2.544 at K=5. This confirms that support calibration also helps
the graph model, but the resulting model is still weaker than the source
regional expert plus support adapter (about 2.024). The graph-message variant
did not separate from the no-message KGML baseline in this spatial test.

A small support-aware regional gate was also tested using the existing KGML and
regional predictions. Internal station-heldout validation selected the
regional expert for both K=0 and K=5; the gate itself was not selected. The
outer regional score was 2.469, while the fixed K=5 residual adapter was about
2.020. Source regionalization and support correction are the useful
components; the first hard gate formulation adds no gain.

Value-blind support schedules were checked as a final low-cost extension.
Fixed evenly spread support remained best, but its common-query score (1.984)
uses a smaller query after taking the union of all candidate schedules and is
therefore not directly comparable with the original paired-query K=5 score
(2.023). Recent, seasonal and flow-quantile schedules did not provide a
credible improvement, so support scheduling is closed for this round.

The result is a spatial adaptation result, not evidence that a deeper GNN is
needed. The useful operation is learning a target-station correction after the
source model has supplied a regional prediction.

## What did not improve the model

* Adding clearly defined static geomorphology (`alt_va`, reach length and
  drainage area) worsened the nested-validation-selected outer MAE from about
  2.447 to 2.489. These fields are not carried into the next model.
* Robust, affine and seasonal residual calibrators did not beat the simple
  support-residual mean at K=1 or K=5. The affine candidate helped at K=3,
  but not enough to replace the fixed mean adapter across the curve.
* Strict cross-analyte source selection did not improve the selected outer
  result. Under the K-session policy, pH and conductivity labels were exposed
  only at the same target months used as DOC support. Validation selected no
  auxiliary profile and source pool 160; outer MAE was 2.5022. The earlier
  full-profile experiment used hidden target-station auxiliary labels and is
  not a valid zero-shot result.

These negative results narrow the bottleneck: the remaining spatial error is
not resolved by adding a few static descriptors or by making the source
distance more elaborate. It is mainly target-station domain shift and local
state that is absent before support observations arrive.

## Recommended next model step

Use the support residual adapter as the spatial-transfer layer of the current
Local--Transport KGML model:

1. keep the existing local/ecology and directed transport representation;
2. train the graph branch on source-domain residuals;
3. at a new station, estimate a small residual correction from K observed DOC
   cells;
4. report the local prediction, graph correction and support correction
   separately.

This gives the GNN a specific job: model the part of the station error that can
be transported across the river network, while the support adapter handles a
new station's local offset. The next experiment should compare this integrated
model with the fixed source expert plus residual adapter using the same paired
query. More GNN layers, geomorphology expansion and unconstrained attention are
not the next priority because their spatial-transfer gain is not established.

## Reproducibility locations

* `experiments/phase4_transfer/spatial_adaptation/fewshot_residual_paired_v1/`
* `experiments/phase4_transfer/spatial_adaptation/adapter_comparison_v1/`
* `experiments/phase4_transfer/spatial_adaptation/source_geomorph_v1/`
* `experiments/phase4_transfer/spatial_adaptation/fewshot_residual_methods_v1/`
* `experiments/phase4_transfer/spatial_adaptation/cross_analyte_source_v3_k_session_30/`
* `experiments/phase4_transfer/spatial_adaptation/regional_residual_product_v1/`
* `experiments/phase4_transfer/spatial_adaptation/adapter_comparison_v1/station_mechanism.png`

The product directory is the recommended source for the paper tables and
predictions. It keeps the full 2,531-cell E3 table while evaluating the main
K curve on the fixed 2,316-cell paired query. The mechanism figure shows that
the largest K=5 gains occur at stations with the largest zero-support bias.
