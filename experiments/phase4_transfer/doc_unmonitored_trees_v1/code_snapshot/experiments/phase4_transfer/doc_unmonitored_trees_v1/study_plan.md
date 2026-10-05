# Matching source tree inputs to entirely unmonitored stations

This source-validation diagnostic isolates the **source input regime**. The
existing tree fits can use prior local DOC at source stations, although K0
queries have none. Train a comparator using station-fold-hidden feature views
for every source target row. Fold mates' DOC also remain hidden in the network
summaries. All source labels still serve as regression targets.

Keep monthly ecology/hydro/time, the eight daily hydro descriptors, target
log1p transformation, 300 trees, the four existing leaf/feature candidates and
validation MAE selection fixed. Partitions 142/143/144 and seeds 42/43/44 use only
source training/validation roles; all outer target roles remain unevaluated.
Compare with the current complete model and the existing matched-input trees.
There is no new attention mechanism, loss weighting or feature expansion.

This diagnoses whether a strong tree base has been trained for the intended
zero-observation information regime. If it helps, rebuild the **existing**
ecology/GRU residual model on station-OOF predictions from that base and check
the learned source transfer. Do not infer a new neural contribution from a
tree-only improvement. Keep the old tree and full model as fixed references.
