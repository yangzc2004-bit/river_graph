# Same unified DOC model: spatial contribution and external replication v2

Status: prospective protocol, written before new training and target scoring.

Version1 stopped at its first validation-table integration because the shared scorer requires a high_doc column. No completed selection, source-test scoring or external-field prediction exists for that attempt. Version2 fixes the table schema only (high_doc=False; no tail endpoint is claimed), starts all fits afresh, and preserves the original frozen attempt. Model, masks, hyperparameters, endpoints, architecture decision and receiving-data policy are unchanged.

The candidate remains the jointly trained environmental MLP + 12-month hydrological GRU + upstream/downstream/global three-head graph Transformer. The ecological-forest/local-anchor correction branch is excluded. Adapter forward results and initialized parameter tensors must exactly reproduce the existing direction_heads implementation.

## Matched spatial controls

Four arms share all 12,565 parameter tensors, environmental and temporal inputs, loss, optimizer, month sampling and validation stopping. direction_heads uses real upstream/downstream/full paths. generic_dense uses self/other category biases and all-node attention. self_only retains every spatial projection, feed-forward/residual self path, and static river covariate but zeros other-node visibility. permuted_river relabels the permitted-node topology: retain category counts and degree distribution, select the lowest actual-connected overlap among 100 fixed topology-only permutations, and disclose remaining overlap. This null is not a strictly unconnected donor control.

Use the user's previously selected station masks 142/143/144 and optimization seeds 42/43/44: 36 fresh fits, no fitted DOC anchors. Fit preprocessing on source stations only. Each source graph includes source stations only; validation/test nodes join source exogenous context during inference. Omitted stations may establish known paths but their features and labels cannot message. Epoch selection uses source validation native MAE, with inherited learning rate .001, weight decay .0001, maximum60 epochs, minimum15 and patience10. Two CPU threads per fit change execution resource allocation only.

## Selection before external scoring

Joint primary endpoints remain native MAE and log1p-MAE. For this new protocol, average paired validation ratios relative to self_only across nine panels. A real or generic spatial arm is eligible only when both ratios are below1 and their equal-weight joint score is at most.99. Select the lowest eligible score; otherwise retain self_only. The null is ineligible. This new explicit 1% practical threshold addresses whether spatial messaging deserves inclusion, unlike the prior temporal/spatial recipe ranking. Existing paper endpoints and prior claims remain unchanged. The primary deployable checkpoint is the selected arm at split142 seed42, fixed before results; no test-selected seed or prediction ensemble.

## Actual cross-river K0 test

Apply every source checkpoint, without adaptation or DOC input, to the 130-station Middle Delaware-Mongaup-Brodhead receiving network (HUC02040104). Name verified against [USGS](https://www.usgs.gov/data/delaware-river-basin-catchments-based-strahler-stream-order-1-national-hydrography-dataset-nhd). Existing label-free inputs and scoring labels have separate archives. Source station overlap is zero. Reconstruct the receiving station graph from cached NHDPlus VAA primary downstream links and NLDI positions; do not impose within-reach order when measures tie. All130 stations have positive stream order. The previously unknown fourth regime feature is recomputed as zero station in-degree, matching the source feature's semantics, before training; this is not a physical headwater claim.

Receiving inference uses its own130 nodes only, with frozen source normalization. Seal all36 source validation fits and the chosen architecture, then generate and seal all36 full130x520 label-free receiving fields, then open external DOC for evaluation. Score all6514 observed external cells, including cells another protocol allocated to support: this is K0, with no support used. Record prediction/protocol/data/mask/model/statistics hashes and compare exactly the same queries in all arms.

The external case has already been inspected by other architecture releases, so it is a replication case for this unified model, not a new blind external cohort. One external network cannot establish global generalization. Random source-station splits test spatial completion and do not constitute cross-river tests. No extrapolation to all physical reaches or continuous-time prediction is validated.

## Reporting and audit

Report the arithmetic mean of nine fit-level MAE, log1p-MAE, RMSE and R2, station-macro MAE and coverage; do not ensemble predictions or treat nine fits as nine independent rivers. Report the prechosen primary checkpoint separately. Recompute all metrics independently and replay saved weights on source and receiving queries. Paired station-block bootstrap intervals are exploratory within these fixed datasets; source partitions overlap and spatial correlation may remain. Report external input availability and distribution shift, and retain an unsuccessful transfer outcome without revising the preselected architecture on its test labels.

Deliver an evidence-based architecture choice and its tested scope. If spatial arms or transfer fail, do not declare the full three-source model universally superior merely because it was the initial candidate.
