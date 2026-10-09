# Independent DOC replication in HUC02040104

Use the fixed five-seed source deployment defined in
`doc_portable_source_fit_v1/deployment_protocol.json`. The full integrated
candidate remains primary. Retain the current full model, ordinary matched
trees, strong station-hidden trees and native-only ablation. No external
prediction outcomes have been examined when this specification is written.

Before scoring, replay every procedure on the saved source-validation cells,
save the mean-native-scale source support policies, and generate all external
point estimates from `inputs.npz` without reading external DOC. Store fitted
source completion identities, point components, source retrieval identities
and the full station-month prediction table. Then open the separate scoring
DOC dataset. K0 scores every valid DOC cell. The separate K curves share five
reserved support candidates and a fixed query acrossK0/1/3/5. Only designated
support values enter the explicit retrospective correction.

Select support alpha on source-validation ensemble query MAE from
{0,.25,.5,.75,1}; ties choose smaller alpha. For each procedure, primary K0
uses the finite-rank90% quantile of absolute log1p source-validation residuals
as a constant log half-width. Each K curve uses the corresponding adapted
source-validation query residuals. These are empirical validation intervals;
validation also selected the fitted recipe. Report interval coverage, median
and mean native width, Q90 coverage and station-clustered intervals. No external
DOC selects alpha, quantile, width, preprocessing, donor memory or model.

Report arithmetic-mean ensemble metrics and separate individual-seed K0
metrics, with5,000 paired station bootstrap draws using seed42. Use the common
source-training Q90 threshold. Primary comparisons are the fixed complete
upgrade versus the current full model and strong station-hidden trees.
Ecological novelty and nearest-source distance use frozen source scaling and
source-station thirds. Hydro availability uses0/1/2 observed channels at each
cell. Missing ecological inputs and empty bins are explicitly counted.

The external river graph is not constructed for this no-message recipe.
Named source-to-external river links are absent, so directional context is zero;
calendar-aligned source DOC global context and ecological profiles remain
available. This tests deployment in an independent basin, after retrospective
ST357 geography, without choosing the external case from prediction quality.
External outcomes are retained and do not retune this version. Future mechanism
development returns to the142/143/144 source roles.
