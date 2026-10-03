# Ecological transfer of DOC residual information

The current state-dependent flow model improves station transfer, but part of
the high-DOC error is associated with different station concentration regimes.
This iteration asks whether residual information from ecologically similar
source stations can improve the existing model at stations with no DOC labels.

Retain the selected environmental context, spatial encoder, observation-aware
GRU, discharge interaction and support-adaptation basis. Fit a compact competing
residual memory from source DOC minus the saved station-blocked OOF context
prediction. The memory estimates the same context correction as the recurrent
branch; they are mixed rather than added:

    prediction = max(0, context + (1-gamma)*(interaction-context) + gamma*memory)

Gamma zero exactly preserves the current interaction prediction. Gamma one
replaces the temporal correction with transferred residual information.
All choices use fixed source-validation K0 query MAE.

## Matched scientific controls

Fit a two-by-two set: global versus ecological donor selection, each with an
intercept-only or concentration-dependent affine residual. All donor stations
contribute equal total weight; observations within each station have equal weight.
The global arm uses all source stations. The ecological arm uses the nearest
20, 40 or 80 eligible stations; the receiving station is always excluded.

Use the nine watershed ecology columns (regime indices 4 through 12): land
cover, precipitation/temperature normals, soil organic matter, elevation and
baseflow index. Fit median imputation and IQR scaling on unique source stations.
Sentinel -1 and nonfinite values are missing. A source donor or target needs
at least five valid columns for local ecological transfer; otherwise use the
global station-balanced donor pool. This is static retrospective ecological
context, not a reconstructed monthly process state.

Affine conditioning is source-station-balanced standardized log1p(context
prediction). Residuals are normalized by source-station-balanced mean absolute
OOF residual. Fit smooth absolute error with epsilon 0.05 in normalized units
plus coefficient ridge, lambda in {0.1, 1}; the ridge center is zero correction.
Optimization is deterministic float64 L-BFGS-B from zero. No tail weighting is
added. The bias model fixes the slope to zero.

Select donor count, ridge and gamma in {0, 0.25, 0.5, 1} using source-validation
overall K0 MAE. Ties prefer gamma zero, stronger ridge, then more donors. These
choices are made before outer-query labels are attached. Keep all four arms;
the ecological-versus-global comparisons distinguish similarity information
from generic concentration calibration.

## Data and evaluation

Use the existing three station partitions 142–144 and seeds 42–44: 36 residual
memory fits on nine frozen expert packages. The original context OOF forest
predictions are genuinely station-held-out. The existing neural model is frozen
and is not presented as independently OOF. No in-sample neural residual is used
to train this memory.

Keep K={0,1,3,5}, fixed query cells, the reserved target support observations,
the frozen v4 GRU basis and the existing validation alpha/ridge grid. Recompute
support residuals against each new base prediction. The matched unchanged
interaction product must remain bitwise equal to the saved reference.

Report raw/log MAE, RMSE, R2, Q90 and ordinary-concentration error, bias, false
Q90 rates, all K curves, station gain/loss and source-donor diagnostics. Use
the existing seed-averaged, partition-equal estimand and 5,000 joint station
bootstrap draws. Distinguish ecological similarity gains from the affine
conditioning gain, and report inactive gamma-zero memories honestly.

These partitions have been examined in previous iterations. This is continued
model development; all arms remain available regardless of their ranking.
Test results do not choose the next default model. The support-adapted product
remains retrospective reconstruction.
