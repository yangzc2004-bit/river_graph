# Chemical-state station calibration for DOC reconstruction

## Scientific question

The preceding nonlinear decoder lowered K0 cross-station MAE2.14%, while its
K5 gain did not persist. This experiment asks whether a handful of DOC residual
observations should be transferred across months according to chemical state,
alongside the existing recurrent-state coordinates. The preceding development
results have already been seen. DOC remains the sole output target.

Keep the trained nonlinear decoder, tree controls, native components,
ecological memory, original station partitions and reserved support/query
observations fixed. This step fits calibration parameters, not neural networks
or forests. Use partitions142/143/144 and seeds42/43/44.

## Chemical coordinates

Use the frozen decoder's learned eight-dimensional embedding of current-month
pH/14, log1p(conductance), and the two availability flags. Its source-only
normalizer and weights remain fixed. Fit two principal directions on all
auxiliary-active months of SOURCE stations. Center each source station's active
embedding to compute pooled within-station covariance. This fitting reads
chemical features and source identities, not DOC values or target-station data.

Whiten using the source eigenvalues, with eigenvalue floor1e-8; coordinates
with eigenvalues at/below the floor are zero. Apply the fixed source projection
around a single SOURCE-global active mean. Do not center a target station over
its full record: each target coordinate uses only that month's chemistry and
fixed source statistics. If both chemicals are absent, chemical coordinates are
zero. The original recurrent basis remains the same retrospective basis.

Compare three representations:

1. legacy: existing two-dimensional GRU basis.
2. masks_aug: legacy plus two PCs computed with auxiliary values zeroed and
   availability retained, using the same frozen chemical encoder.
3. chemistry_aug: legacy plus two PCs from measured chemistry and availability.

The two augmented representations have equal dimension. The mask control tests
whether a gain reflects measurement availability or simply added coordinates.
Fit separate source PCA states for the two modes. Their centering and projection
gauges differ intentionally; support-pair centering removes constant offsets.

## Calibration and fixed comparisons

Apply each representation to direct nonlinear chemistry prediction, its
ecological integration, and the fixed chemical tree. Keep the existing alpha
and ridge grids. Fit all parameters using source-validation station episodes;
score auxiliary-active queries, with identical fixed parent fallback elsewhere.
Retain the prior K0 ecological mix exactly.

A fourth representation, selected, chooses the lowest final active-validation
MAE among legacy/masks_aug/chemistry_aug separately for each pipeline and K.
Ties prefer legacy, then masks_aug, then chemistry_aug. Record all choices and
scores before evaluating target queries. This source-validation route is fixed
in advance; target outcomes cannot change it. Preserve all three ablation
curves. K0 and K1 must be bitwise unchanged across representations: shape
coefficients are zero without at least two support observations.

There are thirteen curves: three pipelines × four representations, plus the
retained general point/ecological model. K0/1/3/5 use unchanged nested supports
and fixed query cells. Positive-K support is retrospective station calibration.

Primary comparisons are fixed at K3/K5. For each of three pipelines compare
chemistry_aug versus legacy, chemistry_aug versus masks_aug, and selected
versus legacy (18 contrasts). Also compare selected integrated neural model
versus the retained general model and versus selected chemical tree at K3/K5
(four contrasts). Total22. Report complete curves, MAE/RMSE/R2/log error,
Q90/ordinary errors and bias, classification tradeoffs, partition/seed directions
and auxiliary-availability groups. Use5000 paired whole-station bootstrap draws,
equal partition weights after seed averaging, and joint station multiplicities
for overlapping partitions. No test-selected representation or K splice.

## Products and interpretation

Save source PCA definitions, coordinates, calibrators, final validation scores,
representation choices, unchanged native grids and final query products.
Every parent curve, K0/K1 invariance and absent-chemistry fallback must reproduce.
Independently replay the frozen chemical embedding, source PCA and calibration.

Any gain is chemical-state-conditioned retrospective calibration. Same-month
chemistry covers19.74% of genuinely DOC-missing cells. Reused station partitions
remain development data; a selected successful model needs a later confirmation
on fresh partitions. If the added coordinates do not help, preserve the accepted
K0 decoder and investigate support-to-query residual transfer directly.
