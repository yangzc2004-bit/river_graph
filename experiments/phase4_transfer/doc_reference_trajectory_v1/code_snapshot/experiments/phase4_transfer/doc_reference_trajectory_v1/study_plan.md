# Environmental reference trajectories inside the existing DOC GRU

## Research question

Source pH/conductance supervision learned those targets but worsened the actual
retained DOC model. Keep DOC as the representation-learning task. The current
native residual uses the environmental tree prediction at the current head;
its GRU does not receive the reference concentration trajectory. Can that
trajectory improve reconstruction at stations without water-quality history?

## Fixed comparison

Keep the retained station-hidden environmental tree/OOF residual target,
original ecology/self encoder, observation-aware64-dimensional GRU,12-month
window,38 current head features, existing interactions and ecological memory.
No chemistry value or availability is added to receiving-station input.

Reuse the retained complete model as the unchanged reference. Fit two matched
new arms: `reference_current` projects one source-standardized log1p tree
prediction only at the current GRU step; `reference_history` uses the SAME
bias-free Linear(1,64) at every valid causal window step. Both projections start
at zero, exactly retaining the old hidden state at initialization. Both have
64 additional parameters and the original training rates,30 epochs and
patience5. The native DOC objective and source Q90 weight2 are unchanged.
This separates better current-reference integration from historical information.
No independent temporal model, auxiliary warm start or unconfirmed feature
candidate is combined with this experiment.

## Dense source reference

The old OOF cache predicts only observed source DOC cells. Refit its five
station-hidden folds with the same cloned tree hyperparameters and source
training roles to predict every month of the held source stations. Remove the
held stations' DOC before constructing either training or inference features;
use the existing inner station-hidden feature construction for the remaining
source stations. Keep all observed-cell predictions/targets equal to the
retained OOF cache (allow only1e-12 numerical replay roundoff). Do not fill
missing reference months with in-sample predictions, future DOC or interpolation
of observed targets. Save fold forests and dense predictions for resumption.

Fit the reference mean/std on dense source OOF predictions only. Validation
history uses the fixed full-source environmental tree; receiving DOC never
enters that tree or scaler. Source preprocessing and water-quality isolation
remain the same as the retained version. Padding and future steps never receive
reference projection; at new sites the saved tree supplies the full trajectory.

## Source evaluation and next direction

Run142/143/144 ×42/43/44. Preserve prior products and use identical all-observation
K0 query cells. Compare each native/complete arm with the actual retained model,
strong trees and the other matched arm. Report MAE, Q90, bias, station/partition
heterogeneity and5,000 paired station draws. Inspect the figure and saved-state
replay, including old input compatibility, source-only reference scaling and
current-only versus causal-history influence. No old geographical or external
query chooses the candidate. If this DOC-aligned information integration helps,
fix its structure before geographical replication; otherwise keep the current
portable release and use the observed error pattern to choose the next mechanism.
