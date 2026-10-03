# Next experiment: teach the support readout the current hidden coordinates

## Motivation

Fixed-readout refresh helps K3 ordinary-DOC reconstruction but does not improve
K5. The prediction trunk was optimized for native reconstruction, whereas the
frozen two-dimensional readout was learned for older support-episodic states.
Train the readout for the selected current state before adding another temporal
operator or changing the recurrent decay clock.

## Minimal model change

Start with the retained daily-head expert (`off`) across the same three
partitions and three seeds. Freeze its encoder, GRU, decay, scalar head,
forest base and ecological residual profile. Learn only a 64×2 support readout:
128 parameters per package, nine fits.

Initialize exactly at the saved v4 readout and include epoch0. Use the same
32 calendar anchors and station scalar-RMS normalization, with its existing
scale floor. Do not refit PCA, whitening or the hidden encoder; do not
QR-orthogonalize the initial map. Learning a 2×2 orthogonal rotation alone
would leave isotropic ridge predictions unchanged and would not test a new
support subspace.

A convenient parameterization is P=P0+||P0||F D, with D initially zero.
This preserves the initial projection exactly and gives a consistent learning
scale across saved readouts. Use Adam, learning rate 1e−3, 30 epochs and
patience5, without a hyperparameter sweep.

## Source support-episode training

Extract selected-model hidden states using the same station-fold-hidden
source inputs used by the native residual expert. The baseline at source
training cells is max(0, context_OOF+selected_scale×native_delta), evaluated
from that same view. Train against the remaining log1p residual using
K={3,5} and ridge={1,10}, alpha1; minimize native query MAE, averaged equally
over source stations and the four K/ridge combinations. Keep support/query
roles explicit in each episode.

The forest baseline is OOF, but the frozen neural expert has already trained
on source labels. These are source-training residuals, not fully OOF neural
residuals; training loss is not transfer evidence. Validation uses genuinely
held source-validation stations, the full-source forest and the unchanged
selected expert. Select the readout checkpoint from the same four validation
episode losses, retaining epoch0 as a candidate.

## Comparisons and outputs

Compare constant, carried legacy, refreshed-fixed and refreshed-learned bases.
Native predictions remain fixed. Refit the existing downstream alpha/ridge and
positive-K ecological mixing grids only on source validation. Report all
direct/integrated K=0/1/3/5 curves. K0 and K1 must remain unchanged; the
mechanism comparison is learned versus fixed refreshed readout at K3/K5,
with legacy as the overall model reference.

Save source episodes, readout checkpoints and traces, readout distances,
anchor/floor statistics and adapted predictions. Evaluate ordinary and Q90
error together, including station/partition consistency. A readout improvement
would establish better support-adaptation geometry for this frozen expert;
it would not establish a new river-message effect or a general forecasting
claim. If it fails, retain the main candidate and return to the recurrent
clock comparison described in the preceding memory experiment.
