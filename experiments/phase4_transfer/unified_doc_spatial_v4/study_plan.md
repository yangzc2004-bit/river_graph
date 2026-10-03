# Learning recurrent states for few-shot DOC station reconstruction

## Question and comparison

Version 3 trained a projection of frozen GRU states and obtained essentially no
additional held-station gain. This experiment updates the existing GRUCell and
observation-decay parameters through the support-to-query objective. Spatial
encoding, environmental forests, the original temporal prediction used in fusion,
and version-2 fusion coefficients remain fixed. The updated recurrent state is
used by the support adapter, not substituted into the frozen prediction head.

Use the same nine DOC packages: station partitions 142, 143 and 144, each with
training seeds 42, 43 and 44. These previously examined partitions provide a
development comparison within ST357. K = 0, 1, 3 and 5 use the existing nested
support schedule and fixed queries, with all five reserved dates removed from
queries at every K. Support can follow query dates in retrospective reconstruction.

## Model and source episodes

Cache the existing frozen spatial encodings, observation ages and support
features. Source stations use their saved station-fold label-hidden input view;
validation and target stations use the training-only view. Load version-3
station-blocked OOF environmental predictions for source residual targets.
Pretrained expert weights were source-trained, not independently fitted per fold.

Initialize from the existing GRU and decay weights. The fixed two-dimensional
readout is the version-3 source PCA whitening matrix followed by its selected
supervised GRU projection. Only GRUCell and decay parameters are trainable.
Each station-month uses its own causal 12-month rolling window with zero initial
state; left-padding months perform no recurrent update.

Normalize the two readout coordinates with a station-specific scalar RMS after
centering on 32 evenly spaced calendar anchor months, or all months when fewer
than 32 exist. The floor is 1e-4. Anchor selection never uses DOC values or target
observation dates. Apply exactly the same normalization to frozen and updated
GRU states. This prevents uniform feature inflation from weakening the ridge
penalty. The normalizer can use later covariates; the complete adapted output is
retrospective, even though each recurrent window respects temporal order.

During training, evaluate only unique observed and anchor windows. Sample nested
K = 3 and K = 5 supports using the version-3 chronological-bin schedule, refreshed
each epoch. All five dates are excluded from queries. Differentiate through the
same centered ridge solve at lambda = 1 and 10 and level shrinkage alpha = 1.
Optimize native DOC query MAE, averaging K/lambda equally and source stations
equally. Source stations require at least six observed values.

Use Adam with learning rate 1e-4, station batch size 8, gradient norm clipping at
1, at most 30 epochs and patience 5. Select the checkpoint by pooled source-
validation query MAE averaged over the same K/lambda settings. Include the
unchanged initial GRU as epoch 0. A two-epoch smoke run checks execution only;
its outer-query metrics will not select a training configuration.

## Final prediction and controls

After checkpoint selection, retain the version-3 adapter selection grid:
alpha in {0, .25, .5, .75, 1}, lambda in {.1, 1, 10, infinity}. Fit adapter choices
on source validation only. Use the saved held-station fusion coefficients for
both validation support and query predictions. Target DOC enters prediction only
through the explicit K support observations.

For both environmental and frozen-fusion bases, save these five adapters:

1. Constant level correction, unchanged from version 3.
2. Version-3 supervised GRU projection, unchanged.
3. Version-3 supervised tree projection, unchanged.
4. Frozen GRU with the new calendar-anchor normalization.
5. Updated GRU with that identical normalization.

The direct effect of recurrent training is comparison 5 versus 4 at K = 3 and 5.
Also compare 5 against 2, 3 and 1 at K = 3 and 5. Comparison 4 versus 2 reports
the normalization change separately. The tree control has the same support
budget and two-dimensional output, but fewer trained parameters; it is not a
parameter-count-matched recurrent control. Report actual parameter counts.

## Analysis and decision

Report MAE, RMSE, R-squared, log MAE, source-training Q90 MAE and complete K curves.
Average seeds within partition and weight partitions equally. Use 5,000 joint
whole-station bootstrap draws for paired comparisons, maintaining repeated
station identities across partitions. Report partition and seed directions.
All comparisons are reported; outer tests do not choose a checkpoint or grid.

Check finite training gradients, actual recurrent parameter change, selected
epochs, and complete model reload. Verify original controls are bitwise unchanged.
Preserve previous versions and manuscript claims. Judge the value of this update
from recurrent-trained versus frozen-normalized performance first, then from its
performance against the existing best adapters. If it adds no clear value, report
that result without searching additional hyperparameters on the same test panel.
