# Align station-support adaptation with the updated DOC recurrent model

## Scientific comparison

The completed daily-memory experiment uses a newly trained encoder/GRU for
its central prediction, but retains the older v4 `gru_tuned_anchor` basis for
station-support adaptation. Test whether synchronizing these representations
improves cross-station reconstruction with a few target observations.

Use the completed nine packages from `doc_daily_hydro_memory_v1`: three
station partitions and three seeds, with off/current-only/full-history experts.
Freeze all checkpoints, native predictions, context predictions and ecological
residual memories. Do not refit any neural model, forest or readout.

For each expert, compare three two-dimensional support bases:

- `constant`: the unchanged constant-only correction;
- `legacy`: the unchanged saved v4 recurrent representation;
- `refreshed`: the selected current expert's 64-dimensional hidden state,
  projected by the same saved v4 64×2 readout, with exactly the same fixed
  calendar-anchor mean and station scalar-RMS normalization.

Use the v4 checkpoint's anchor count and scale floor, expected to be 32 and
1e−4. No new PCA, feature selection or supervised projection is fitted.
Compute the current model's hidden states from its original train-visible
inputs. Its scalar residual-head extra channels do not enter these states.

## Support evaluation

Refit direct alpha/ridge choices and positive-K ecological mixtures on the
same source-validation episodes and grids. Lock K0 ecological gamma to its
existing source-validation choice; it does not depend on the support basis.
Evaluate K=0/1/3/5 at identical fixed query cells and nested supports.

Report all 18 products: three experts × three bases × direct/integrated.
The primary comparisons are refreshed versus legacy at K1/K3/K5 for each
expert and pipeline, totaling 18 contrasts. K0 predictions must be bitwise
invariant to basis choice. Constant and legacy outputs must exactly reproduce
their corresponding parent products. Report all source-validation curves
before analyzing target performance.

Use native/log-space MAE, RMSE, R², Q90 and ordinary error, recall and false-high
rate. Keep the same equal-partition estimator after seed averaging and 5000
paired whole-station bootstrap draws. Preserve partition/seed directions and
every fixed contrast; target outcomes do not select a basis, model, K or route.

Refreshed-vs-legacy changes the selected hidden representation and the
resulting source-validation support/mixing choices. It does not isolate a
new projection-learning algorithm. Direct comparisons are interpreted before
integrated comparisons. K0 central predictions remain the completed parent
models, rather than being improved by this postprocessing.

## Products and interpretation

Save raw projected states, normalized bases, anchor means/RMS/floor counts,
source-validation adaptation states, query predictions and parent bindings.
Replay the projection and normalization independently, refit validation
choices and reproduce all query products. Keep previous experiments intact.

Calendar anchors may follow individual query months. This uses the existing
retrospective, label-free hydrologic record normalization and reserved station
support; it is not a month-start forecasting representation. These are the
same previously examined development partitions, not a new validation cohort.

If refresh improves support adaptation, carry the synchronized representation
into subsequent model development. If it does not, retain the existing main
candidate and test the inherited recurrent decay clock next. No fixed
minimum improvement is imposed on this diagnostic.
