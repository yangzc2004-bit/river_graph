# Existing DOC residual with a zero-observation-matched environmental base

The station-hidden tree experiment reduces its matched-tree reference MAE by
4.968% (95% interval 3.059–7.026%) on the selected source-validation panel. It
still trails the current full predictor by 2.010%. This is a useful base change,
not a completed neural model improvement.

Rebuild the **existing** ecology encoder/observation-aware GRU native residual
on this tree base. Keep the current monthly/daily inputs and interactions,
native-MAE loss, source-Q90 tail weight 2, zero readout initialization, 30 epochs
and patience 5. Retain the fitted current ecology/GRU initialization; do not add
layers, windows or a new attention mechanism.

The five station-blocked OOF base forests exclude the held fold from both their
regression targets and every training feature view. Their remaining training
rows also have whole-station-fold-hidden local water-quality history. Tree
hyperparameters are those fixed in the preceding source-validation experiment.
DOC residual source inputs use the existing station-fold-hidden views. Historical
raw preprocessing is retained; the concentration readout normalization is refit
only from the new source OOF predictions. No auxiliary chemistry is used.

Compare old full model, old matched-input trees, station-hidden trees, new native
residual, and new residual + the unchanged ecological-memory method. Use all
validation observations for K0 in partitions 142/143/144 and seeds 42/43/44.
Only source training/validation roles are available. Report 5,000 paired station
bootstrap intervals, Q90, hydro availability and ecological strata. A reliable
candidate can proceed to the five already saved HUC4 geographical tasks; the
external DOC-only cohort remains prediction-blind until internal confirmation.
