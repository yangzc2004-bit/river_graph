# River information under nested station withholding

Date 2026-10-08. The geographical comparison is complete: all added readouts
selected zero. Inspection showed zero or one supported calibration station in
four regions, making many station-CV contrasts exactly unidentifiable. This
extension is recorded after those results, before its correction fits.

## Four procedures and fixed inputs

Retain unchanged current predictor, simple upstream correction, river-path and
whole-form conditioned correction, and availability-matched non-ancestor
correction. Use the same measured geometry, source OOF departures, lags, ages,
matching and regularization as the geographical comparison. Do not select a
different lag, feature subset, procedure or output scale from scores.

## Receiving station roles

Reuse saved source-validation predictions for splits 142, 143, 144 and seeds
42, 43, 44. Each receiving station was outside the base gradient population.
Those labels previously selected the base model; this is retrospective
development, not independent confirmation or external validation.

Split the receiving stations into three outer folds. Shuffle supported and
unsupported groups separately using the split number, assigning round-robin.
Grouping uses source observation availability, never receiving concentrations.
All receiving stations remain outside the donor bank. Train each correction
on two outer folds, selecting regularization by inner three-fold station CV,
and predict the third. No held outer station label enters its correction fit.
Save every fitted state before assembling and scoring the outer-query panel.
Nine packages yield 27 held-station blocks and 108 four-procedure scores.

## Evaluation

Report all-observation K0 MAE, transformed MAE, RMSE, bias, station-equal loss,
Q90 error and recall, with split means equally weighted after seed averaging.
Use 5000 paired station bootstrap draws jointly across overlapping partitions.
Compare structure with current, simple and non-ancestor controls, and retain
all form/support/distance strata. Receiving stations, not months or training
seeds, supply independent resampling units. All positive intervals describe
this selected source-development population and require later confirmation.

The original geographical zero corrections remain visible. This extension
tests learnability with distributed receiver support; it cannot replace a
geographical success or establish that a shape causes DOC concentration.
