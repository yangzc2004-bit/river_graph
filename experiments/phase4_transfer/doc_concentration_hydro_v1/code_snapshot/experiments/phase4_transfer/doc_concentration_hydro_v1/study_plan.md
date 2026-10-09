# Source concentration–hydrology correction of the existing DOC model

The completed deployment shows an unresolved cross-environment residual/bias
problem. This new development version tests a small source-derived correction
on the existing environmental reference and observation-aware residual model.
Earlier geographical, independent-basin and temporal evaluations remain results
of their original version. They do not select this development mechanism.

## Question and fixed comparison

Can concentration regime and observable hydro conditions explain source-station
OOF errors, and does transferring that correction improve new-source-validation
stations? Compare a two-coefficient log-affine control with a ten-coefficient
concentration–hydrology correction. Give the strong environmental tree exactly
the same correction; retain the preceding and currently retained full models.

The conditional basis contains an intercept, source-standardized log prediction,
a source-Q75 hinge, temperature, signed log discharge, two hydro visibility
indicators, causal flow anomaly, previous-12-month flow support and a
concentration-by-flow-anomaly interaction. Continuous input coordinates are
bounded at three source SD; predictions are not upper-clipped. Its smoothed
native-concentration MAE uses fixed ridge .1 and smoothing .05. No coefficient,
knot or statistic uses receiving-station DOC labels. This version does not
add a free neural gate or alter the fitted GRU, ecological encoder or tree.

Fit each correction only on saved station-hidden source-training OOF tree
predictions and their source DOC labels. Validate the stored station-fold
exclusions first. Apply the calibrated tree correction to the matched native
and integrated model outputs, retaining their existing neural and ecological
corrections. This deliberately tests whether the corrections complement or
duplicate one another; the tree-only corrected arms isolate calibration value.

Use partitions142/143/144 × seeds42/43/44. The nine completed
`doc_unmonitored_residual_v1` packages provide the fixed initialization,
reference predictions and OOF inputs. New-station K0 contains no DOC, pH,
conductivity or their input histories. Only source training/validation roles
are materialized; the previous receiving-test role is not evaluated here.

## Products and interpretation

Save all five previous arms plus six correction arms (tree/native/integrated
for each correction mode), their fitted coefficients, aligned validation
predictions and normal prediction sidecars. Report MAE, bias, log error, Q90,
hydro availability, prediction-concentration strata and correction overlap.
Seeds are averaged inside a partition; partitions are equally weighted.
Use5,000 paired station bootstrap draws, keeping a station's months together
across overlapping development partitions. These are source-development results,
not a new independent confirmation or an isolated neural-memory contribution.

A useful correction must improve the retained complete model on source
validation and remain competitive with its equally calibrated strong-tree
control. A failed correction is retained as a mechanism result; no evaluated
geographical/external queries are used to repair its coefficients. Subsequent
source development should act on the observed source failure, rather than
enlarging the entire architecture at once.
