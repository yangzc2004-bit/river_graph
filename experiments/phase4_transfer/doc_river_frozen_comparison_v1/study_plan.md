# River structure information on a fixed DOC predictor

Date: 2026-10-08. Previous source and geographical scores are known.

## Question

Can upstream DOC and measured river structure improve the complete DOC model
at stations with no local water-quality observations? The comparison separates
upstream values, structural conditioning and matched non-upstream values.

## Four procedures

1. The saved complete `available_real_integrated` predictor.
2. The same predictor plus simple upstream departure aggregation.
3. The same predictor plus path and whole-network conditioned aggregation.
4. The same structural readout with matched non-ancestor source DOC.

The fixed complete ecological encoder, GRU, ecological retrieval and integration
are preserved. The new regularized graph readout operates after that complete
prediction. No backbone parameters or ecological fusion weights are changed.
This is an information experiment on a deep predictor, not a new end-to-end GNN.

## Regions and fitting roles

Use HUC4 1013, 1019, 0708, 1030 and 1101, with seeds 42, 43 and 44. In each
saved geographical package the target region remains excluded from source
training and donor banks. The cyclic-next validation region supplies residual
head calibration labels. Three-fold receiver-station CV selects regularization
from 0.1, 1, 10, 100 and 1000, including an unchanged-prediction option.
Refit the chosen readout on all calibration stations and score the target region.
There are 15 packages and 60 procedure results, including the reused reference.

Calibration predictions are out of the base's gradient training population,
but those validation labels previously selected base stopping and fusion. They
are not called complete-model OOF predictions. The final geographic target
selects neither the base nor the correction. These regions have historical
scores; this is retrospective geographical withholding inside ST357, not a new
independent external validation or an unbiased search-wide significance test.

## River inputs and matched support

Retain at most 20 nearest mapped upstream sources within 3000 km, lag months
0, 1 and 3, and explicitly aged last observations valid for at most 12 months.
Donor values are source-training DOC log1p departures from saved station-OOF
environmental predictions. Calibration and target labels never enter the bank.

Each real donor slot is paired without replacement with a source outside the
receiver's ancestor set in the complete mapped station graph, with no 3000-km
exclusion cap. Shared-COMID receiver/ancestor aliases are excluded. Matching
uses log drainage area and observation-availability overlap, not concentrations.
Retain the intersection of real/control validity and the same conservative
maximum age in both banks. The simple, structural and control procedures thus
have identical cell-by-cell support counts, lag counts and age inputs. Actual
path slots stay fixed in the control; they are fictitious associations for its
substituted DOC values, not claimed physical paths. Availability matching uses
the saved source observation schedule for this retrospective reconstruction.

The simple readout uses equal source means at each lag, age-weighted means,
receiver daily hydrology and the frozen reference scale. Structure adds eight
saved path descriptors, continuous whole-form descriptors and 50/200-km distance
decay bases. All procedures use the same allocated feature layout; structural
columns are zero for the simple arm. No intercept is fitted. Missing source
support or zero departures produces zero correction exactly.

Whole-form descriptors are fixed measured geometry: aspect, network axis ratio,
drainage density, mainstem share, sinuosity, tributary balance and confluence
position. Imputation and scaling use source training stations. Original form
classes remain diagnostic groups, without reclassification by DOC results.

## Endpoints and interpretation

Primary K0 is every saved valid target DOC cell, with three seed losses averaged
within each region and five regions equally weighted. Report native MAE, RMSE,
bias, log1p MAE, station-equal MAE, Q90 MAE/recall and tail counts. The separate
K=0/1/3/5 curve retains the existing fixed queries and support adaptation.

Use 5000 paired station bootstrap draws, seed 42, with each station's months
resampled together. Primary comparisons are each correction versus unchanged
base, structure versus simple, and structure versus matched non-ancestor.
Report all comparisons and regions; intervals are descriptive pointwise
intervals conditional on this retrospective experiment. Add original form,
supported/unsupported, nearest upstream distance and common-support strata.
No-source predictions must remain identical to the saved reference.

Independent structure value requires improvement beyond both simple aggregation
and the matched donor control on the held regions. Improvement only over the
base establishes usefulness of the full correction, not structure specifically.
If calibration selects zero, retain it as a result of this procedure; do not
claim that no possible graph architecture could help. Test results will not
choose another alpha, geometry subset, region, lag or model.

Complete all packages, replay predictions, test label isolation and causal
message indexing, inspect figures and write a final research decision. Preserve
old experiments and large local source-model caches.
