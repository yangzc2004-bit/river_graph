# River form and monthly DOC: same-calendar comparisons

## Question and relation to the existing work

Compare DOC level, temporal variability and high-value frequency between the
fixed elongated, mainstem-sparse and broad tributary-rich river forms. Keep
river morphology central. Environment, basin area and measured monthly hydro
provide the comparison context.

Earlier station-summary and pulse-routing results have been seen. This is an
exploratory extension of the source-role study, not a preregistered independent
confirmation. Unlike a comparison of each station's own observation calendar,
this version compares matched rivers in **the same calendar months** and uses
held-region environmental/hydro backgrounds to describe remaining differences.
Existing frozen results and released DOC models remain unchanged.

## Fixed population

Use the existing 297-station morphology panel, geometry-only class definitions,
and the exact covariate-selected, non-nested pair list from
`doc_river_morphology_effect_v1`. Do not rematch on DOC, dynamic hydro, or a
response effect. DOC is restricted to the existing permitted source-training
union of station roles 142/143/144. No geographic/external model-test
predictions are inputs.

An availability-only audit found:

- Primary common-DOC population: at least 12 common calendar months and six
  distinct months-of-year. Class1/class3: 20 pairs in 11 HUC4s.
- Measured-hydro population: the same criteria, additionally positive measured
  discharge and measured temperature at both receivers. Class1/class3:
  17 pairs in 10 HUC4s.
- Longer-record sensitivity: at least 24 common months and six months-of-year.
  Class1/class3: 13 pairs in seven HUC4s.
- Class1/class2 and class2/class3 each have only three eligible pairs in one
  HUC4. Retain descriptive results without a geographic confidence interval.

Month indices must align to a unique continuous calendar. No missing DOC
months are filled. Each pair uses exactly the same dates for its two members.
Availability exclusions and all candidate pairs are retained in a ledger.

## DOC responses

Primary comparison: broad (class3) minus elongated (class1). All other fixed
class contrasts remain visible. Record the two member values and the paired
difference, giving each pair equal weight after averaging within its months.

Responses, all retained:

1. Native DOC mean and median (mg/L).
2. Native DOC standard deviation, coefficient of variation and IQR.
3. Mean and SD of log1p DOC.
4. Observed high-DOC frequency using the **existing fixed 10 mg/L threshold**;
   report counts and denominators, not individual event peaks.
5. Mean and SD of log1p DOC deviations from the cross-fitted environmental
   background, and from the cross-fitted environment+hydro background.
6. High-DOC frequency minus background-predicted frequency, for each background.

The same months reduce calendar mismatch; they do not imply simultaneous grab
sampling or a measured event pulse. Within-station SD of background deviations
describes residual variation; it is not a directly estimated buffering rate.

## Source-only background fits

Use all permitted observed months at the fixed 297 stations, with five
HUC4-blocked folds defined on stations. Fit only on other HUC4s for every
receiving station. Each training station has total fitting weight one.

Environmental numeric inputs: log1p measured basin area, wetland, forest,
agriculture, urban, precipitation, climate temperature, latitude, longitude,
monthly sine/cosine and linear calendar year. HUC2 indicators are fitted from
source training regions only. Current DOC and its history are not inputs.

The hydro background additionally uses log1p positive discharge (original
dataset cfs) and observed temperature, with missingness indicators. All missing
numeric medians and weighted scales are fitted on source folds only. Ridge
alpha10 predicts log1p DOC. Logistic C1, max_iter1000 predicts DOC >=10 mg/L.
No class or shape inputs enter these two adjustment backgrounds.

Background deviations are descriptive conditional associations. They do not
identify a causal effect, erase unmeasured environment differences, or turn
missing hydro into measured hydro. The complete-hydro comparison is reported
alongside the larger common-DOC population, not selected by its result.

## Additional information in river form

On the identical station-month population/folds, compare six fixed Ridge arms:

1. Environmental calendar background.
2. Environmental calendar + measured hydro background.
3. Hydro background + the fixed form labels.
4. Hydro background + branch organization (log drainage density, mainstem share).
5. Hydro background + paths (sinuosity, normalized mean path, distance CV).
6. Hydro background + all seven footprint/branch/path variables defined in the
   preceding morphology study.

This tests incremental monthly information beyond environment and hydro. It
does not train a new neural reconstruction model or measure a gain over the
full released model. No operator, class, feature block or parameter is chosen
from the held-fold score. Report native/log MAE with station-equal averaging,
and all paired information gains.

## Analysis and outputs

Use 5,000 HUC4-block bootstrap draws, seed42, retaining paired rivers and their
months together. Confidence intervals with one HUC4 are unavailable. Intervals
summarize geographic sampling variation conditional on the fixed fitted
backgrounds; they do not bootstrap refitting or adjust for all related outcomes.
Report omitted-HUC4 paired effect ranges and coverage/hydro balance.

Outputs: shared-month records, source-fit states and predictions, all paired
responses/intervals, information comparisons, English research decision and
English/Chinese scientific figures. Preserve prior matched-pair assignments,
earlier morphology findings, and pulse experiments.

Scientific reading: identify which observed DOC properties differ between
forms, whether they remain after the common-calendar/hydro comparisons, and
whether measured internal structure carries information that the coarse form
labels omit. A lack of clear class differences is retained as a result rather
than prompting a new classification.
