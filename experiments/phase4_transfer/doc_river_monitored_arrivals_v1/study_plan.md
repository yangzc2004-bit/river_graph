# Whole-network monitoring coverage and observed DOC overlap

## Question

Do the sampled tributaries represent the whole-network arrival structure, and
do their measured DOC fluctuations coincide at the monthly observational scale?
Keep the original elongated, broad and mainstem-dominated outline classes.
Connect whole geometry to actual source/outlet observations before attributing
field peak attenuation to the controlled routing mechanism.

This is an exploratory follow-up after seeing the preceding routing and monthly
mixing results. The new elements are a non-overlapping, potentially multi-gauge
upstream frontier and its coverage of the **complete** mapped receiving network.
It does not repeat the previous 59 two-source connection prediction fits.

## Population and information

- Retain all 297 original station-network instances in the coverage inventory.
- Use only the existing source-training cell union of roles 142/143/144.
  Reconcile the saved cell list with the three masks before reading values.
- Keep all 295 physical receiving COMIDs in the inventory. For observational
  summaries, choose one receiving station per COMID by allowed-month count,
  then station ID; preserve the other stations as labelled aliases.
- Exclude the existing gross drainage-area mapping mismatches from gauge inputs;
  retain unknown reported areas as unknown. Require at least 24 common months,
  three sampled years and six distinct calendar months for an eligible source
  set. These are availability criteria, never DOC-value criteria.
- Among aliases on an upstream COMID, choose greatest overlap with the receiver,
  then station ID. Exclude receiving-reach aliases from upstream candidates.

## Whole-network source selection

Use the original geometric-mainstem-conditioned routing definition from the
preceding study as the main analysis and the saved shortest routes as a separate
sensitivity. Both definitions were inspected previously; neither is new physical
flow measurement. Off-trunk successors stay fixed; no dense inferred graph.

1. Accumulate each unique incremental catchment area exactly once.
2. Form an eligible gauge frontier: remove a gauge when another eligible gauge
   is downstream of it. Frontier catchments cannot overlap on the selected tree.
3. Choose the two frontier gauges with the largest combined area that meet
   common-date requirements. Add remaining gauges in decreasing area order
   whenever the same requirements remain satisfied. The procedure is greedy,
   not a claim of globally optimal monitoring coverage.
4. Reject known summed official upstream drainage greater than 1.1 times the
   receiver's official area. Unknown official areas do not count as verification.
5. Keep the selected set fixed through time. Never replace an unavailable gauge
   month by another branch, interpolate DOC or pick a set by downstream fit.

Report covered incremental area, number of independent inputs, effective input
count, coverage of lateral-unit area, and covered versus complete path variation.
Source weights are covered catchment-area shares, not measured instantaneous
flow shares. COMID catchments extend to reach ends; source-to-receiver gauge
distances are cropped using the saved NHD measures and reported separately.

## Observed signal analysis

Use exactly common months of the selected sources and receiver. Remove a common
intercept, annual sine/cosine and linear-year design in native DOC concentration.
For fixed area weights, mixture residuals equal the weighted source residuals.
Retain their exact covariance decomposition:

`weighted source variance - mixture variance = amplitude/balance component + asynchronous component`.

Report source coherence, mixture buffering, asynchronous contribution,
source-mixture/outlet correlation and outlet/mixture residual SD ratio. These
are fluctuation summaries; they do not estimate DOC loss or event travel times.
High-DOC coincidence is descriptive: use each station's Q90 on the same common
months, report counts of >=2 high sources and receiving high samples, retain
empty/small conditional denominators explicitly.

Use 200 independent within-calendar-month permutations of source values as a
timing diagnostic. Re-project each permuted series onto the same calendar design.
Keep the receiver, weights and dates fixed. Shuffles are descriptive controls,
not causal forecasting inputs. No delay is optimized on receiving outcomes.

Reuse the raw WQP acceptance rules and rebuild activity means on permitted
source months. Reconcile result-weighted raw monthly means with the dataset.
For each common month, choose one activity per station by minimum calendar-day
span, then distance from the receiver date and deterministic metadata ties.
Selection never reads DOC. Show cuts of 0, 1, 3 and 7 days, each with its own
coverage counts. Compare selected activities and monthly means on identical
dates and fixed source sets; require the same availability criteria for summaries.
These cuts assess sampling alignment, not a sequence of event travel times.

## Aggregation and interpretation

One vote per unique physical receiver, after computing its fixed-set statistics.
Observation systems join receivers whose complete mapped catchments overlap;
shared upstream gauges also join systems. Resample complete systems 5,000 times.
Retain HUC4 sensitivity and per-form counts. A one-system form has no population
interval; more samples at that system do not create independent rivers.

Report descriptive geometry/signal correlations, their system intervals and
omitted-system range. No mechanism or model is selected by these associations.
Keep full versus sampled arrival geometry together so a partial frontier cannot
be mistaken for a complete test of whole form.

Report the unadjusted broad-minus-elongated difference using the same shared
system resampling draws for both forms. This is an exploratory class comparison,
not environmental matching or a causal shape effect. Do not infer a difference
merely by comparing separate within-form intervals. The two mainstem-dominated
receivers remain explicitly small-sample descriptive cases.

## Outputs and continuation

Coverage inventory, candidate/selected gauges, covered geometry, monthly and
dated records, signal summaries, routing-definition sensitivity, and real-map
and observation figures in English/Chinese. Save a short English research
decision and runnable analysis/plot/verification entry points. Historical model
results and classifications remain unchanged; no neural training.

Use the coverage map to identify the observation systems that can test
**whole-form pathway arrangement -> source overlap -> outlet DOC fluctuations**.
Event peak/width measurements require synchronized sequences shorter than the
candidate transit separation; the existing sparse monthly records are not
converted into artificial continuous event curves.

## Follow-up from the monitoring-coverage inventory

After the first coverage calculation, one system supplied all 81 common months
with at least three sampling days at every selected station. The station names
identify two Loch Vale tributaries and the outlet of The Loch. Add a separate
weekly-resolution case, selected by this sampling-density metadata, then receiver
ID. This addition follows inspection of the availability results; it does not
change the 32-network primary summaries or select a case by DOC fluctuations.

Use only the already permitted common months. On each exactly common calendar
day, select one activity per station by earliest known UTC timestamp, then
activity ID. Preserve the original DOC activity means and show a sensitivity
using all accepted activities' same-day mean. No activity or calendar date is
reused. Report actual UTC spans separately: same calendar day is not simultaneous
sampling. Compare calendar/year-adjusted fluctuations and within-month deviations;
the latter use months with at least three common sampling days. Independently
shuffle sources within the same year-month for the within-month diagnostic.

Save daily selections, sampling counts, both signal adjustments and a year-by-year
ledger. A figure year is selected by most common sampling dates, then earliest
year. Show actual sampling points and connections only when consecutive samples
are at most 14 days apart. These are sampled sequences, not reconstructed event
curves. This is one lake-influenced receiving system, not additional independent
replication of river-form classes; it cannot by itself resolve event transit,
DOC removal or the effect of lake retention.
