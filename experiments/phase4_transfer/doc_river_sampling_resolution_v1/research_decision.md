# Research decision: the sampling clock and river-structure mechanisms

## Main finding

**Asynchronous tributary variability survives closer sampling-date alignment.**
The next river-structure experiment should separate branch mixing from storage
within the common trunk. The archive supports a tributary variability study;
it does not provide an adequately resolved concentration event sequence for
estimating storage-related peak widening on these monitored connections.

This study retains the 59 real tributary pairs, 22 receivers and 11 connected
monitoring systems from the preceding river-footprint study. All 3,026
connection-months and 1,092 unique receiver-month observations are accounted for.
The three whole-network outline classes are unchanged. This is a follow-up
within ST357, not external validation, and no prediction model was retrained.

## What the dated records show

The original archive contains 9,173 accepted DOC result/activity records on
9,037 station-days and 7,085 station-months across 71 stations. Reconstructing
the original result-weighted monthly mean agrees with the frozen dataset to
within 0.0000031 mg/L. Four pairs of activity identifiers share the same
station/date/time/time-zone metadata; their DOC values also agree. These IDs
are therefore not counted as additional independent sampling days.

The median three-station sampling span is **4 calendar days**. Only 617 of
3,026 connection-months (20.4%) can be aligned to the same calendar day. Even
these same-day triplets span a median two hours. Across stations, the median
station-specific sampling interval is **28 days**; the pooled interval median
is 27 days. Date and activity selection uses metadata alone, never DOC values.

| Maximum sampling span | All selected pair-months | Pairs with ≥24 months | Receivers | Monitoring systems |
|---|---:|---:|---:|---:|
| Same day | 617 | 16 | 2 | 1 |
| ≤1 day | 945 | 20 | 5 | 3 |
| ≤3 days | 1,500 | 35 | 8 | 5 |
| ≤7 days | 2,110 | 45 | 11 | 6 |
| All original months | 3,026 | 59 | 22 | 11 |

The many same-day records mostly repeat a small set of rivers. Their number
does not create 617 independent river experiments. Also, in 1,802 selected
triplets (59.6%), at least one upstream sample occurs after the receiver sample
in UTC. Close dates are useful for variability comparisons, but are not
automatically an upstream-to-downstream arrival sequence.

## Does date alignment erase the mixing result?

No. For each date cut, monthly means and selected activities are compared on
exactly the same pairs and months. The concentration mixture still uses the
existing drainage-area proxy weights. Each connection receives its existing
calendar projection; connections are averaged within receiver, receivers get
equal weight, and entire monitoring systems are bootstrapped 5,000 times.

The covariance decomposition expresses the reduction relative to weighted
source variance as an amplitude-imbalance part and an asynchronous-variability
part. The latter remains substantial:

| Maximum span | Monthly asynchronous contribution | Dated-activity contribution | Paired difference, percentage points (95% CI) |
|---|---:|---:|---:|
| Same day | 31.77% | 30.89% | −0.88; one system, population CI unavailable |
| ≤1 day | 17.20% | 19.48% | +2.28 [−0.72, +7.18] |
| ≤3 days | 20.18% | 21.50% | +1.32 [−1.32, +6.30] |
| ≤7 days | 16.83% | 17.56% | +0.74 [−1.29, +4.91] |
| All | 16.25% | 16.10% | −0.15 [−1.45, +1.02] |

For the ≤3-day cohort, the dated-activity estimate is 21.50% with a
system-bootstrap interval of [17.83%, 27.78%]. The changes in this table do
not show a systematic collapse of asynchronous variability after replacing
monthly means. Different cuts retain different rivers, so the rows are not a
dose-response curve. A separate saved comparison measures the effect of month
selection against the complete monthly series of the same eligible pairs.

The overall total mixture-variance reduction changes only from 21.03% to
21.29%. This describes an area-weighted concentration mixture; it is not
observed DOC removal or a measured reduction of an individual downstream peak.
The measured outlet/mixture log-SD ratio is more sensitive to sample selection:
its full-cohort estimate changes from −0.264 to −0.217, and the dated estimate's
interval spans zero. Keep this downstream outcome separate from the exact
source-mixture identity.

## Flow and common-trunk storage

The daily archive supplies three positive measured flows on the selected dates
for 905 of 3,026 pair-months (29.9%). No daily gap is filled. Among 2,098 unique
source-date comparisons where the source and receiver dates differ and both
flows are measured and positive, the larger daily flow is at least 1.5 times
the smaller in 25.6%, and at least twice the smaller in 14.9%. Sampling offsets
can therefore cross important hydrologic changes. These daily means do not
identify instantaneous flow shares at the junction.

Only **two** pair-months in the entire cohort have three or more distinct DOC
sampling days at every station. The five connections with mapped lake/reservoir
segments in their common trunk cover four receivers and two systems; **none**
has such a dense common month. The archive cannot yet separate a widened
downstream DOC event from an event that happened between sampling visits.

The figures show real mapped connections and actual samples alongside daily
flow. Example selection uses geometry and data coverage. DOC points are not
joined into an invented daily concentration curve, and missing flow series
are explicitly labelled. Waterbody-path fraction is mapped geometry, not a
measured residence time.

## Next experiment: keep river structure as the explanatory variable

Proceed with controlled transport on the measured river footprints:

1. Hold the tributary DOC input, mixture shares and mean arrival time fixed.
   Change only independent-branch path differences to isolate relative arrival.
2. Hold branch geometry fixed. Compare a pure common-trunk time translation
   with a nonnegative, unit-gain storage-response kernel at the same mean
   arrival time. This distinguishes delayed arrival from a lower, wider peak.
3. Report peak height, centroid and response width separately. Reuse mapped
   branch/trunk storage and the original outline classes for interpretation.
   Scenario time stays relative unless hydraulic residence data support days.
4. Link these operators back to the date-aligned observations and actual flow
   coverage. Use them as process hypotheses for a later directed temporal graph
   operator; do not estimate removal rates from these sparse concentration pairs.

This directly advances the user's question: **which aspects of river shape
organize DOC arrival and mixing, and which internal structures could buffer
the signal?** Landscape-source differences remain background information.
