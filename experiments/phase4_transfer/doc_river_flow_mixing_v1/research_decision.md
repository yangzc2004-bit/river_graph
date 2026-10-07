# Branch mixing and the remaining downstream DOC response

## Research decision

Flow-weighted mixing reproduces the **direction of lower relative DOC
variation** at confluence C7 in all seven years with adequate matched flow.
The receiving stream has additional reduction in absolute SD in only three
of those seven years. Branch mixing is therefore a concrete explanation to
carry forward; a consistent extra downstream buffering effect has not been
identified in these windows.

The next structural question is how branch arrangement controls the
**synchrony of the incoming concentration signals**. Differences in travel
paths may change their overlap at the junction. Our campaign observations
measure concentration synchrony; hourly event comparisons are needed to test
the associated propagation timing.

## Observations and comparison

The preceding laboratory study identified 11 eligible spring windows with 74
matched three-site campaigns. We retain those selections and match each
laboratory sample to the published daily discharge on its fixed UTC+1 date.
No flow interpolation or replacement of missing values is used.

Eight windows have at least five campaigns with all three discharges:
**seven C7 windows with 47 campaigns, and one C9 window with six campaigns**.
All 53 comparison campaigns use the same calendar day at all three sites.
Every concentration and variability comparison uses these same campaigns.

Three original windows remain in the availability table:

- C7 in 2011 has no valid C2 discharge on its seven laboratory dates.
- C16 in 2012 has four complete-flow campaigns, below the five-campaign rule.
- C16 in 2018 has no matching C14 discharge; that gauge record ends in 2017.

There are 57 complete-flow campaigns across the original 11 windows, including
the four retained C16 campaigns. Only 53 belong to windows summarized for
variation. Calculations on all 74 original campaigns are preserved.

Sources are the [SITES daily-flow collection](https://meta.fieldsites.se/collections/2M74g6oPabfeXXoOkaiGbFs2)
and the laboratory campaigns and directed paths preserved in
`doc_river_event_observations_v1`. Discharge is derived from measured stage and
weir, flume or channel rating relationships, rather than an independently
observed instantaneous flow at each chemistry sample.

## What simple branch mixing explains

The calculated upstream mixture is

\[
C_{mix} = \frac{Q_A C_A + Q_B C_B}{Q_A + Q_B}.
\]

It is a mixture of the **two measured inputs**. At C7, its CV is lower than the
arithmetic mean of the two branch CVs in seven of seven windows. The actual
receiving-stream CV is also lower than that branch average in all seven.
These C7 comparisons retain every original DOC campaign in their respective
years, so the direction is not produced by dropping chemistry samples.

Across the seven years, the marginal median CVs are 0.216 for the branch
average, 0.187 for the flow-weighted mixture and 0.150 for the receiver. These
are descriptive medians, not a paired effect estimate or the fraction of
downstream variance explained.

| Receiver and year | Campaigns | Branch mean CV | Flow-weighted mix CV | Receiver CV | Mix SD | Receiver SD | Median measured flow share |
|---|---:|---:|---:|---:|---:|---:|---:|
| C7 2015 | 8 | 0.310 | 0.282 | 0.218 | 5.638 | 4.467 | 0.767 |
| C7 2016 | 7 | 0.205 | 0.191 | 0.129 | 3.685 | 2.708 | 0.733 |
| C7 2018 | 7 | 0.216 | 0.175 | 0.150 | 3.020 | 2.980 | 0.653 |
| C7 2020 | 8 | 0.220 | 0.212 | 0.201 | 4.369 | 4.498 | 0.743 |
| C7 2021 | 6 | 0.161 | 0.055 | 0.071 | 1.019 | 1.454 | 0.736 |
| C7 2023 | 5 | 0.182 | 0.121 | 0.130 | 2.088 | 2.518 | 0.638 |
| C7 2024 | 6 | 0.227 | 0.187 | 0.193 | 3.198 | 3.679 | 0.651 |
| C9 2015 | 6 | 0.081 | 0.084 | 0.071 | 1.364 | 1.074 | 0.570 |

SD is in mg C/L. Flow share is `(Q_A+Q_B)/Q_receiver`; window medians are
shown. Across C7 windows, the median of those flow-share medians is 0.733.

The reduction in CV cannot alone establish a reduction in absolute
variability. At C7 the receiver CV is below the calculated mixture in four
of seven years, while receiver SD is below it in only three. In 2020, for
example, receiver CV is smaller but its SD is slightly larger because the
mean concentration also differs. The receiver is not consistently a smoother
version of the partial upstream mixture.

## Synchrony is a measurable part of the mixing mechanism

Using each window's mean flow fraction as a fixed weight, its mixture variance
decomposes exactly:

\[
\operatorname{Var}(C_{mix}) =
w^2 \operatorname{Var}(C_A)
+(1-w)^2 \operatorname{Var}(C_B)
+2w(1-w)\operatorname{Cov}(C_A,C_B).
\]

The covariance term is negative at C7 in **2016, 2021 and 2023**. In those
windows the coeval branch signals offset each other and reduce the mixture
variance. It is positive in the other four C7 windows and increases it.
Thus the same mapped junction can combine branch signals differently across
years. Flow weights also vary, so the fixed-weight decomposition is reported
alongside the full daily-weighted calculation.

This identifies a mechanism worth connecting to structure: the timing and
coincidence of incoming signals matter in addition to the number of branches.
Covariance of sparse concentrations does not by itself measure a river
transport delay or show that path length caused their temporal association.

## Partial coverage and the C9 sample change

The monitored C2 and C4 drainage areas sum to 63.6% of the stated C7 area.
Their flow contribution has window medians from 0.638 to 0.767. At C9, the
two monitored areas sum to 54.3% of its area and the median measured flow
share is 0.570. At C16 the two measured areas cover 26.8%, although no C16
window qualifies for the variation comparison with flow.

The remaining flow can bring additional DOC along the shared path. Differences
between receiver concentration and the calculated mixture therefore combine
unmonitored inputs, daily-flow approximation and downstream processes. The
stored `C * Q` values are approximate point fluxes in g C/s, not integrated
event loads or a complete chemical retention balance. Catalogue coordinates
for the chemistry sites and their discharge gauges are identical; this
checks their advertised locations, rather than providing an independent
survey of sampling position.

C9 loses one laboratory campaign because receiver flow is missing on that
date. On all seven original campaigns, receiver CV was 0.099 versus branch
average 0.094. On the six complete-flow campaigns those values become 0.071
and 0.081. This change belongs to the sample subset; it is not a newly
replicated reversal of the C9 result. C7 supplies the repeated field lead.

## What this means for the river form study

C7 has highly unequal monitored path lengths: 0.015 and 1.090 km, with only
0.014 km shared. C9 has nearly equal lengths of 2.085 and 2.035 km, with
1.708 km shared. These contrasts keep independent branch length and shared
downstream position as the structural variables. With seven windows at C7
and one at C9, a regression would mainly compare two locations; no class
ranking or structural slope is inferred from eight rows.

The research sequence is now:

1. **Establish mixing from actual observations.** This stage shows that the
   weighted combination of measured branch signals can produce the lower
   relative variation observed at one repeatedly sampled junction.
2. **Examine event timing.** Use the preserved hourly case to compare flow
   arrival and optical DOC timing, with reconstructed/high-turbidity peaks
   distinguished in interpretation. Select and assess events using discharge
   and coverage rather than concentration-response direction.
3. **Test the structural link across independent configurations.** Relate
   incoming signal synchrony and downstream response to independent route
   difference, confluence position and common length. Retain the original
   three whole-network classes and seek further complete-flow confluences.

This keeps the main question on how river structure organizes the combination
and timing of DOC signals. Land-source differences are background variables,
not a replacement for the structural question.

## Reproduction

```bash
uv run python scripts/analyze_doc_river_flow_mixing_v1.py
uv run python scripts/plot_doc_river_flow_mixing_v1.py
uv run python scripts/plot_doc_river_flow_mixing_v1.py --chinese
uv run python scripts/verify_doc_river_flow_mixing_v1.py
```

Inputs, calculations, source records, code snapshots and bilingual figures
are retained in this new directory. Prior observation tables and the original
morphology assignments are unchanged. No prediction model was trained.

SITES attribution is retained: "This study has been made possible by data
provided by the Swedish Infrastructure for Ecosystem Science (SITES)."
