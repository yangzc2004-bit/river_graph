# Observed tributary coordination and receiving DOC

Date: 2026-10-08. Previous river-form and controlled-timescale results have been
seen. This is a new exploratory observational follow-up, not a confirmation of
a physical travel-time model.

## Question

Does the coordination of actual tributary fluctuations correspond to stronger
or smoother receiving DOC, and how does that coordination differ across the
original elongated, mainstem-dominated and broad whole-network forms?

Keep the full river form, cropped source paths and common-corridor context. No
new land-cover grouping, DOC model training or transport-parameter fitting is
introduced.

## Fixed sources and sampling

- Reuse the preceding 32 unique receiving networks, disjoint monitored-frontier
  source sets, source-role labels, accepted WQP activities, mapped path geometry
  and catchment-overlap components. Do not select a different source set by DOC.
- The main monthly panel uses exactly the previous 1,333 common receiver-months.
  Source concentrations are reconstructed from the saved selected-activity long
  table and reconciled against its monthly-series product.
- Same-calendar-day selected activities are the strict timing follow-up. The
  prior metadata-only sampling rule remains fixed. Sampling-span sensitivities
  of 1, 3 and 7 days are reported separately; monthly values are also evaluated
  on exactly these dates to separate date selection from averaging.
- Receiver eligibility remains at least 24 common months, three sampled years
  and six calendar months. Retain excluded subsets in the availability table.
- Whole-form comparisons retain both all receivers and the previous >=80%
  represented-area subset. Systems with overlapping catchments are resampled
  together. A shortest-route analysis uses its independently saved sources.

## Measures

Project all source and receiving concentrations on the same intercept,
month sine/cosine and linear-year design, as before. Also report raw centered
concentrations as a sensitivity.

1. Source coordination: variance-weighted pair correlation, recovered from the
   source covariance matrix and fixed source area shares.
2. Receiving fluctuation: receiving SD divided by sum(w_i * source SD_i), the
   perfectly coordinated source reference. This denominator does not contain
   pair covariance. Receiving/actual-mixture SD remains a secondary bridge to
   the preceding study.
3. Coincident excursions: each source and the receiver exceed their own 75th
   anomaly percentile. Compare receiving-excursion frequency when >=2 sources
   are high with when exactly one is high. Report both denominators and retain
   a descriptive effect only when both groups contain >=5 dates. A 90th
   percentile sensitivity is retained, with its own small-sample counts.
4. Do not present the mathematical relation between source correlation and
   source-mixture variance as independent evidence of downstream propagation.

## Within-network checks

- Non-overlapping calendar five-year blocks, beginning at years divisible by
  five; retain blocks satisfying the same 24/3/6 coverage rule. Reuse each
  network's full-panel projected anomalies, then center within each block.
  Report receiver-centered associations only for networks with >=2 eligible
  blocks. A metadata-only preflight finds 28 blocks in 21 receivers, with four
  receivers having repeated blocks; do not enlarge this population after
  inspecting the outcome.
- Retain the previously selected dense Loch Vale case. Use all same-day records,
  and within-month anomalies only in months containing >=3 joint dates. Compare
  annual periods with >=12 dates and >=3 months. Selected activities and daily
  means of all accepted activities are separate sensitivity versions.
- Annual associations and peak-risk intervals in that case resample calendar
  years; they describe this one lake-influenced river, not independent form-level
  replication. Preserve per-year sample counts and omitted-year sensitivity.

## Analysis and deliverables

Receiver-equal summaries and form differences use 5,000 whole-catchment-system
bootstrap draws, seed 42. Individual intervals are exploratory, with no
multiple-comparison confirmation claim. Within-network period associations
give each receiver equal total weight, retaining all its periods together.

Save source/receiver anomaly series, all network and period metrics, excursion
counts, sampling availability, whole-form contrasts, associations, dense-case
results and bilingual standalone scientific figures. Independently check key
covariance, peak-count and bootstrap calculations and inspect the actual images.

## Interpretation

Coordinated tributary changes can support the proposed arrival-overlap mechanism
as an observational association. These observations cannot by themselves recover
event transit times, DOC mass loss, causal lake effects or a region-independent
whole-form effect. Within-network changes hold geometry fixed and therefore test
the input-timing part of the mechanism; they do not estimate a changing shape.
