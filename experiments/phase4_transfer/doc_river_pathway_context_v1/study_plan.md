# River form, monitored pathways and receiving DOC variability

Date: 2026-10-08

## Question

Do tributary entry positions, shared downstream paths and mapped waterbodies
help explain why a receiving DOC signal can be smoother or more variable than
its observed tributaries? Keep the original elongated, mainstem-dominated and
broad whole-network classes. This is an exploratory follow-up after inspecting
the monitored-arrivals study; it is not a new confirmatory model experiment.

## Design

- Reuse the 32 fixed disjoint monitored frontiers and their common months.
  Preserve source-role 142/143/144 visibility, physical receiver aliases and
  the seven catchment-overlap systems. No new prediction training.
- Trace actual gauge-to-receiver paths with gauge NHD measures and original
  geometric mainstem routing. Save the shortest-route sensitivity separately.
- Measure source-path dispersion, the downstream corridor common to every
  gauge, mainstem entry positions and mapped lake/reservoir length. A missing
  reach tag does not establish absence of a lake. Retain the documented Loch
  Vale context as a separate evidence field, rather than changing the NHD data.
- Examine all receivers and the metadata-defined >=80% area-coverage subset.
  Also audit 50/70/90% coverage, without choosing the cut by DOC response.
- Audit comparison opportunity among all 295 physical networks: same HUC4,
  original elongated/broad classes and area ratio <=2. Record every candidate
  and whether both have sufficient common DOC and >=80% monitoring coverage.
  This is a sampling-overlap audit, not environmental or causal matching.
- Decompose receiving anomaly variance exactly into upstream-mixture variance,
  variance of the receiving-minus-mixture mismatch and their covariance. The
  mismatch is a concentration-signal discrepancy, not an identified local DOC
  source, mass balance or retention process.
- On common months with positive measured monthly discharge at every gauge
  and receiver, compare area-share and varying flow-share mixtures on exactly
  the same dates. Require the previous 24-month/3-year/6-calendar-month rule.
  Report discharge closure; analyze the 0.8--1.2 closure subset only if it
  meets the same sampling rule. Monthly means do not resolve event routing.

## Analysis and products

Each receiver has one vote. Use 5,000 paired complete-system bootstrap draws.
Report geometric associations and broad-minus-elongated contrasts with their
intervals, retaining zero-crossing results. Distinguish algebraic mixing from
observed receiving attenuation. Output path-reach tables, a receiver panel,
variance identities, coverage/comparison ledgers, flow sensitivity, English
research decision and English/Chinese standalone figures. Recompute and check
the consequential quantities independently before interpreting them.

## Interpretation

Keep the chain **whole form -> pathway arrangement -> tributary overlap ->
receiving DOC variability**. Environmental context supports a fair structural
comparison; it does not replace that question. If high-coverage forms cannot
be compared within the same region, identify that specific sampling gap and
retain the controlled geometry evidence separately from field results.
