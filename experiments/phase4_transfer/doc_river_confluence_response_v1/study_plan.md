# Real confluences: incoming-wave overlap and local mixing geometry

## Question

Do two real incoming branches supply synchronized water waves, and does a
confluence's measured downstream geometry correspond to a consistent DOC
response? This follows the single-pair hourly audit with actual branch–branch–
receiver observations. It does not modify the original three whole-network
morphology classes or any prediction-model experiment.

## Two complementary observation sets

1. **Krycklan water timing.** Retrieve the public SITES 30-minute stage/discharge
   objects for the eleven sites in the four previously mapped configurations.
   Reuse all twenty spring windows from `doc_river_event_observations_v1`,
   selected from receiver daily flow before this analysis. Each window includes
   the fourteen calendar days on either side of its peak date: 29 days, 1,392
   half-hour slots, using the source's fixed UTC+1 clock. These are seasonal
   windows, not twenty independent storms. Keep laboratory DOC as actual sample
   points; do not interpolate it to half-hourly DOC or estimate it from flow.
2. **Five Tom's Creek confluences.** Retrieve Plont's public HydroShare chemistry,
   discharge and fall channel-transect files, author analysis and supplement.
   All five confluences and both 2021 campaigns are retained. Three downstream
   lateral transect values are spatial samples; two seasons at one confluence
   are repeated observations. This is a local confluence-geometry study, not
   a classification of five complete catchment forms.

The source schemas and measurement scope have been inspected before this plan.
No result-dependent site or window selection is permitted. The existing hourly
optical case remains separate because of its flow/rainfall-derived DOC rows.

## Calculations

### Incoming water-wave timing

- Match the exact half-hour clock and report valid coverage at each site and
  jointly. Never fill missing water or carbon observations.
- Compare the sampled maximum clocks, retaining the full range of tied maxima
  and the count of disconnected tied-maximum runs. These are seasonal peak
  clocks, not hydraulic travel-time estimates.
- Subtract each branch's window minimum and normalize its remaining flow to
  unit sum. Wave overlap is `sum(min(profile_a, profile_b))`. Compare on the
  same jointly observed clock; report whether all slots were observed.
- Calculate `max(excess_a + excess_b) / (max(excess_a) + max(excess_b))`.
  This describes peak coincidence of the incoming water signals at their
  gauges. It is not an observed outlet peak reduction or DOC retention.
  Also show the amplitude-balanced version, obtained by scaling each branch's
  excess maximum to one, and the original peak-amplitude share. A dominant branch
  can make the unscaled ratio near one even when the two waves are staggered.
- Show sensitivity to a within-window 10th-percentile baseline, with negative
  excess clipped to zero. No lag search or shifted best-fit alignment is used.
- Report the branch-flow sum / receiver flow, correlation and waveform change.
  The two measured branches do not account for the complete receiver basin.
- Primary descriptive comparisons require at least 95% joint clock coverage;
  all other windows and their missingness remain in the availability table.
  Display results for the fully observed subset as well. Group by configuration
  and retain independent path difference and shared-trunk length from the map.

### Local confluence geometry and measured DOC

- Recompute normalized flow weights from the two reported upstream discharges.
  Preserve and compare the source's supplied weights rather than assuming all
  supplied rows use the same denominator.
- The reference mixture is `(Q_main*C_main + Q_trib*C_trib)/(Q_main+Q_trib)`.
  Average the archived upstream concentration values within confluence/campaign,
  and expose how many distinct values were available. Downstream is the equal
  mean of the three lateral transect concentrations, **not** a flux-weighted
  cross-section concentration. Retain all three points and their range.
- Report downstream-minus-reference concentration in mg C/L and percent;
  whether the reference lies inside the measured lateral range; branch-flow
  share; and a parallel conductivity mixing diagnostic.
- Aggregate width once per surveyed transect. Average depth across its surveyed
  points, then equally across transects; do not give repeated width copies extra
  statistical weight. Compare downstream/mainstem width, depth and width CV.
  Retain the source's separate survey IDs, including channel/pool subunits.
  Rock/tree-obstructed missing depths are counted and left missing; the reported
  depth average describes available measured points, not zero-filled obstructions.
- Fall geometry is paired only with fall DOC. Retain the author's published
  100-m normalized residence-time summary and its definition. Correlations with
  DOC are descriptive at five confluences; list leave-one-confluence ranges and
  season-to-season sign changes instead of treating 30 rows as independent sites.

## Interpretation and products

Water-wave overlap, measured concentration mixing and local channel shape are
different links in the proposed mechanism. Their correspondence motivates a
complete synchronized three-site DOC campaign; it cannot substitute flow for
carbon timing, establish removal from concentration alone, or rank whole-network
forms without their full geometry. Keep the focus on paths, junction arrangement
and post-junction channel structure.

Save a source inventory, source-linked tables, English research decision and
bilingual scientific figures. Reproduce source calculations with meaningful
synthetic and local-data replay tests. Keep raw downloads under `data/raw/` and
preserve all preceding experiment directories.
