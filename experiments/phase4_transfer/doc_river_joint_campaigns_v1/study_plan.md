# Joint-calendar observations at four actual river configurations

Date: 2026-10-08.

## Question

Does downstream DOC variability differ from the measured branch mixture on
the same sampling dates, and does the comparison persist when four actual
confluence arrangements are observed on a joint calendar?

This extends the saved spring-window study to the full laboratory archive.
The availability inspection found 97, 97, 308 and 61 complete-flow campaigns
at C12, C16, C7 and C9 before the same-day/duplicate-day rules below. Previous
spring results are known. This is an exploratory follow-up, not an independent
replication or a newly selected prediction-model evaluation.

## Fixed population and information

Use all four nearest-monitored separate-branch configurations from the saved
Krycklan mapping: C1+C10→C12, C13+C14→C16, C2+C4→C7, C6+C7→C9. Retain their
measured branch and common-path lengths. They are local arrangements in one
connected research catchment; do not assign new whole-network morphology labels.

Use the saved laboratory DOC, daily discharge and source metadata. Match each
laboratory sample at most once within a configuration, with at most 12 hours
between all three timestamps. Require complete valid daily flow and a common
fixed-UTC+1 calendar date. Exclude every duplicate receiving-date campaign from
the comparison, preserving them in the availability ledger. Do not interpolate
DOC, fill flow gaps, infer unobserved peaks, or change original records.

Compare (a) all eligible dates within each configuration and (b) the exact
intersection of eligible dates across all four configurations. Dates enter the
intersection by availability, not by DOC response. The nested C7/C9 configurations
are explicitly dependent.

## Calculations

1. Calculate the partial upstream flow-weighted concentration mixture, retaining
   the actual upstream/receiver flow ratio and contributing-area coverage.
2. Report native mean, SD and CV; primary comparison is the outlet/mixture SD
   ratio after projecting every concentration series onto the same intercept,
   annual sine/cosine and linear calendar-time design. No extrapolation is used.
3. Calculate branch correlation and the exact fixed-mean-flow-weight variance
   decomposition on the same adjusted dates. This distinguishes synchronized
   inputs from cancellation; the mathematical identity is not a new field law.
4. Define low/middle/high flow using receiver daily positive-flow terciles within
   the eligible observation span, including days without DOC. Reuse the full
   population's seasonal projection within states. Report observed state contrasts
   without calling samples continuously resolved flood peaks or transit lags.
5. Repeat the absolute-SD comparison on dates with upstream/receiver flow ratio
   0.8–1.2, recording the available count. This is a water-coverage sensitivity,
   not a claim of a complete carbon budget.
6. Use 5,000 whole-calendar-year bootstrap draws, refitting the seasonal projection
   in each draw. Joint-calendar comparisons share the same resampled years across
   configurations. Intervals describe temporal repeatability at these sites, not
   independent river systems or population-level morphology effects.

## Deliverables and decision

Save complete campaign/availability tables, joint-date comparisons, water-state
and yearly summaries, paired configuration contrasts, figures and a short English
research decision. Retain unfavorable and unavailable comparisons. Connect the
results to actual path difference and common-corridor arrangement, without fitting
a structural slope to only four nested configurations.

This version should answer how much of the observed buffering belongs to branch
combination, how variable the remaining outlet response is, and whether differing
calendars created the apparent configuration ranking. If actual event lag and
whole-form contrasts remain unresolvable, close that inference explicitly and
specify the measurements needed; do not generate another pulse simulation.

No new predictive model is trained and no earlier experiment is modified.

Data attribution: This study has been made possible by data provided by the
Swedish Infrastructure for Ecosystem Science (SITES).
