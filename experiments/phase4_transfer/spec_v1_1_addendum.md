# Phase 4 transfer specification addendum (v1.1)

Status: **protocol clarification and correction after exploratory query
results were seen, 2026-09-24**. This document preserves `spec_v1.md`, the
transfer charter, and the transfer endpoints as historical frozen files. It
does not claim that the corrections below were preregistered before the
exploratory Stage 2/2A evaluations. No cross-analyte training has been run.

## External graph station rule

The external graph node set is the union of valid `Stream` stations returned
by the three analyte inventories, after HUC filtering and coordinate checks.
Each analyte keeps its own observed-cell mask on that shared node order. The
all-three station intersection is a mapping diagnostic, not an additional
eligibility threshold. This follows the ST357 shared-graph convention, where
the three analytes have different active-station sets.

Two failed HUC8 screens leave external validation pending. They do not prove
that no external basin can qualify. Future candidates are ordered by canonical
HUC8, then candidate name, within a recorded candidate manifest. Only
availability, QC completeness, station metadata, graph connectivity, and
feature/split feasibility may enter selection. A candidate requires the full
data gate before being selected; no threshold is relaxed using model results.

## Whole-basin exclusion and task membership

The five existing HUC6 tasks retain their frozen `hide_rows` and
`task_component_rows` in `experiments/kshot_protocol_v2/regions.json`.
`510020` remains an alias for canonical HUC6 `051002`.

- Every row of the target HUC6 is hidden from source fit, source selection,
  climatology, and source-derived thresholds.
- Target support and query cells are generated only on the frozen largest
  component (`task_component_rows`), with fixed query and nested K sets.
- The source set is the complement of the whole `hide_rows` set. It is never
  the complement of the component alone.

## Distinguish the two label regimes

Stage 2A is a descriptive **same-analyte support diagnostic**. Its reference
climatology and tail threshold use target-analyte observations outside the
whole target HUC6. These are privileged reference labels relative to a
leave-one-analyte model. The diagnostic therefore measures local support
value given an available same-analyte reference; it does not establish
cross-analyte transfer or ecological information value.

The planned leave-one-analyte route excludes the target analyte from every
source training and source selection episode, and permits only K support
labels during target adaptation. The target-unseen K=0 output scale is not yet
defined. A pooled mean of differently measured source analytes, target-analyte
statistics from other basins, or a newly invented target prior does not solve
this without changing the protocol. The output mapping, transforms, and
source-only model-selection episodes must be specified and tested in a new
version before transfer training can start.

## Baseline selection remains a transfer-launch prerequisite

The final “best simple baseline” comparator must be chosen without target
query error. Its selection episodes and comparable scoring scale must be
frozen before a valid transfer run. Under the target-unseen regime, validation
cannot secretly reuse the held-out analyte. Privileged same-analyte controls
(including single-analyte H2X) must remain a separate, explicitly labelled
comparison group. The earlier phrase “lowest validation MAE for each target
analyte” did not resolve this label-role conflict and is not executable
authorization.

## Descriptive Stage 2A v2 estimator

The v2 reanalysis uses this order for every method and K:

1. Mean absolute error across query cells within an analyte, basin, calendar
   month, and task-seed realization.
2. Equal mean over task seeds within that analyte, basin, and calendar month.
3. Equal mean over eligible calendar months within each analyte and basin.
4. Equal mean over the five basins within each analyte.

Paired K differences use the same tasks and this same estimator. Different
task seeds are repeat allocations of observations, not additional ecological
samples. Report unique calendar-month counts separately from task realizations
and unique station-month cells. Do not multiply the month count by seeds.

Bootstrap the **calendar month**, retaining all seeds, methods, K values, and
basins for a sampled month together. For a pooled analyte estimate, sample the
union of eligible months and apply each sampled month's multiplicity to every
basin in which it occurs; recompute within-basin month means and then equal
basin means. Basins are fixed strata and are not resampled. Empty-stratum
draws must be handled explicitly by the implementation and recorded. This
preserves shared calendar-month variation across basins. Confidence intervals
remain descriptive after the earlier query exposure.

Report analytes separately in their native units. Equal raw-MAE weighting
does not make mg/L, pH, and uS/cm commensurate, so no pooled raw-MAE number
across analytes is authorized by this addendum.

## Gate boundary

Stage 2A does not evaluate the complete Stage 2 gate. Required controls,
missingness regimes, information decomposition, and valid transfer label
roles remain prerequisites. No Stage 2A direction or confidence interval
unlocks Stage 3. The detailed correction history is in
`stage2a_baseline_correction.md`.
