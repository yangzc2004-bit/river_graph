# Single flow pulses and DOC response in the mapped Kervidy catchment

Date: 2026-10-09

## Scientific question

How closely does DOC follow a single outlet flow pulse: does its peak precede
or follow the flow peak, and is its response more concentrated or longer-lived?
Use this observed response to develop a comparable event measurement for the
river-form study. Kervidy geometry is fixed, so event variation is not an
estimate of an independent morphology effect.

The four annual flow-selected windows are already known. They motivated
single-pulse segmentation because some windows contain several rises. This is
an exploratory method fixed before applying it to the full corrected-DOC
period, not an unseen confirmation test. Existing experiment results remain
unchanged.

## Pulse definition (flow only)

Use the existing quarter-hour UTC discharge records without interpolation or
smoothing. Split at every missing quarter hour. Find local flow maxima using
SciPy, with at least six hours between maxima and a seven-day prominence
window. A candidate must have prominence at least 0.020 m³/s and at least 20%
of its recorded peak. The absolute increment is approximately 0.35 mm/day
over the 4.89 km² basin; it is an analysis scale, not an instrument accuracy
claim. Repeat at relative prominence 15% and 25%, changing no other setting.

Bound a pulse by the nearest observed flow values on each side at or below
peak minus 90% of prominence. Retain all candidates, but distinguish events
whose bounded span is 3–96 hours and contains no second qualifying flow peak.
Record boundaries at the original quarter-hour observations. Censored flow
maxima stay in the inventory but do not identify peak lag or width.

Use a fixed antecedent interval six to one hours before the flow start to
estimate flow and DOC baselines (median; at least 16 of 21 expected records).
Follow DOC from flow start until 24 hours after flow return, or the start of
the next flow candidate, whichever comes first. Record the resulting follow-up
limit. DOC values do not select the flow candidates or their isolation.

## Response measurements

Require 90% occupied DOC bins and no gap over one hour in the response and
antecedent intervals to resolve timing; keep failures with their reasons.
Report no-rise DOC responses separately. A positive DOC pulse has excess
above its antecedent median of at least max(0.5 mg/L, 10% of that median).

For resolved positive pulses, report DOC-peak minus flow-peak time, including
uncertainty from equal-valued peak plateaus. Measure each half-excess width
from the observed below-half-excess points bracketing its peak. Do not infer
crossings across a gap greater than one hour. Missing crossings are left/right
censoring, not zero duration. DOC recovery means the first observed return to
within 10% of its peak excess above antecedent baseline. No return before
follow-up ends is a censored recovery, not an assigned 24-hour duration.

Report the DOC/flow half-width ratio only for pairs with both crossings.
Bootstrap within-catchment summaries by calendar-month blocks (5,000 draws,
seed 42), keeping all qualifying pulses in a sampled month together. These
intervals describe event variation within one catchment, not uncertainty in
a cross-form effect. Check threshold sensitivity and year distributions.

## Geometry and deliverables

Inspect the separately published BD Topage network and its documented direction
fields, if delivered, as an independent check of the existing map. Only release
rooted path metrics if its outlet connectivity and direction can be verified.
Do not construct transport paths from polygon shape or assume map coordinate
order is flow direction.

Save the complete pulse inventory, response measurements, sensitivity table,
readable figures and English research decision. Use chronological examples;
do not select examples by the largest DOC response. State which measurements
are now suitable for comparing independent real river networks next.
