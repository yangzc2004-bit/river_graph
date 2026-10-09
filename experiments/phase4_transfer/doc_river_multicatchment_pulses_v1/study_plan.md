# Independent catchment DOC pulse comparison

## Research question

Do independently observed river outlets show the same DOC peak delay and pulse
spreading as Kervidy? Assemble the accompanying real river maps so that subsequent
comparisons can test tributary path dispersion and shared downstream routing.
Catchment source composition is context; it is not the definition of river form.

## Selection before new response measurements

Retain Kervidy and acquire all available records in two independent published
DOC/discharge archives: Rappbode (Germany, 2018–2023) and Bouleau (Canada,
2018–2019). The Vermont NEWRnet three-catchment archive is discoverable but its
files explicitly require owner permission, so it is recorded as unavailable.
Wood Brook's public Figshare metadata endpoint returned HTTP 403. Neither source
will be accessed through an alternative path intended to bypass that restriction.

Rappbode contains laboratory-calibrated optical DOC, interpolated gaps below two
hours and a 2.5-hour moving average. It is a processed-signal comparison, not an
unfiltered replication. Bouleau has separate observed/calibrated and predicted DOC
columns: use the former, never substitute the latter for a missing observation.
Inspect the source's DOC calibration, timestamps, flow units and gap handling.

## Common measurement protocol

Use only the recorded flow to select events. Keep the preceding Kervidy rules:
6-hour peak separation, 7-day prominence window, relative flow prominence 20%
(15% and 25% sensitivities), bounded 3–96-hour single pulses, baseline 6 to 1 hours
before event start, and response follow-up to flow return plus 24 hours, shortened
by the next qualifying flow envelope. Retain all candidate and unresolved events.

The absolute prominence is 0.35 mm/day of specific discharge, converted using
each published catchment area. This replaces the Kervidy-specific 0.020 m³/s
criterion for this new comparison only. Keep the previous Kervidy results intact.

Measure native-cadence signals first. Also compare at one-hour cadence using
existing observations sampled on each archive's hourly clock, without averaging
or filling missing values. Do not infer sub-hour precision from hourly data.
Coverage is at least 90%; no gap greater than one expected sampling interval for
hourly data, or one hour for quarter-hour data. Scale expected baseline counts to
cadence. Native records labelled with an unspecified time zone are analysed on
their original clock and never assigned an invented UTC offset.

Outcomes: DOC minus flow peak lag; DOC/flow half-excess width ratio; positive
response denominator and unresolved/censored counts; calendar-month-block
bootstrap medians (5,000 draws, seed 42); period and prominence sensitivity.
Conditional recovery times remain separate from censored observations.

## Geometry and interpretation

Use provider river vectors when accessible, otherwise retain the original
published river map with a clearly labelled qualitative description. Do not
invent headwaters, raster-trace a schematic into measured paths, or substitute
catchment land-cover class for river-network morphology. Only a validated rooted
network supports measured path dispersion and shared-terminal-length endpoints.
Events are temporal observations within a catchment; they do not increase the
number of independent river forms. No regression treating hundreds of events as
hundreds of independent catchments will be used.

## Deliverables

Public-source retrieval manifest; clock/unit/processing audit; all event
inventories; native/hourly/sensitivity summaries; pulse comparison figures;
real-map evidence inventory; English research decision and reproducible analysis.
No training or alteration of earlier experiments.
