# Separating river transport from DOC processing with paired tracers

## Question and connection to river form

The structure experiments identify two operations: different incoming paths
separate arrivals, and a shared reach can spread the combined response. This
version asks whether laboratory DOC follows the conservative water response, or
changes beyond that response, in a publicly archived field addition. It examines
the transport/processing link needed to interpret whole-network form; it does
not assign a new morphology class to a short reach.

## Sources selected before computing response contrasts

Use all three Blaine Creek additions from the public archive accompanying Hall,
Hotchkiss, Baker and Plont (2026), DOI 10.1007/s10021-025-01030-2. Retain both
sampling locations, 41 and 61 m from the release, for every addition:

- 8 August 2019: labelled glucose;
- 9 August 2019: labelled leaf leachate;
- 15 August 2019: labelled glucose.

The HydroShare resource is
https://www.hydroshare.org/resource/988f0d0aa46249b2b654145cf5fbf895/ (CC BY 4.0).
The linked author code is https://github.com/robohall/DOC_uptake. Retrieve the
original analysed laboratory CSVs and site metadata. Preserve author-processed
intermediates and the processing code separately for an inspectable comparison;
never relabel their interpolated/edited values as original measurements.

The White Clay Creek eight-station experiment is an additional data lead, DOI
10.6073/pasta/b839940664e7c198a5472530978a1e3e. Its public portal currently
redirects to login and its metadata API returns an explicit authorization error.
Record that access outcome and proceed with accessible evidence. No author
contact or restricted-data access is part of this task.

## Analysis order

1. Establish actual measurement columns, original record grain, sampling clocks,
   release times, missing values and sampling gaps for all six pulse/site series.
2. Distinguish total DOC from excess isotope-labelled DOC. Compute atom fractions
   from the archived isotope equation and retain the raw isotope measurements.
   Make background choices visible and check their sensitivity.
3. Compare the measured conservative and labelled-DOC responses at the same
   sampling times. Show the actual points; do not fill missing laboratory values
   or turn oxygen incubations into DOC curves.
4. Separate conservative travel/spreading diagnostics from tracer-normalized DOC
   transmission. Report sparse sampling, background drift and incomplete tails
   beside any duration/integral result. Use explicit integration windows and
   bounded adjacent-observation integration, not extrapolated tails.
5. Compare upstream/downstream pairs within each addition. Do not count two sites,
   many samples or two glucose dates as independent river forms. Do not infer
   whole-network class effects from one 20 m segment.

The exact observable definitions and justified background/window choices will
be written in method notes after source inspection and before contrast
computation. Any unavailable parameter or unreliable contrast remains visible.
The primary outcome is an empirical decomposition of transport and chemical
response, not another model-selection exercise or another prescribed kernel.

## Deliverables

An independent versioned analysis, source inventory, original-point products,
paired tracer diagnostics, English research interpretation, and matched English
and Chinese scientific figures. Keep the original three river classes and all
previous structural results intact. No prediction model training is scheduled.
