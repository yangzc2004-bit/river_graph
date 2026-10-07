# Whole-network form and actual junction geometry

## What this step answers

The original elongated, sparse/mainstem and broad-branching forms describe an
entire river network. We now measure the local channel geometry at their
previously selected major confluences. This asks whether an apparent
whole-network contrast can be explained simply by different incoming angles or
by bending immediately after the junction.

**The three forms have strongly overlapping local geometry.** Their incoming
angle medians are approximately 83–85 degrees at the main 250 m scale. The
original elongated/broad matches do not establish a difference in incoming
angle, area-weighted flow deflection or immediate downstream sinuosity.
Whole-network form therefore should not be represented by one characteristic
junction angle. The next structural mechanism to resolve is the arrangement
of junctions, branch arrival separation and shared downstream processing length.

## Actual measurement population

The frozen morphology/DOC panel has 297 station-network instances. We retain
the 295 area-balanced junctions selected in the preceding storage-placement
study; the two original networks without an eligible junction remain excluded.
Selection used upstream geometry and contributing area, not DOC response.

All 295 are measurable over exact 100 and 250 m centreline lengths. At 500 m,
293 remain measurable; two common downstream corridors terminate before that
length. They stay unavailable rather than being extended. The observed mapped
junction endpoint gaps are zero at the primary scale.

The 295 receiving networks contain **251 different physical junctions**. Twenty-
nine junctions recur in multiple nested receiving networks; five occur under
more than one whole-network form. Repeated junctions are retained as part of the
network description and are not called independent streams. The original three
representatives are shown at both whole-network and junction scales, with actual
mapped channel geometry and physical scale bars.

The current expanded mapping catalogue has additional classified entries; it is
not used to replace the original 297-network DOC panel. Label agreement is checked
against that original panel, including its three station aliases absent from
the expanded representative catalogue.

## Local geometry at 250 m

Directions come from exact projected centreline arclengths, with upstream branches
pointing out from the junction and the common channel pointing downstream. The
incoming angle uses the two upstream chords. Deflection uses incoming flow
direction versus the downstream chord. Sinuosity is mapped arclength divided by
the end-to-end distance. These are mapped centreline measurements, not surveyed
channel widths or discharge ratios.

| Original whole-network form | Network instances | Physical junctions within form | Incoming angle median | Area-weighted deflection median | Downstream sinuosity median |
|---|---:|---:|---:|---:|---:|
|Elongated / tributary-rich|102|92|83.48°|41.27°|1.0204|
|Sparse / mainstem-dominated|28|27|82.71°|38.61°|1.0295|
|Broad / tributary-rich|165|137|84.50°|41.50°|1.0232|

Physical counts by form overlap at the five shared junctions. Area weights use
the original contributing-area shares and do not substitute for measured flow.
The broad distributions of individual angles are more substantial than these
differences between class medians.

## Original matched comparison

Use exactly the original 22 elongated/broad pairs from 12 HUC4 groups. The source
pair table also contains seven pairs involving the sparse class; those are not
part of this pre-existing elongated/broad contrast. All 22 pairs remain
measurable at all three scales; no pair contains the same physical junction
on both sides.

The table reports broad minus elongated, paired mean differences, with 5,000
resamples of complete HUC4 pair groups. These intervals describe the matched
geographical comparison and do not supply independent field-process replication.

| Primary 250 m metric | Mean paired difference | 95% HUC4-group interval |
|---|---:|---:|
|Incoming angle|−1.30°|−21.79 to +22.82°|
|Area-weighted deflection|+0.03°|−17.21 to +14.90°|
|Downstream sinuosity|−0.0080|−0.1515 to +0.0687|

No primary contrast establishes a class difference; the intervals also do not
establish equivalence. In fact, all six reported geometric contrast intervals
span zero at 100, 250 and 500 m. The main scientific result is the separation
between whole-network form and local junction geometry.

Measurement scale matters. Incoming-angle rankings at 100 versus 250 m have
Spearman rho 0.819, versus 0.576 at 100 versus 500 m; the median absolute angle
change across the latter scales is 17.41 degrees. A longer chord incorporates
additional bends, so there is no scale-free junction angle in this mapping.
All three scales are retained rather than selecting a favourable one.

## Observed mixing data: located, not yet retrieved

[Meem et al. (2025)](https://doi.org/10.1029/2025GL114640) provides a separate
observational route for testing geometry against downstream lateral mixing.
The [Illinois Data Bank record](https://databank.illinois.edu/datasets/IDB-5324086)
and successfully retrieved DOI metadata identify `pmx_all data.csv` and
`pmx_binned data.csv`, describing 150 image-derived mixing events at 43 junctions.
That is the archive's description, not a count independently reconstructed here.

The metadata request succeeded, but both public archive landing hosts failed
with TLS connection errors from this workspace; the published site supplement
was also unavailable. **Zero event observations from that archive have been
analysed in this version.** The access record preserves the actual requests.
Original event CSVs and site coordinates are needed to complete the proposed
observed geometry–mixing relationship. No values were read from a figure or
substituted from a fitted mixing curve.

## Research direction

Keep river morphology at the centre of the study. The useful structural chain is:

**whole-network arrangement → timing and sequence of confluences → conservative
mixing/spreading opportunities → observed DOC response.**

This version directs the next work toward three questions:

1. Does a network spread its major junctions along a long trunk, or concentrate
   them near its receiving point? Reuse the fixed network classes and measure
   junction positions without using DOC to choose examples.
2. Do comparable incoming DOC signals reach those junctions together or at
   separated times? Connect the existing independent branch-path experiment to
   actual simultaneous branch observations where available.
3. How much common downstream channel remains after those inputs merge, and
   how does measured mixing change along it? Retrieve the identified mixing
   curves before estimating any geometry-to-mixing response.

The preceding Blaine co-tracers still supply the carbon-processing link in one
stream segment. Their glucose response decline is not assigned to any of the
three network forms by this local-geometry analysis. This step does not add a
new DOC removal rate, external morphology validation or prediction-training run.

## Products and reproduction

- [Whole networks and actual junctions](figures/whole_forms_and_actual_junctions.png),
  [Chinese figure](figures/whole_forms_and_actual_junctions_cn.png).
- [Form distributions and matched differences](figures/junction_geometry_by_network_form.png),
  [Chinese figure](figures/junction_geometry_by_network_form_cn.png).
- 885 station-scale geometry rows, all original matched comparisons, scale
  sensitivities, physical-junction reuse and actual mapped primary-scale vertices.

```bash
uv run python scripts/fetch_doc_river_confluence_mixing_v1.py
uv run python scripts/analyze_doc_river_confluence_mixing_v1.py
uv run python scripts/plot_doc_river_confluence_mixing_v1.py
uv run python scripts/plot_doc_river_confluence_mixing_v1.py --chinese
uv run python scripts/verify_doc_river_confluence_mixing_v1.py --full-replay
```

The fetch entry point currently records public metadata/access attempts; it does
not claim to have downloaded the event CSVs. Mapping uses cached USGS NHDPlus /
NLDI geometry and the existing directed corridor. Necessary known-angle,
orientation, short-route, local-gap, turning and fixed-pair tests accompany the
analysis. Figure and replay checks are recorded in `verification.md`.
