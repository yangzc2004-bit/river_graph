# Actual confluence geometry and observed mixing distance

## Research question

Which actual junction structures mix incoming water rapidly, and which retain a
separated signal along the downstream channel? Connect this measured operation
to the original elongated, sparse/mainstem and broad-branching network forms.
Keep the original whole-network labels fixed.

The preceding paired-tracer analysis supplies transport and labelled-DOC
responses in one stream segment. This version seeks replication of the
structure-to-mixing link at multiple real junctions. Aerial lateral mixing,
temporal pulse spreading and laboratory carbon processing are complementary
measurements with different units and observational support.

## Public sources and analysis sequence

1. Retrieve the event tables accompanying Meem et al. (2025), DOI
   10.1029/2025GL114640, from the Illinois Data Bank resource
   10.13012/B2IDB-5324086_V1. Retrieve published site/date metadata from its
   publicly linked supplement. Record unavailable files rather than substituting
   values from a plot.
2. Establish the actual junction/event/cross-section grain, measurement units,
   duplicate keys, supported reach length and missing fields. Keep the original
   cross sections and date identifiers.
3. Derive each event's observed mixing response over downstream distance. Use the
   source definition Pmx >= 0.95 for observed complete mixing; retain events that
   remain incomplete at their measured reach end as censored observations.
   Do not replace their lengths by an extrapolated fitted crossing.
4. Compare mixing at fixed dimensionless distances only inside each observed
   curve's support. Distinguish raw crossings and noise-sensitive isolated
   threshold exceedances. Show repeated events within their real junctions.
5. Recover available actual junction geometry: incoming directions, width
   balance and downstream curvature. Use only measured/cached public geometry;
   unavailable geometry remains unavailable. Cross-sectional widths and channel
   centerline geometry have separate definitions.
6. Examine geometry–mixing relationships while preserving repeated events at
   the same junction as a group. Whole-network form remains context; a local
   confluence angle does not relabel a whole network.

## Research products

Original event/site inventory, source-linked response tables, actual geometry
where recoverable, bilingual scientific figures and an English interpretation.
Report what this dataset establishes about mixing, then reconnect that operation
to the existing DOC mechanism chain. No prediction training or re-selection of
the prior DOC models is scheduled.

## Available-network geometry while the event archive is unavailable

The original event CSVs and site supplement have not been retrieved. The
successful DOI metadata request is not a substitute for those observations.
`access_records.json` retains the actual failed public download attempts.

Proceed independently with the already mapped ST357 upstream networks. Reuse
the 295 previously selected area-balanced junctions (selection did not use DOC),
the original three whole-network labels and 22 covariate-selected elongated /
broad pairs. Do not reselect junctions after seeing their angles.

Measure directions over exact 100, 250 and 500 metre centreline arclengths;
250 m is the main local description. Incoming angle is the angle between the
two directions pointing from the junction into the upstream branches. Flow
deflection compares each reversed incoming direction to the downstream chord.
Downstream sinuosity is arclength / end-to-end distance; turning is accumulated
absolute direction change on 25 m sampled chords. Use EPSG:5070 metres and
infer line orientation from the directed corridor rather than cached coordinate
order. Keep mapped gaps, routes shorter than the requested scale and missing
geometries explicit; no extension of a short channel. The maximum accepted
digitization gap is 20 m. Repeated physical junctions and nested catchments are
reported, not counted as independent streams.

The existing upstream contributing-area fraction describes branch balance;
it is not channel width or observed discharge. Form distributions are
descriptive. Original paired differences use HUC4-group resampling (5,000 draws)
and are displayed at all three scales; scale sensitivity is part of the result.
These products characterize actual structure. They do not yet measure a
geometry–mixing or geometry–DOC response in the unavailable public event data.
