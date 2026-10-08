# Acquisition and method notes

The public observation service supports minute equality, but not modulo filters.
The first long-period request returned 10,000 chronological records and an empty
next page although a bounded later-month request returns later observations.
That partial attempt is retained under the raw directory. The final retrieval
uses non-overlapping calendar-month requests: fewer than 3,000 possible
quarter-hour observations per request, below the requested page size of 10,000.
Every delivered timestamp is checked against its requested month and minute.
The service's inconsistent count/next-page fields are not used as completeness
evidence. Missing source months or bins remain missing.

The catalog-linked rating-curve description states that water level is measured
every minute with an OTT Thalimedes encoder, and discharge is calculated from
three rating-curve equations. Thus the minute grid has a documented level
measurement basis; it is not a minute-by-minute direct flow gauging. The same
description specifies an upper reported value of **1,238.414 L/s**: values above
the maximum measured discharge are represented by that limit. Zero means
non-flowing and NA means missing. This is a source censoring convention, not a
new clipping rule. Its occurrence must be reported before interpreting a
flow-selected maximum or timing difference.

Two exported flow values are 1,238.41413437507 L/s: 0.000134 L/s above the literal
method cap, within rounding to its reported 0.001 L/s precision. They are flagged
as at the reported cap, with raw values unchanged. The annual maximum selection
continues to use raw reported flow and the original earliest-tie rule; rounding
is not used to choose a different display event.

Method source:
https://sensorthings.umrsas.inrae.fr/public/sensors/metadonneeshQ20250326.pdf

The catchment layer was discovered in the published GeoSAS WFS capabilities.
Its official metadata identifies the approximately 5 km² Coetdan/Kervidy basin
upstream of the Naizin monitoring location, derived from a 20 m DEM. The river
layer is an older field-map export covering additional downstream reaches. Its
delivered attributes contain only `Id`, all zero; the richer fields described
in metadata are absent. The catchment polygon and observed gauge are therefore
kept separately, and clipping does not create a directed transport graph.

Boundary source:
https://geosas.fr/geonetwork/srv/api/records/82abb5f4-0ddc-4406-90ec-57b3fb36daf2
