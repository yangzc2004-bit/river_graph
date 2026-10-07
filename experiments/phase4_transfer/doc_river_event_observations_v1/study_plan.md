# Real observations for river-structure mechanisms

## Question

Can observed tributary and receiving-stream DOC records resolve staggered
arrival and peak broadening, the mechanisms suggested by our real river
geometry and controlled-pulse analyses?

This study follows `doc_river_structure_profiles_v1`. It does not repeat model
training or change the three existing morphology classes. The relevant
exposure is the arrangement of flow paths and confluences, rather than a
comparison of terrestrial source types.

## Data selection

1. Reuse the existing ST357 sampling audit and inspect its two densely sampled
   connection-months. Preserve station IDs and activity times.
2. Audit the complete public SITES Krycklan stream-chemistry collection,
   following each object's latest version at retrieval. Download daily flow
   for the same monitored catchments and the official linked river geometry.
   Select records by availability, dates and connectivity, before interpreting
   DOC responses. Record collection membership and object versions.
3. Inspect published paired high-frequency studies as alternatives. Keep
   laboratory DOC, calibrated optical proxies and reconstructed/modelled DOC
   in separate evidence categories. An unavailable public file is not a
   negative scientific result.

SITES data are licensed under CC BY 4.0. Attribution is retained with the
original object URLs. Any publication using these data must acknowledge:
"This study has been made possible by data provided by the Swedish
Infrastructure for Ecosystem Science (SITES)."

## Analysis

- Normalize timestamps using each file's stated timezone, preserve original
  timestamps, check duplicate sample times, units and missing-value codes.
- Report station-specific DOC cadence and synchronous sampling overlap.
- Obtain upstream/downstream relations from directed mapped geometry;
  geographical proximity alone does not establish a connection. Use catchment
  nesting only when polygons are verified to represent complete upstream basins.
- Select event windows from discharge and calendar information, not DOC
  maxima. Assess sampling on the rising limb, near the flow peak and on the
  recession, and report gaps relative to the event duration.
- Plot actual sample points. Do not interpolate sparse DOC into an apparent
  high-frequency curve or estimate transport lag below the sampling resolution.
- Assess separate capabilities: event concentration response, longitudinal
  comparison, and replication across network forms. Success on one does not
  establish the others.

## Deliverables

Versioned public-data retrieval, cadence/connectivity/event-coverage tables,
inspectable figures, and a short research decision identifying the strongest
observational next step. Original geometry, simulations and monthly DOC
results remain unchanged. This is a mechanism-data study, not independent
validation of the DOC prediction model.

## Geometry clarification during import, 2026-10-07

The public chemistry metadata polygons do not consistently describe complete
upstream drainage areas: several connected sites have almost disjoint polygons,
and their areas disagree with discharge-file catchment areas. Consequently,
their containment is retained as a diagnostic, not used to reject mapped
upstream routes or claimed as independent corroboration. Stations are inserted
at their measured position inside the published `FROM_NODE` → `TO_NODE` reaches.
Only snaps within 20 m are used; mutually reachable routes are excluded.
Separate upstream branches are checked from this directed topology. No missing
river connections are fabricated. This import clarification uses coordinates
and map metadata, not DOC concentrations.
