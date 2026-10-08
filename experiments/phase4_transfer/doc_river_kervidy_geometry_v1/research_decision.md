# A geographically anchored DOC response case

## Result

The field-data search now has a concrete geometry–flow–DOC connection at
Kervidy–Naizin. We acquired the public river map, the official upstream
catchment polygon, the actual monitoring-outlet location and the discharge
companion for the corrected optical DOC period. This is progress beyond the
earlier chemistry-only suitability audit: **81,302 corrected DOC records have
an exact UTC discharge match**, and 32 additional records match an existing
flow observation within two minutes. No values were interpolated and no time
shift was fitted from DOC.

The official catchment polygon measures **4.890 km²**. The delivered river map
contains **4.495 km of mapped river inside that polygon**. The published gauge
is inside the polygon, about 0.76 m from its boundary and 2.44 m from the
nearest mapped river. These are cartographic alignment distances, not a
measurement of routing accuracy. The full river export contains 11.680 km and
extends well downstream of the gauge; using that whole export to classify the
monitored upstream form would be wrong.

The actual map is visibly branched. We retain its geometry rather than giving
it a new form label from its DOC response. It is one catchment, not independent
replication of a narrow-versus-broad effect. The delivered line attributes
contain only `Id` (all zero), without the richer hydrological fields described
by the catalog. Small cartographic gaps at joins and the absence of direction
fields prevent treating the unmodified export as a validated transport graph.

## Observations and method

The [corrected DOC archive](https://doi.org/10.57745/OFOUWE) supplies 81,739
laboratory-corrected optical estimates from October 2020 to September 2023.
These are sensor estimates with laboratory correction, not 81,739 laboratory
analyses. Earlier raw optical estimates remain excluded from the corrected
record. There are 405 corrected DOC timestamps without a flow observation
within the two-minute matching tolerance; they remain in the DOC product.

The catalog-linked public AgrHyS service supplies **100,685 quarter-hour flow
records** in the requested period. Source values are in dm³/s (L/s), converted
by 0.001 to m³/s. The source method describes minute-by-minute water-level
measurements and a three-equation rating curve, rather than direct minute
discharge gauging. Zero denotes non-flowing; 29 delivered observations reach
the reported upper limit of 1,238.414 L/s, allowing the precision of that limit.
Two exports have extra digits within its 0.001 L/s rounding interval. Raw
values are preserved, and none exceeds that interval.

The long-period service pagination was incomplete. Retrieval therefore uses
non-overlapping calendar months, each smaller than a single requested page;
delivered clocks are checked against each scope. The partial initial attempt
is retained locally. This is an acquisition repair, not a change to the DOC
selection or analysis question. The source method and acquisition details are
in `acquisition_notes.md`.

Across the bounded request there are 101,568 nominal quarter-hour timestamps;
883 are absent from the delivered flow series. All 36 monthly objects were
checked against the compact flow file, and an independent direct timestamp
merge reproduced the 81,302 exact DOC–flow matches. The missing records were
not filled.

## Flow-selected response windows

Before the joint analysis, the display rule was fixed as the largest reported
positive flow in each observed calendar year, with a window three days before
and four days after. Equal raw reported maxima use the earliest timestamp.
DOC values do not select the windows. October–December 2020 and
January–September 2023 are partial annual observation spans, not full annual
extreme-flow samples. Both 2021 and 2022 span full calendar years, with source
gaps retained.

| Observation year | Selected reported maximum (UTC) | Corrected DOC bin coverage | Largest DOC gap | Dense joint window | Qualification |
|---|---|---:|---:|---|---|
| 2020 | 29 December 03:45 | 100.0% | 0.25 h | Yes | Partial annual span; multiple flow/DOC pulses |
| 2021 | 21 January 15:15 | 100.0% | 0.25 h | Yes | Multiple pulses in the fixed window |
| 2022 | 19 December 09:15 | 94.7% | 4.75 h | No | Gap interrupts the large DOC response; retained in the figure |
| 2023 | 23 March 21:00 | 99.9% | 0.50 h | Yes | Selected flow maximum reaches the reported ceiling |

All four windows have dense flow records. The dense joint designation requires
at least 90% occupied 15-minute DOC/flow/joint bins and no gap over one hour;
it does not establish a separate complete single storm. The 2022 window is
not swapped for a more attractive second event. The 2023 reported maximum is
censored, so its timestamp and magnitude do not identify the true flow maximum.

The panels show a useful observation: the same mapped network produces
different DOC response shapes across events, including multiple pulses,
sharp rises and extended recessions. The 2020/2021 windows already contain
multiple pulses; a single whole-window DOC maximum would not consistently
refer to the selected flow pulse. We therefore do not release a pooled
DOC-minus-flow peak lag or half-width from these windows. Dense records make
event segmentation possible; they do not make a multi-pulse window one event.

## Meaning for the river-form study

The river-form question is still about **path layout, confluence arrangement
and the shared downstream route**. This case anchors that question in an actual
branched network with subdaily outlet observations. It also shows why a static
form label alone cannot determine a unique DOC response curve. This does not
measure the independent effect of morphology: changing event inputs and water
conditions coexist with fixed geometry.

The next empirical comparison should measure relative arrival and spreading
for flow-defined individual pulses in mapped networks, rather than rank whole
windows by their largest DOC value. Repeated Kervidy pulses can characterize
within-network response variation; independent networks with verified geometry
are needed to test how that variation changes with branch-path dispersion and
shared-route length. Concurrent branch observations would further separate
the incoming DOC signal from downstream mixing. Keep this geometrical focus;
do not replace it with a land-cover classification.

The prior controlled real-network experiments and simultaneous confluence
observations remain separate evidence. The unexplained Arctic cross-year
sequences remain excluded from timing interpretation, and the legacy Arctic
flow access limitation is unchanged. No new model was trained or tuned.

## Sources and reproduction

- [Official upstream basin boundary](https://geosas.fr/geonetwork/srv/api/records/82abb5f4-0ddc-4406-90ec-57b3fb36daf2), 2021 edition, derived using a 20 m DEM.
- [Public river map](https://geosas.fr/geonetwork/srv/api/records/f5ceee60-c8db-4a96-a4f4-a7c922b43c8f), 2011 edition, field-map lineage.
- [AgrHyS public flow catalog and linked service](https://catalogue.theia.data-terra.org/meta/TheiaOZCAR.AGRH_DAT_ore_AgrHys_Naizin).
- [Discharge method description](https://sensorthings.umrsas.inrae.fr/public/sensors/metadonneeshQ20250326.pdf).
- Flow attribution: Fovet et al. (2018), [AgrHyS: An Observatory of Response Times in Agro-Hydro Systems](https://doi.org/10.2136/vzj2018.04.0066).
- DOC attribution: Faucheux et al. (2024), [Kervidy–Naizin corrected sensor dataset](https://doi.org/10.57745/OFOUWE).

Run the fetch script once (existing raw objects are reused and the manifest is
not overwritten), then the analysis and plot scripts in the project's uv
environment. Plot both the English and Chinese variants. Raw downloads stay
local; compact matches, geometry, tables, figures and source records are saved
here. Public maps retain the mandatory credit:
**Source : UMR 1069 SAS INRA - Agrocampus Ouest**.

Validation: 1,370 tests passed, two skipped, with eight existing Shapely geometry
warnings. Ruff passed. The historical artifact audit returned exit 0; its 85
old predictions still have no sidecars, so this is a historical metric check,
not a claim that those predictions have complete identities. Both language
versions of the new map and response figures were visually inspected.
