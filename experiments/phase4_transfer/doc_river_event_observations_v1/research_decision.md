# Observed confluences and the river-structure mechanism

## Research decision

The next useful experiment is an observed upstream–downstream event comparison.
We have now retrieved real laboratory DOC, daily discharge and mapped stream
paths, and found repeated windows with smaller receiving-stream concentration
variation. This provides a concrete field lead for the mixing part of our
structural explanation. Hourly optical records provide a separate route to
testing timing and waveform changes.

The explanatory variables remain **branch-path differences, confluence
position and shared downstream length**. The original three river forms are
retained. New cases are not assigned to those classes simply because their DOC
responses look different.

## 1. What changed since the previous study

Our original activity-date audit found only two connection-months with at least
three sampling days at both upstream sites and the receiver. That archive
cannot regularly resolve the event timing suggested by the controlled-pulse
experiments.

We therefore retrieved the complete public SITES stream-chemistry collection
and its daily-discharge companions, following the latest individual object
versions available at retrieval. The collection contains **5,057 valid
laboratory DOC values at 16 sites**. Fifteen are in Krycklan; C18 is the separate
Degerö site. Fourteen sites map within 20 m of the published Krycklan streams.
C3 is 59.49 m from its nearest mapped reach; C18 is outside the map. Both remain
in the sampling audit, but neither supplies a directed stream connection.

The median of station-specific sampling intervals is **13.93 days**, falling
to **6.98 days during March–June**. Thus this source supplies denser spring
observations, rather than continuous laboratory DOC.

Sources: [SLU Krycklan data](https://www.slu.se/en/about-slu/organisation/departments/forest-ecology-management/research/disciplines/forest-landscape-biogeochemistry/research/krycklan-catchment-study/data/),
[SITES chemistry collection](https://meta.fieldsites.se/collections/368_zFIGZFTzg1Ynt5tNEGPv),
[daily-flow collection](https://meta.fieldsites.se/collections/2M74g6oPabfeXXoOkaiGbFs2),
and the [SLU-linked mapped network](https://ttiwarir.github.io/krycklan-map/).
Individual object versions and PIDs are retained in `retrieval_manifest.json`.

## 2. Actual river arrangements

Stations are inserted at their measured position inside the real directed
reaches. Paths end at the receiving station, rather than the downstream end of
its reach. This prevents a tributary joining below the station from being
counted as an upstream input. Metadata polygons are spatial context, not
verified complete upstream basins.

There are 24 usable station-to-station directed connections and four
nearest-monitored, separate-branch configurations:

| Upstream pair → receiver | Path A (km) | Path B (km) | Shared path (km) | Shared fraction | Eligible matched windows | Smaller receiver CV |
|---|---:|---:|---:|---:|---:|---:|
| C1 + C10 → C12 | 1.102 | 2.856 | 0.096 | 0.048 | 0 | — |
| C13 + C14 → C16 | 8.400 | 8.169 | 3.132 | 0.378 | 2 | 1 |
| C2 + C4 → C7 | 0.015 | 1.090 | 0.014 | 0.026 | 8 | 8 |
| C6 + C7 → C9 | 2.085 | 2.035 | 1.708 | 0.829 | 1 | 0 |

The shared fraction is common-path length divided by the arithmetic mean of
the two source-to-receiver lengths. It is not area-weighted, a travel-time
fraction or a storage estimate. These partial monitored paths do not account
for every unmonitored input along the route.

This gives us real structural contrasts to study: a very unequal pair close
to the C7 junction, nearly equal long paths to C16, and a predominantly shared
route to C9. They are monitored configurations within one research catchment,
not independent realizations of our original three whole-network forms.

## 3. What the laboratory DOC actually shows

Each receiver's annual March–June discharge maximum selects a ±14-day window.
At least 98 of the 122 spring flow days must be valid. This produces 97
station-year flow windows, of which 20 belong to the mapped confluence
configurations. C12 has no qualifying receiver-flow window under this rule.

Twelve of those 20 windows have at least five sampling days at all three
positions, at least two samples before and two after the flow maximum, and a
sample within two days of it. Eleven also provide at least five one-use,
three-station sampling campaigns with a full timestamp span no greater than
12 hours. The 105 matched campaigns across all 20 candidate windows are
preserved, including the windows excluded from the variation comparison.

For the 11 eligible windows, we calculate each station's sample SD divided by
its mean DOC, using the same campaigns. **Nine show smaller receiver CV than
the arithmetic mean of the two upstream CVs.** Eight of those nine are the
eight repeated windows at C7; the other is C16 in 2018. C16 in 2012 and C9 in
2015 show small changes in the opposite direction. The unit of this result is
an observed window, not a DOC prediction, an independent river network or a
percentage of DOC removed.

At C7 the two upstream concentration series can be out of phase. Their
campaign correlations range from −0.66 to +0.69 across the eight windows.
At C16 the correlations are approximately +0.99 in both eligible years.
This is a useful mixing hypothesis: asynchronous branch concentrations can
produce a less variable mixture than synchronous ones. It remains an
observational lead. Daily flow weights, unmonitored inflows, mean concentration
differences and downstream transformation have not yet been separated.

In particular, a comparison with the **average of two CVs** is not a comparison
with a measured or flow-weighted mixed concentration. The next analysis should
make that distinction directly. Sparse points also cannot determine hourly
arrival lag or DOC peak width; the figures retain the actual points.

## 4. Complementary hourly case

The public Turbolo workbook contains paired upstream San Nicola and downstream
Fitterizzi hourly records. DOC is estimated from corrected fluorescence with
laboratory calibration. The source paper also reports statistical retrieval of
some high-turbidity DOC peaks, and the workbook supplies no per-point recovery
flag. It is therefore a separate optical evidence stream.

Duplicate timestamps arise from overlapping author-selected event windows.
Identical measurements are collapsed. All records at two conflicting upstream
timestamps on 2021-03-20 are excluded; the alternatives are rounded versus
full-precision DOC values, and none is selected as the preferred truth.

The cleaned pair has **422 synchronized hourly records in 15 contiguous
blocks**. Nine contain at least 24 records; **seven span at least 24 elapsed
hours**. The longest has 53 records over 52 hours. These denominators are kept
separate. The plots show the earliest three blocks spanning at least 24 hours,
chosen by record dates and coverage. Published timestamps retain their source
clock because the workbook and README do not state its timezone.

These curves allow the next event-shape analysis. This release does not yet
estimate lag, width or a three-form contrast. The two nested sections also
have intervening inputs, so their differences cannot be assigned exclusively
to network geometry.

Sources: [public workbook and README](https://researchdata.cab.unipd.it/803/)
and the [source study](https://doi.org/10.1029/2022WR034397).

## 5. Next scientific step

1. **Test mixing at the observed confluences.** Pair the actual laboratory
   campaigns with daily upstream flow. Compare the receiving DOC with the
   flow-weighted branch mixture and report flow closure. Separate dilution or
   averaging from additional downstream smoothing. Repeat across years at
   each configuration before comparing configurations.
2. **Test timing in the hourly pair.** Select windows from discharge, inspect
   high-turbidity and possibly reconstructed points, and distinguish complete
   responses from cut-off or multi-peak blocks. Estimate peak timing and
   duration only where the observed resolution and event coverage allow it.
3. **Relate responses to actual arrangement.** Use independent branch-length
   difference, common-path length and position of the junction. Keep the
   original morphology definitions unchanged; a broader class claim needs
   additional independently mapped networks with adequate DOC observations.

The scientific aim is still to explain how river structure organizes the
arrival and combination of DOC signals. Source-landscape contrasts can be
considered as background, rather than replacing that question.

## Reproduction and acknowledgement

From the repository root, with preserved raw files:

```bash
uv run python scripts/analyze_doc_river_event_observations_v1.py
uv run python scripts/audit_doc_river_optical_case_v1.py
uv run python scripts/plot_doc_river_event_observations_v1.py
uv run python scripts/plot_doc_river_event_observations_v1.py --chinese
uv run python scripts/verify_doc_river_event_observations_v1.py
```

Public retrieval is implemented in
`scripts/fetch_doc_river_event_observations_v1.py`; its first retrieval freezes
the collection/object versions. Raw observations and map downloads remain in
the ignored local data directory. Committed analysis tables, figures, source
receipts and code snapshots preserve the observed result and its inputs.

The SITES and Turbolo data are CC BY 4.0 with source attribution. Publications
using SITES must include: "This study has been made possible by data provided
by the Swedish Infrastructure for Ecosystem Science (SITES)." The linked
map's provenance is recorded separately without assigning it the chemistry
collection's licence.

No model was trained, and no existing model result, morphology classification,
test endpoint or previous experiment was changed.
