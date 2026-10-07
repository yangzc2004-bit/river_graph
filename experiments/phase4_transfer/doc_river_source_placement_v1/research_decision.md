# Real river networks: source amount, source placement and DOC dynamics

## Research question and decision

This study asks whether river shape influences DOC partly by organizing where
terrestrial sources enter the network and how far they are routed to a gauge.
It adds local catchment and riparian landscapes to the preceding morphology
and tributary-integration studies. The next scientific priority is **source
placement interacting with hydrologic activation**, rather than adding all six
static placement summaries to the current neural model.

The main result is straightforward: the tested source-placement summaries do
not improve geographically blocked prediction of station median DOC. There
are useful exploratory clues for DOC variability, especially forest placement,
but the recent-period sensitivity does not establish a repeatable seasonal
effect. No neural model was trained or changed in this study.

## What was measured

- EPA StreamCat NLCD 2019 catchment and 100-m riparian percentages cover
  1,087,269 of the 1,110,725 unique reaches in the cached upstream memberships.
- Unique incremental NHDPlus catchment area weights potential wetland and
  forest source landscapes. Cumulative upstream area is never summed across
  reaches.
- Directed channel distance follows primary and secondary NHD connections.
  It is the shortest path from a source reach midpoint to the receiving reach
  outlet, including half the receiving reach length. It is a routing proxy,
  not measured travel time or distance from a soil parcel to a stream.
- Six fixed features distinguish amount from position: wetland and forest
  source-distance ratios, near-outlet excess within 50 km, and 100-m riparian
  enrichment. Distance ratio compares source-weighted distance with the
  drainage-area-weighted distance. Riparian enrichment compares the source
  percentage in the buffer and the same represented catchments.
- Missing land cover is not zero. Catchment raster completeness and represented
  drainage area must each meet the 95% rule. Riparian raster completeness is
  unavailable from the successful API response; represented riparian area is
  reported separately.

There are 339 stations with sufficient mapped landscape coverage. After source
DOC availability and shape-class eligibility, the common diagnostic cohort is
297 stations in 62 HUC4 groups: 102 elongated tributary-rich, 30 mainstem
dominated sparse, and 165 broad tributary-rich. The recent-period cohort has
92 stations in 31 HUC4 groups. DOC summaries use only the 21,459 unique allowed
source-training cells from splits 142/143/144. They do not read geographic or
external model test outcomes.

## Station-location review

The 15 gross mapped/reported drainage discrepancies were reviewed independently
against USGS registered features, coordinates, drainage areas and hydrolocation.

| Outcome | Sites | Interpretation |
|---|---:|---|
| Alternative hydrographic reference broadly agrees with reported area | 2 | Candidates for a separately rebuilt physical study graph |
| Hull Hollow is an independent control stream | 1 | Earlier USGS field documentation establishes that it is not the downstream receiver |
| Scale, position or drainage discrepancy remains unresolved | 12 | Keep out of the primary physical analysis |

For 03075001, the registered feature lies 9.34 km from official gauge
coordinates. Coordinate hydrolocation proposes COMID 3785965 with upstream area
13,504.70 km2 versus reported 13,545.64 km2. For 06884025, the alternative
COMID 4024569 has 9,185.85 km2 versus reported 7,127.65 km2, within the broad
factor-of-two screen. These are candidate references, not completed station-
specific watershed delineations. All split-catchment gauge-basin API requests
failed; failures are retained in `location_sources.json`.

The historical ST357 graph and its predictions were not rewritten. The current
primary cohort still excludes all 15 originally flagged sites. Hull Hollow's
independent-control interpretation is supported by the
[USGS field study](https://pubs.usgs.gov/wri/1985/4197/report.pdf).

## Amount-versus-position reconciliation

The first analysis used the existing whole-watershed wetland/forest controls.
After those results were seen, catchment reconstruction identified large
aggregate differences at the three previously flagged Kansas catchments
06887000, 06889000 and 06892350. The comparisons were rerun using source amount
and position computed from the **same represented catchments** in every arm.
Alpha, folds, source-position features, responses and inclusion rules stayed
fixed. The first outputs and executing analysis code remain in
`initial_watershed_controls/`; the original plan remains unchanged, with the
reconciliation recorded in `study_plan_v1_1.md`.

## Geographically blocked diagnostic

Four Ridge models (alpha 10) were compared on identical stations in five
HUC4-blocked folds. Imputation, scaling and region encoding fit each training
fold. The response is the station median of permitted monthly DOC **means**.
The table uses equal fold weights and 5,000 paired HUC4 bootstrap draws;
station-bootstrap intervals and log1p errors are also available.

| Comparison | All permitted source months: MAE reduction, HUC4 95% interval | Positive folds | Since 2009: MAE reduction, HUC4 95% interval |
|---|---|---:|---|
| Shape added to environmental controls | 9.38% [4.08, 14.12] | 5/5 | -2.44% [-13.98, 5.44] |
| Placement added to environmental controls | -3.29% [-12.31, 1.73] | 2/5 | -0.14% [-6.07, 6.30] |
| Placement added to shape model | -6.21% [-19.09, 0.91] | 2/5 | -2.00% [-5.92, 2.72] |
| Shape added to placement model | 6.82% [1.06, 12.17] | 4/5 | -4.34% [-14.74, 2.36] |

For the all-period cohort, native-scale MAE falls from 1.897 to 1.719 mg/L
when shape is added. Adding placement instead gives 1.960 mg/L; adding it to
the shape model gives 1.826 mg/L. The 9.38% result is a station-summary
diagnostic, **not** an improvement of the current monthly neural/K0 model.
The recent-period sample changes both sample size and station composition;
its different result cannot be assigned solely to temporal mismatch with NLCD
2019. No arm supplies evidence that the six static placement variables add
predictive information beyond the tested controls.

## Dynamic-response clues

Exploratory adjusted regressions control source amount, continuous morphology,
hydro, climate and sampling time. Coefficients below are changes in log1p
response per station standard deviation of a predictor, with HUC4 bootstrap
95% intervals. All tested shape/placement terms and outcomes are retained.

| Association | All permitted source months | Since 2009 |
|---|---|---|
| Forest source-distance ratio -> DOC CV | +0.0342 [0.0116, 0.0614] | +0.0554 [-0.0233, 0.1435] |
| Forest riparian enrichment -> DOC CV | -0.0449 [-0.0779, -0.0099] | -0.0480 [-0.1318, 0.0648] |
| Forest riparian enrichment -> seasonal amplitude | -0.0497 [-0.0750, -0.0265] | +0.0048 [-0.0865, 0.1086] |

Thus, farther-upstream forest sources and less riparian forest concentration
are associated with greater DOC variability in the all-period sample. The CV
directions persist in the smaller recent sample, with intervals spanning zero;
the seasonal-amplitude relationship does not persist. These are mechanism
leads from multiple exploratory outcomes, not verified mediation or estimates
of DOC production, retention or transport coefficients.

The physical maps show a clear distinction that basin-wide percentages miss:
similar source amount can occupy headwaters, near-outlet tributaries or a
continuous riparian corridor. Morphology classes summarize this organization
but overlap substantially. They do not define universal DOC response types.

## Next research step

1. Build a source-only monthly DOC/hydro panel with the same mapping exclusions.
   Analyze within-station DOC anomalies against flow anomalies and season;
   contrast source amount interactions with source-position interactions.
   Keep hydro missingness explicit and avoid treating an unmeasured month as
   a low-flow month.
2. Test whether high flow activates distant sources differently from nearby
   riparian sources. Examine temperature and antecedent flow as alternatives
   to attributing changes to channel distance alone. Use station/HUC4 clustered
   uncertainty and geographically held-out response curves.
3. Develop field interpretable examples only on independently checked
   tributary/receiver station pairs. Compare high- and low-flow behavior where
   temporal overlap supports it; the Hull Hollow example is a control stream.
4. If repeatable dynamic interactions emerge, condition the upstream transport
   branch on source landscape, routing exposure and hydro state. Compare it
   with matched explicit-feature and no-message models to identify the river
   operator's contribution.

This sequence turns the question from which shape has higher DOC into how a
river network connects terrestrial sources, mixes tributaries and modulates
the resulting DOC signal.

## Products and sources

The `analysis/` tables retain inclusions, exclusions, both population periods,
all model comparisons and complete association terms. Four figure families
are available in English and Chinese; maps use actual cached river geometry
and local land cover, with outlet markers and physical scale bars. Their
representatives are chosen on morphology centrality and geometry completeness,
without selection on DOC fit.

Land-cover definitions follow the
[EPA StreamCat readme](https://www.epa.gov/national-aquatic-resource-surveys/streamcat-dataset-readme)
and [metric definitions](https://www.epa.gov/national-aquatic-resource-surveys/streamcat-metrics-and-definitions).
Acquisition follows the
[official StreamCat API](https://usepa.github.io/StreamCatWebServices_Public/).
Station references use the
[USGS NLDI service](https://api.water.usgs.gov/docs/nldi/basin/).
See `validation.md` for executed checks and `README.md` for reproduction.
