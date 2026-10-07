# Research decision: real river forms and DOC responses to flow

## Main finding

**Flow changes accompany positive season-adjusted DOC responses in all three
fixed river forms. A reproducible difference between elongated and broad
tributary-rich forms is still unresolved.**

The matched rivers show a larger unadjusted log-DOC change in elongated forms.
That difference becomes imprecise after seasonal/calendar adjustment, and the
native-concentration sensitivity does not retain the same ordering. The next
morphology question should examine identifiable internal routing processes,
rather than assign a universal DOC response to a coarse outline class.

This study keeps morphology central. Ecological and basin-area covariates
provide the comparison background. No source-placement descriptors or neural
model selection are part of this version.

## Observation basis

The inputs are the fixed 297-station real-morphology cohort and 18,688 permitted
source-role DOC observations from the preceding monthly comparison. At these
stations, 39,529 measured positive-flow months within the permitted observation
spans define each station's flow terciles. Months without DOC contribute to the
flow references. The low/high state labels therefore do not depend on DOC.

The final double-precision execution yields **205 eligible stations**, 49
HUC4 regions and **13,918 DOC+flow station-months**. An initial availability
inspection with lower-precision flow references reported 204 candidates;
threshold boundary rounding explains the extra eligible elongated station,
03212500 (25 observed months, seven low and six high in the consistent
double-precision calculation).
The saved hydro-only references and final inclusion ledger are authoritative.
The fixed criteria were retained.

| Form | Eligible stations | HUC4 regions |
|---|---:|---:|
| Elongated / tributary-rich | 74 | 29 |
| Mainstem dominated / sparse | 21 | 11 |
| Broad / tributary-rich | 110 | 40 |

Every eligible station has at least 24 DOC+positive-flow months, six calendar
months, two years, and six low and six high observations. Temperature sensitivity
retains the same 205 stations with its own measured-temperature month subset.
All station response designs are full rank.

The matched primary comparison retains ten original elongated/broad pairs in
six HUC4 regions, with **366 shared pair-months**. Each member is fit on exactly
the same dates as the other member. There is one eligible pair for each
sparse-class contrast, both in HUC4 1013; their results remain descriptive with
unavailable geographic intervals. All 29 original pairs remain in the ledger.

## Within-river responses

The primary adjusted response is high-minus-low **log1p DOC**, from a fixed
within-station regression with state indicators, seasonal sine/cosine and year.
The temperature sensitivity adds observed temperature. Class summaries first
give each station equal weight; 5,000 HUC4 bootstrap draws keep regions intact.

| Form | Season/year-adjusted response | 95% interval | Plus-temperature response | 95% interval |
|---|---:|---|---:|---|
| Elongated | +0.1165 | [0.0275, 0.2120] | +0.1183 | [0.0361, 0.2072] |
| Mainstem sparse | +0.1493 | [0.0372, 0.2573] | +0.1343 | [0.0415, 0.2273] |
| Broad | +0.1285 | [0.0672, 0.2008] | +0.1277 | [0.0642, 0.2012] |

The adjusted point response is positive at 141/205 stations (68.8%). This
supports a general positive flow-associated DOC response in this source cohort.
It does not by itself separate the three morphology classes.

The raw native-scale changes are +0.209, -0.285 and +0.469 mg/L, respectively;
all their class intervals include zero. Raw log responses and native arithmetic
means weight high concentrations differently, and season/calendar adjustment
changes the comparison. A positive adjusted log response should not be rewritten
as a demonstrated increase in each class's unadjusted arithmetic DOC mean.

Observed high-DOC frequency (fixed 10 mg/L) rises by 1.91 percentage points
[-1.45, 5.45] in elongated rivers, 4.17 [0.62, 9.14] in sparse rivers, and 4.23
[1.56, 6.94] in broad rivers. These are within-class summaries from different
station populations; direct form contrasts are evaluated using the matched
pairs below.

## Matched differences in response

Values below are **broad high-minus-low response minus elongated high-minus-low
response**, with equal weight per pair. They compare changes within each river,
so a fixed difference in its mean DOC level is not the endpoint.

| Metric | Broad minus elongated | HUC4 95% interval |
|---|---:|---|
| Raw log1p response | -0.1565 | [-0.3175, -0.0037] |
| Season/year-adjusted log1p response (primary) | -0.0981 | [-0.3880, +0.1112] |
| Plus-temperature log1p response | -0.1064 | [-0.3814, +0.0922] |
| Raw native DOC response | -0.331 mg/L | [-0.997, +0.387] |
| Season/year-adjusted native response | +1.624 mg/L | [-1.369, +4.864] |
| Plus-temperature native response | +1.474 mg/L | [-1.371, +4.556] |
| Observed high-DOC frequency response | -1.42 percentage points | [-7.97, +6.25] |

The raw log contrast favors a larger response in elongated rivers, but the
predefined primary adjusted contrast includes zero. Four of ten adjusted pair
differences are positive. Omitting one HUC4 gives adjusted differences from
-0.224 to +0.030, and the native adjusted sensitivity changes direction.
The aggregate evidence does not establish stronger buffering in broad networks.

The within-pair dates are the same, but each station's locally defined low/high
states can occur in different months. A joint-state check retains ten pairs,
88 common-low and 91 common-high pair-months. The B-minus-A DOC gap changes by
-0.028 mg/L [-0.874, +0.636] between joint high and low states. The corresponding
log gap change is -0.1527 [-0.3330, +0.0238]. Both geographic intervals include
zero. Joint calendar counts and annual sine/cosine balance remain inspectable.

## Internal morphology measurements

The fixed association models relate each measured adjusted log response to
form labels or separately added footprint, branch and path blocks. They control
basin area, ecological context, the observed log-flow change and HUC2. One
eligible station, 06607500, lacks agriculture, urban, precipitation and climate
temperature covariates; the explicit covariate ledger leaves **204 stations**
in each explanatory model. Every bootstrap draw refits the association model.

All predefined morphology-feature intervals include zero in both response
populations. The class3-minus-class1 coefficient is -0.0111 [-0.0995, +0.0694],
or -0.0189 [-0.0986, +0.0586] with temperature-adjusted responses. Branch density,
mainstem share, normalized path length and path dispersion also fail to identify
a stable modifier in this version.

Sinuosity retains a negative point association in both fits: -0.0477
[-0.0972, +0.0098], and -0.0589 [-0.1087, **+0.000029**] in the temperature
sensitivity, per station-population SD. The small positive upper bound must
remain visible; rounding it to zero would incorrectly strengthen the evidence.
This is a routing/processing hypothesis for a dedicated follow-up, rather than
a demonstrated effect or a selected winning mechanism.

## Scientific direction

1. **Keep the real outlines and matched catchments fixed.** The new evidence
   answers the within-river flow question and supplies actual response cases;
   it does not motivate reclassifying rivers using DOC outcomes.
2. **Resolve the structural operation.** Use the existing identical-input routing
   experiments and observed upstream/downstream data to distinguish tributary
   arrival synchrony, path-length dispersion and channel exposure. A coarse
   broad/elongated label combines those different properties.
3. **Make a focused routing/processing comparison.** In a new version, hold
   source input and drainage area constant and separately change path geometry
   and a stated channel-processing scenario. Compare conservative mixing with
   processing before attributing a DOC level change to geometry. Preserve both
   positive and negative observed response examples.
4. **Match temporal resolution to the question.** Monthly DOC supports the
   state contrasts above. Observed arrival times, transient concentration peaks
   and hysteresis need contemporaneous DOC observations at finer resolution;
   daily flow alone cannot supply that missing DOC information.

The resulting storyline is: **river forms organize pathways; the observed
consequence must be tested through those pathways and hydrologic states.**
Current evidence shows the common flow response and its geographic variation,
while the independent morphology contribution remains a focused open question.

## Interpretation and reproducibility

This is exploratory ST357 source-role analysis reusing earlier cohorts, not an
independent external validation. Geographic intervals are conditional on the
fixed station response fits and flow references; association models are refit
in each draw, without a multiple-comparison correction or monthly measurement
error model. Local terciles represent different absolute discharges. No missing
DOC observations were reconstructed for these response estimates.

No existing neural model was trained or tuned. `README.md` gives the analysis,
figure and replay commands; `verification.json` and `validation.md` record the
completed checks. Earlier morphology, routing and DOC prediction results remain
available as separate evidence.
