# River morphology as the focal DOC question

## Scientific question

Do elongated, mainstem-dominated and broad tributary-rich river networks differ
in DOC concentration and variability when their environmental context and size
are comparable? Which differences in channel-path organization could explain
the DOC patterns?

This study restores river morphology to the explanatory foreground. Land cover
and climate are comparison covariates. No new source-placement outcome is the
primary question. The preceding source and flow studies remain useful supporting
evidence, with their results preserved.

## Design, recorded before fitting new DOC comparisons

Use the fixed, geometry-only three-class typology and the deduplicated source-
training union of splits142/143/144. No geographic confirmation or external
model-test predictions enter this study. Retain the preceding location and
coverage screen and station response criteria. Historical results have been
seen; this is an exploratory follow-up, not independent confirmation.

### 1. Compare different forms in similar environments

Primary contrast: broad tributary-rich (class3) minus elongated tributary-rich
(class1). Mainstem-dominated contrasts are retained but likely small-sample.
Match without replacement **within the same HUC4**, using only covariates:
log measured basin area, wetland, forest, agriculture, urban, precipitation,
climate temperature, latitude, longitude and median observation year.

Fixed admissibility: area ratio <=2, wetland difference <=5 percentage points,
forest difference <=20 points, precipitation ratio <=1.25, climate temperature
difference <=3 degrees C, median observation-year difference <=10 years, and
covariate standardized RMS distance <=1. Global minimum-cost assignment uses
an inadmissible cost large enough to prioritize the maximum number of valid
pairs. Identical or nested receiving networks are excluded using actual upstream
COMID membership. Neither DOC values nor any shape-response effect selects a
pair. Report complete matching diagnostics and standardized covariate balance.

Covariate-only preflight found 27 class1/class3 pairs in16 HUC4s before the
non-nesting screen; class2 comparisons had only four pairs each. The primary
geographic level and calipers are not relaxed after inspecting DOC.

Primary response: difference in station median DOC, mg/L. Supporting responses:
log1p median DOC, harmonic seasonal amplitude, DOC CV, observed frequency above
the existing source Q90=10 mg/L, and detrended within-station interquartile
flow response from the preceding study. The same covariate-selected pairs are
used; missing response values are excluded only for that response and counted.
Use 5,000 paired HUC4 bootstrap draws, preserving station pairs. Multiple
responses are exploratory and no significance-based selection is performed.

### 2. Measure the river, independent of vegetation placement

Use every upstream reach's existing shortest directed channel distance to the
receiving outlet and its unique local catchment area. No wetland or forest
weights enter the path distribution. Report coverage, area-weighted mean path
distance/sqrt(measured basin area), distance CV and a fixed20-bin area-weighted
distance-to-outlet distribution normalized by each network's maximum distance.
Distances are reach-midpoint channel-path proxies, not hillslope paths or
measured water residence times. Secondary links and nested-basin dependencies
retain the definitions in the preceding geometry study.

### 3. Hold the input signal constant and change the measured routing kernel

For every actual network, propagate the same unit Gaussian concentration-anomaly
pulse, with constant total flow and uniform input concentration per catchment.
Normalize the maximum channel-path delay to one for every network, keeping the
input pulse, source strength and total flow identical. Only relative path
organization changes. Compare output peak and spread. A no-delay reference
checks that spreading, rather than a change in supplied concentration, produces
the difference. This is a controlled constant-velocity routing scenario, not
simulated or observed field DOC. No reaction-rate fitting or causal inference
from observed class means is attempted.

### 4. Separate geometric information blocks

On the identical eligible station cohort, use fixed Ridge alpha10 and five
HUC4-blocked folds to predict station median DOC. Compare environment+area,
+footprint (basin aspect, network axis ratio), +branch organization (drainage
density, mainstem share), +paths (sinuosity, normalized mean path, path CV),
+all morphology, and the full morphology model with each block removed.
Fit preprocessing only on each training fold. Equal-fold MAE and paired HUC4
bootstrap identify joint and incremental information, not released-model K0
performance. Keep every block result, including adverse differences.

## Mechanism basis

Channel distance distributions connect network geometry with routing under
explicit hydraulic assumptions; they should not be equated with measured
residence times. [Moussa2008, Water Resources Research](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2007WR006118).
Observed DOC patterns along river networks can change with hydrologic conditions
and in-stream processing. [Casas-Ruiz et al.2020, Global Biogeochemical Cycles](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2019GB006495).

## Deliverables

Covariate-selected pair list and balance table; all paired DOC contrasts;
vegetation-independent path descriptors and profiles; identical-input routing
scenario; geometric information-block comparisons; English research decision
and Chinese/English scientific figures. The research narrative remains:
**river form -> flow-path organization/mixing/processing -> DOC level and
variability**, with environmental sources used to interpret that pathway.
