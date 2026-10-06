# Whole-network planform and DOC: analysis plan

## Question
Do elongated, mainstem-dominated and broad tributary-rich upstream networks
show different DOC levels, seasonal dynamics and useful upstream information?
Use the completed geometry-only typology without changing classes or examples.

## Data and roles
Geometry comes from `doc_river_planform_typology_v1/analysis/station_classes.csv`.
Describe DOC using the deduplicated union of source-training cells from the
existing 142/143/144 roles. This union is a descriptive development population;
a source station in one split may be validation in another. Do not interpret
it as a new independent confirmation, or read geographic/external test results.
Keep unclassified and unavailable geometry cases in the inclusion table.
All response summaries use the existing minimum of 12 observations and six
calendar months. DOC does not change shape classes.

## Analyses
1. Station median DOC, CV, IQR, annual peak-to-trough log1p harmonic amplitude,
   source Q90 frequency and season/year-adjusted concentration-flow slope.
   Station-level summaries and calendar curves retain observation counts.
2. Adjusted class contrasts and five continuous morphology measurements:
   log basin aspect, log network axis ratio, log drainage density, mainstem share,
   and mainstem sinuosity. Include log measured polygon area, cover, climate,
   local flow/temperature, geography, HUC2 and sampling covariates. Also report
   a version without local hydro adjustment. Report all coefficients.
3. Five HUC4-blocked folds with fixed Ridge alpha=10 compare environment plus
   basin area, the same inputs plus classes, and plus continuous morphology.
   Target is station median DOC, not monthly reconstruction. Preprocessing is
   fitted on each training fold. Use paired 5,000 station resamples and a HUC4
   block sensitivity. This is an explanatory prediction diagnostic.
4. Reload existing matched upstream/both-direction/no-message KGML validation
   predictions for strict temporal and spatial leave-out scenarios. Compare
   paired errors by planform and actual same-month visible upstream station
   support. These are historical development diagnostics, not released-model
   improvements. Preserve identical query cells and average seeds first.
5. Relabel the existing fixed, connected source-edge DOC anomaly associations
   by receiving planform, keeping the same edge population across lag buckets.

## Uncertainty and interpretation
Use 5,000 resamples, seed 42, retaining stations with all their months. Report
HUC4-block intervals for adjusted associations and blocked predictions. Nested
basins can extend across HUC4; regional blocks reduce but do not remove spatial
dependence. Class boundaries were estimated from geometry and are held fixed.
Classes with fewer than ten contributing stations are flagged, not hidden.
Q90 frequencies describe observed samples, not every calendar month.
Plot actual units and retain continuous shape variables alongside broad classes.
These analyses establish associations and information value, not a physical
effect of changing river geometry. No neural training or model selection occurs.

## Outputs
Inclusion, station and class responses, adjusted associations, geographic
diagnostic predictions, paired message gains, upstream associations, figures,
source file identities and an English research decision. Existing results stay
unchanged. New research mechanisms follow from the combined evidence.
