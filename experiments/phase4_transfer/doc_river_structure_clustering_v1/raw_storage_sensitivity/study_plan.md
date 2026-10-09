# River organization and DOC response

## Question
How do confluence, longitudinal integration and upstream storage organize DOC
concentration, temporal variability, seasonality and concentration–flow response?
This is an exploratory observational study of station-centred river neighbourhoods,
using the completed physical NHDPlus atlas. No neural model is trained here.

## Structure classification (specified before reading cluster DOC results)
Ward hierarchical clustering uses structure only. Candidate sizes are 3–6.
Transform skewed positive quantities, median-impute missing structure values,
standardize and balance four physical feature blocks equally in squared distance:
river scale (order, log area, log slope); branching (junction density and upstream
length at 5/20/50 km); mixing (major-junction fraction at 20 km and tributary
area balance at 5 km); storage (fractions at 5/20/50 km).
Remove redundant features within a block at absolute Spearman correlation >0.90,
retaining the first feature in declared order. Choose maximum silhouette among
partitions whose smallest class has >=20 stations; tie: smaller class count.
If none qualify, report the largest minimum class and flag sparse classification.
Report 100 independent 80% station-subset refits, adjusted Rand index against
the full classification restricted to the subset. Resampling preprocessing is
refitted; this is classification stability, not a DOC confidence interval.
Also retain all candidate labels and geometry-only feature-block ablations.
Names describe structural centroids, never DOC concentration or model error.
Monitoring degree, water quality, ecological cover and climate do not enter
clustering. Physical labels from the previous atlas remain interpretation aids.

## DOC population and response
Use the union of source-training cells in partitions 142/143/144, deduplicated
by station-month. Read no new geographical or external test predictions.
Report stations with >=12 DOC months, >=6 represented calendar months; retain
all eligible/omitted station counts. One station is one descriptive replicate.
Station outcomes: median DOC (mg/L), CV, IQR, source-pool Q90 exceedance fraction,
annual harmonic amplitude and peak month (log1p DOC, season plus centred year),
and season/year-adjusted log1p DOC vs log1p discharge slope (>=24 paired months).
Class seasonal curves average station calendar-month medians equally; harmonic
amplitude/phase do not substitute missing seasons. Report 2,000 station-bootstrap
intervals for class medians. DOC thresholds use permitted source cells only.
Upstream associations use the existing fixed-edge seasonal-anomaly table;
deduplicate source-role repeats, stratify by receiving class and retain all lags.

## Adjusted association
Station-equal regressions relate log1p median DOC, log1p CV, log1p harmonic
amplitude, Q90 fraction and C–Q slope to classes or continuous structure.
Controls: wetland, forest, agriculture and urban cover; long-term precipitation
and temperature; median observed flow and temperature at source DOC months;
flow/temperature availability; median observation year, record span, calendar
coverage, observation count; latitude/longitude and HUC2 fixed effects.
Missing controls are median-imputed with missingness indicators. Bootstrap whole
stations (2,000 draws) for coefficient intervals, retaining original class labels.
Interpret intervals as conditional-on-classification exploratory intervals.
Flow-adjusted models describe structure conditional on hydrology; they are not
total physical effects. Also report a no-local-hydrology sensitivity.
Continuous coefficients are per one station SD in the transformed feature.
Check rank/conditioning, report all fitted contrasts and missing outcomes.
Supplement with 5-fold HUC4-blocked ridge comparison of environment-only,
environment+classes and environment+continuous structure; preprocessing is
fit within each fold, alpha=10 fixed, station-equal MAE.
Display additive spline curves for selected continuous variables over 10–90%
support (5 knots, degree 2); use bootstrap intervals and label response scale.
No causal claim follows from clustering alone. No endpoint from older experiments
is changed, and no new geographical/external DOC test is used for model tuning.

## Deliverables
Re-runnable classification, stability and station-response tables; adjusted
contrasts, structure curves, HUC4-blocked comparison; four publication figures;
English research decision describing the river information suggested for GNN
development. Preserve the physical atlas and all prior model results.

## Methods references
- [Ward hierarchy](https://scikit-learn.org/stable/modules/clustering.html#hierarchical-clustering)
- [Adjusted Rand score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_rand_score.html)
