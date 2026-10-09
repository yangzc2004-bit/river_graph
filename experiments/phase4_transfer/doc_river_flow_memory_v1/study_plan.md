# River structure, flow response and monthly hydrologic memory

Date: 2026-10-09. Exploratory follow-up after the previous morphology, flow
activation and independent NEON comparisons have been seen.

## Question and model destination

Does actual river branching/path organization explain how DOC responds to
current flow versus the preceding month's flow? The intended model destination
is a structure-conditioned river/history message operator in the existing DOC
residual framework. This study supplies evidence for an operator; it does not
train or claim improvement of the production model.

The earlier flow-activation study already evaluated contemporary C-Q slopes.
The additional evidence here is (1) paired current/previous-month flow on an
identical population, (2) chronological held-out prediction with all nuisance
parameters fitted on the earlier years, and (3) separate structural blocks for
the two measured responses. A previous-month association is hydrologic memory,
not a measured one-month river travel time.

## Sources and population

- ST357 dataset: `data/processed/mississippi_graph_graphfix_st357.pt`.
- DOC is restricted to the existing union of source-training cells from role
  splits 142/143/144, saved in `doc_river_mechanisms_v1/analysis/source_cells.npy`.
- Actual morphology/path measurements are read from the completed
  `doc_river_morphology_effect_v1/analysis/station_morphology_doc_panel.csv`.
- Current and preceding calendar-month discharge must both be observed, finite
  and strictly positive. No filling/interpolation, no preceding DOC requirement.
  Zero/reverse flow remains inventoried but is outside this log-Q analysis.
- Main population: all source months; sensitivities: observed temperature and
  source months since 2009. All three are reported, regardless of result.
- Monthly discharge is the existing aggregate of NWIS daily means (cfs), not
  discharge measured at the exact DOC sample time. Original sample-day alignment
  and event hysteresis are not inferred from these data.

## Response estimation

Fit per station: log1p(DOC) ~ intercept + calendar sine/cosine + year trend
  + log(Q current) + log(Q preceding month); add measured temperature in its
  declared sensitivity. Fit current-only and nuisance-only models on exactly
  the same rows. Natural log Q is centered on fitting rows; slopes are invariant
  to a multiplicative discharge unit conversion.

Require >=24 paired months, >=6 calendar months, >=2 years, full-rank residual
flow design and condition number <=100. Preserve all exclusions. Report current,
previous and combined coefficients and their identifiability diagnostics.

Chronological check: split each station's unique years into earlier and later
halves. Require >=24 training and >=12 held-out months, >=6 calendar months in
each half, >=2 years in each half, and identified training design. All centering,
trend reference and coefficients use only earlier observations. Compare
nuisance-only, current-flow and current+previous-flow prediction on identical
later months. Bound predictions to nonnegative log1p DOC. Score log1p MAE and
native MAE, first average months within station, then stations equally. Resample
whole HUC4 groups 5,000 times, paired, conditional on saved fitted predictions.

## Structure comparison

For current, previous and combined response descriptors, fixed Ridge(alpha=10)
predicts each station's fitted response with five HUC4-held-out folds. Context:
area, wetland/forest/agriculture/urban amount, climate, coordinates, median flow
and record year. Separate increments: footprint, branching, paths, all structure.
Preprocessing is fitted inside each training fold. Average station errors within
fold and folds equally; use the same estimator in paired HUC4 bootstrap. Also
report within-response structure interactions with equal station weight and
HUC4 bootstrap. All comparisons have pointwise intervals, without multiplicity
correction; consistent out-of-region results matter more than a selected sign.

## Independent data and completion

Official NEON continuous discharge DP4.00130.001 is an appropriate independent
source, but its data download endpoint now requires an authorized API token.
Do not seek credentials or replace it with a nearby unverified gauge. Record
this acquisition boundary; public laboratory DOC and geometry remain available.
The existing independent morphology comparison stays complete and unchanged.

Complete this bounded study with reproducible code, an executed notebook,
figures inspected at reading size, independently checked metrics, and an English
research decision mapping each finding to the proposed graph operator. A null
structure result does not imply absent physical structure effects. It means
these descriptors do not yet prescribe a validated dynamic operator.
