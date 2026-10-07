# River source placement and hydrologic activation

## Question
Does a change in discharge alter monthly DOC differently depending on river
morphology and the position of wetland/forest source landscapes? This develops
the dynamic leads in `doc_river_source_placement_v1`; it is not a new neural
model benchmark.

## Population and observations
Use the same ST357 source-training union (142/143/144), 21,459 unique permitted
DOC cells, and the reconciled source-placement/shape panel. Keep its mapping
and landscape exclusions. Join only by unique station-month, using HUC from
the original node metadata. Discharge is the existing NWIS daily-mean aggregate
in cfs; temperature is the existing monthly measurement aggregate. Missing
hydro remains missing. Negative flows may denote reversal in the original
data: exclude them only from this nonnegative, log1p C-Q analysis, preserving
their values and exclusion reason in the panel.

Require at least 24 usable DOC/flow months, six calendar months, two calendar
years and a log1p-flow SD of at least 0.1. Retain all station exclusions.
Primary population uses all permitted source dates; sensitivity populations
are source dates since 2009 and months with observed temperature. Do not
choose stations based on their DOC-response sign.

## Within-station response
At each station regress log1p DOC against centered log1p discharge and its
square, removing that station's intercept, annual sine/cosine and linear year
trend from the response and both flow terms. This separates within-station
changes from differences in concentration between stations. The temperature
sensitivity adds observed temperature to these station-specific nuisance terms.
Do not infer event hysteresis or travel time from monthly aggregates.

Save station-level linear C-Q coefficient, curvature and fitted interquartile
flow contrast, alongside rank, condition, residual variation and sample size.
The contrast is a change in log1p DOC from the station's flow Q25 to Q75;
it is not a percentage change in DOC mass or export.

## Source and morphology moderation
Fit a pooled within-station regression with equal total station weight.
The linear flow coefficient depends on environmental/source-amount controls,
the five existing continuous shape descriptors and the six fixed placement
features. Keep one common quadratic flow term. Imputation indicators and
standardization use station-level covariates; report identification/rank.
Report complete coefficient tables with 5,000 HUC4-cluster bootstrap draws,
seed 42, for all three populations. A companion adjusted class model uses
the three fixed planform classes instead of continuous shape and placement.
Report adjusted class-specific linear-flow responses without recategorizing
the networks from DOC outcomes.

As a separate geographic diagnostic, predict the measured station linear C-Q
coefficient with the four preceding Ridge feature sets (alpha 10), five HUC4
folds, preprocessing fit inside each fold and equal-fold MAE. Report 5,000
paired station/HUC4 bootstrap comparisons. This tests transfer of response
descriptors, not DOC reconstruction at an unmonitored station.

## Interpretation and products
Source-distance or riparian interactions are candidate explanations of
hydrologic response. High flow can increase concentration through mobilization
or decrease it through dilution; no response sign is forced. Compare all-period
and temperature/recent sensitivities before choosing a next model mechanism.
No geographic/external model outcome or non-source DOC label is used. Preserve
the old graph, predictor and experiments.

Deliver monthly panel, station fits/exclusions, response summaries, complete
moderation tables, geographic diagnostics, English/Chinese scientific plots,
reproducible scripts and an English research decision.

## Scientific motivation
[Wen et al. (2020)](https://hess.copernicus.org/articles/24/945/2020/)
distinguish temperature-dependent production from hydrologically regulated
export, including flushing and dilution. [Lupon et al. (2023)](https://hess.copernicus.org/articles/27/613/2023/)
show that spatially organized inflow paths interact with hydrologic conditions.
[Prijac et al. (2023)](https://hess.copernicus.org/articles/27/3935/2023/)
study high/low-flow connectivity in a peatland stream. These motivate hypotheses;
they do not establish the same processes in ST357.
