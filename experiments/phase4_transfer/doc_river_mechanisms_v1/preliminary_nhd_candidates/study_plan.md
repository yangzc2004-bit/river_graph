# River morphology and DOC mechanisms: first observational study

## Scientific questions
1. Does integration of independent monitored tributaries explain downstream
   DOC better than either tributary alone, and when does their asynchrony buffer
   concentration variation?
2. Do differences in terrestrial source environments accompany longitudinal
   DOC changes, after distinguishing new drainage inputs from path properties?
3. Do river-path length, storage exposure and hydro conditions accompany
   attenuation or amplification of upstream DOC signals?

This study starts from real NHDPlus upstream membership and the previously
estimated, geometry-only three-class planform typology. It does not change the
classes or train a new neural model. Follow-up process simulation and eventual
morphology-conditioned model development build on these observations.

## Observation population
Use the deduplicated union of source-training cells from the existing
142/143/144 station roles, identical to `doc_river_planform_doc_v1`. This is a
descriptive source population, not independent geographic/external validation.
Read DOC only for these station-months. Raw WQP samples are filtered by the
existing dissolved, mg/L, uncensored definition, then by the allowed monthly
population; collapse same-day samples to a median. Record negative DOC and
exclude it from mixing/log summaries without removing high DOC values.

## Real confluence combinations
Enumerate every pair of monitored incoming stations at a receiver. Exclude
nested branches (one source lies in the other's upstream network), identical
COMIDs and branches sharing more than 1% of the smaller unique catchment area.
Keep exclusions in the inventory. Locate the first common primary-route reach,
record its distance to the receiver, source-to-receiver lengths, source drainage
coverage and intermediate drainage. Use summed unique VAA incremental areas,
not cumulative `totdasqkm`, which was inconsistent in earlier geometry audits.
Missing mapped geometry does not erase topology or observation availability.

Each eligible monthly combination has at least 12 common source DOC months.
Compare area-weighted concentration proxies, and positive-discharge-weighted
proxies on their own common populations. Flow closure is diagnostic, not a
concentration-conservation requirement imposed on incomplete tributary budgets.
Use a separate same-day WQP subcohort; for its case summaries require at least
12 common days in at least 12 months. Daily mean discharge is a proxy for flow
at the grab-sample time. Use first mean-flow series and existing duplicate QC;
only positive, conflict-free flows enter weights. No date interpolation or
nearest-date widening. Monthly means may not coincide with DOC sampling days.

Compare mixture MAE against the mean of the two individual-source MAEs on
exactly the same records (no outcome-based best-source selection), downstream
and source CVs, branch seasonal/year-adjusted correlation, within-source-range
frequency and signed departure from the mixture. Separate full monthly,
flow-available monthly, same-day and same-day-flow populations. Report all cases.
Subset diagnostics fixed by geography/data: receiver within 20 km of the first
common reach and source drainage coverage >=80%; do not call long compressed
station edges instantaneous confluence mixing.

## Source and path diagnostics
On all connected edges with >=12 common source months, report paired native
and log concentration changes, DOC signal correlation after calendar/year
adjustment, and an additional adjustment for local positive flow and temperature
where >=24 common records support it. Report path length, storage fraction,
additional drainage fraction and source/receiver wetland/forest differences.
Regress paired log concentration change and signal correlation on these fixed
geometry/environment/hydro covariates, standardizing over eligible edges and
using HUC4 block intervals. These are observational pathway diagnostics: a
concentration change can reflect new lateral input as well as processing.
Existing watershed-average StreamCat cover cannot locate wetlands along paths;
spatial source-placement tests require later catchment-resolved data.

## Estimands and uncertainty
Average combinations within each receiver before estimating medians and paired
loss gains; receivers have equal weight. Report receiver bootstrap and HUC4
regional sensitivity, 5,000 draws, seed 42. Repeated triples do not increase the
number of independent stations. Source stations may be shared between receivers
and cross regional boundaries; report shared-station components and do not
equate the regional sensitivity with independent experimental replication.
Also resample shared-station connected components as a conservative dependency
sensitivity; retain receiver and regional estimates alongside it. Pathway
associations include both HUC4 and shared-station-component intervals.
Use the fixed geometry classes plus continuous shape, not DOC-selected classes.
No retrospective choice of winning sampling window, subgroup or model.

### Analytical review, before release
Initial unreported summaries compared the mixture with the average of the two
individual-source errors. Convex averaging alone can reduce that error. Therefore
also report the **unweighted concentration average** as an explicit reference,
and the extra gain from area/flow weights; integration gains alone are not
evidence that hydrological weighting works. This additional diagnostic was added
after inspecting the initial integration summaries, without selecting cases.
One-receiver or one-region intervals are non-identifiable and remain missing.
Missing StreamCat source/receiver cover stays missing with availability flags,
median imputation plus missingness indicators in pathway associations.

## Controlled illustrative simulation
Equal total source input, channel length and event series; change only topology
(chain, balanced branches, elongated tributary network), and compare synchronous
versus asynchronous source forcing. Advect signals with synthetic step delays,
optionally apply first-order removal. Units are normalized, not measured travel
time. Show conservative mixing and processing separately and retain source
budget diagnostics. Simulation demonstrates consequences of explicit assumptions;
it is not fitted evidence that one physical mechanism caused field differences.

## Deliverables and next research
Candidate inventory, paired source records, per-receiver and class summaries,
source/path diagnostics, simulation, readable figures and English research
decision. Existing model results stay unchanged. Next: finer source-location
data and a source-role morphology-conditioned upstream residual experiment if
the observed pathways identify useful information.
