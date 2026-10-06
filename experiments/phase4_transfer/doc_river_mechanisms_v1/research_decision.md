# Research decision: tributary integration, source contrasts and river processing

## Research direction
Continue investigating **how network organization changes the integration of
terrestrial sources and the transmission of DOC signals**. The first empirical
study and controlled simulation are complete. This is source-cohort mechanism
development; no DOC reconstruction model has been retrained and no geographic
or external model test has been used.

## What the data now support

### 1. Multiple tributaries carry complementary predictive information
After the independent gauge-location screen, 133 source-pair/receiver
combinations have at least 12 common DOC months, covering 40 receivers and
24 HUC4s. The flow-weighted population has 93 combinations and 36 receivers.
These are not 133 independent confluences: there are 17 shared-station components.

| Common-month diagnostic | Area weights | Positive flow weights |
|---|---:|---:|
| MAE reduction vs mean of the two individual-source errors | 19.44% | 19.10% |
| Shared-station-component 95% interval | 12.32–25.16% | 7.21–27.01% |
| Extra reduction vs equal-concentration averaging | 8.39% | 8.18% |
| Shared-station-component interval for extra reduction | −4.37–14.41% | −9.86–16.16% |

Combining tributary observations is useful in this diagnostic. The first gain
partly reflects arithmetic averaging and must not be described as a gain from
hydrological knowledge alone. The added value of the particular area/flow
weights is not robustly established. These are proxies computed from observed
source concentrations, not improvements of the current neural reconstruction
model or evidence of causal effects of river shape.

### 2. Buffering depends on timing and source completeness
Mean receiver-minus-source-average CV is −0.045 for the area population and
−0.060 for the flow population. Receiver and HUC4 intervals include zero;
shared-station-component intervals do not. This supports further work on
buffering but does not settle a universal downstream reduction in variability.
Only about 43%/42% of downstream observations fall between the two monitored
source concentrations. The median monitored-source drainage coverage is 26%.
Most combinations therefore integrate incomplete source budgets and long river
paths, rather than measure two-stream conservative mixing at a junction.

Strict same-day records produce 16 area-weighted combinations at two receivers
and six flow-weighted combinations at one receiver. They share one observation
component. No independent same-day, near-complete confluence remains after
location screening. Two near-complete receivers have monthly records; their
large proxy gains are examples, not replication across network classes.

The illustrative routing experiment holds eight source inputs, total flow and
14 equal-length reaches fixed. Different routing arrangements change timing;
asynchronous sources can cancel seasonal variation, and topology can also
align event pulses. First-order removal changes mean concentration according
to path exposure. Consequently, “more tributaries always buffer DOC” is not
the appropriate mechanism. **Source synchrony, arrival-time distribution and
processing exposure jointly determine the downstream response.** The delays
and removal rates are assumptions, not field estimates.

### 3. Source environment and processing can be distinguished with better inputs
The screened longitudinal analysis contains 189 observed edges, 103 receivers,
41 HUC4s and 37 shared-station components. Ninety edges support paired calendar
and local-hydro adjustment; median DOC correlation changes from 0.342 to 0.315.
The common hydro context explains some, but not all, observed co-variation.

Receiver-averaged multivariable associations show:

- Temperature contrast accompanies a positive longitudinal log1p DOC change
  (coefficient 0.086 per receiver SD; HUC4 interval 0.004–0.143; shared-station
  interval 0.018–0.157). This is an exploratory association among correlated
  contrasts, not a controlled temperature experiment or a measured reaction rate.
- Greater storage-path exposure accompanies lower upstream–downstream signal
  similarity (−0.050 per SD; HUC4 interval −0.103 to −0.003), but the
  shared-station interval includes zero (−0.135 to 0.027). Investigate lake and
  reservoir processing with finer hydro information.
- Wetland/forest contrasts, path length and added drainage do not yet isolate
  a consistent independent effect. Watershed-average cover cannot locate the
  DOC source patches relative to the river or measure their hydrologic connection.

These pathway associations are hypothesis-generating, across several outcomes
and covariates. Retain every coefficient in the tables. Pairwise concentration
changes include lateral additions, dilution and transformations.

## A consequential finding: some monitored stations are assigned to wrong reaches
The near-complete candidate involving Hull Hollow Creek (`03201720`) had
source-flow sum/receiver-flow ratio 8.88. Official USGS metadata reports
drainage areas of 0.98, 1.01 and 0.22 square miles for the two presumed sources
and receiver. The actual report describes Hull Hollow as a separate control
stream; the downstream sampling station is `03201722`, absent from ST357.
Thus that candidate is not a true confluence observation.

Sources: [USGS WRI 85-4197](https://pubs.usgs.gov/wri/1985/4197/report.pdf),
[USGS Hull Hollow metadata](https://www.waterqualitydata.us/provider/NWIS/USGS-OH/USGS-03201720/).
Expanded USGS metadata were obtained for every ST357 station: 327 mapped areas
are broadly consistent, 15 differ by more than a factor of two, and 15 have no
official drainage area. A discrepancy is a location-review flag, not proof that
every flagged reach is wrong. The new mechanism cohort screens these cases and
impossible reported drainage ordering. All pre-screen outputs are preserved in
`preliminary_nhd_candidates/`; historical graph/model products remain unchanged.
Future structure-based interpretations should use the location-reviewed subset.

## Next work, in scientific order
1. **Build a site-checked confluence observation set.** Review the 15 gross
   mapping discrepancies and reconcile gauge locations with actual stream names,
   reported drainage and finer hydrography. Expand beyond the ST357 modelling
   cohort to nearby monitored tributaries/receivers selected by independent
   topology, temporal overlap and source-budget coverage. The Lake Hope report
   identifies `03201722` as one targeted candidate. Keep DOC outcomes out of
   station/location selection. Obtain synchronized DOC and instantaneous/short
   interval flow where available; retain daily means separately.
2. **Represent terrestrial source placement.** Add catchment-level wetland,
   forest, soils and riparian cover along connected tributaries. Compare equal
   overall land cover with different source-to-channel and source-to-outlet
   distance distributions. This addresses where DOC enters, beyond broad shape.
3. **Estimate transport/processing context.** On site-checked source/receiver
   pairs, use river length, waterbody exposure, flow and temperature; compare
   added-drainage-dominated and storage-dominated paths. Distinguish signal delay
   from net concentration change; monthly lags are not physical travel time.
4. **Integrate the useful mechanisms into the current model.** Compare a local
   model with continuous morphology/source-placement features against the same
   model plus a real upstream branch. Condition the upstream operator on source
   support, source drainage coverage, source synchrony and processing context;
   retain no-message and matched regional-source controls. Develop on source
   roles before new geographic confirmation. No automatic mass-conservation
   penalty on incomplete source budgets.

The working scientific account is:

> River morphology organizes DOC source connections, arrival times and processing
> exposure. Predictive river information should therefore be conditioned on the
> sources actually represented and the hydro conditions under which they connect.

## Reproduction
Run the metadata fetch (cached), analysis with 5,000 draws, verification and EN/CN
plots through the uv environment. Eight new scientific tests cover branch
independence, the metadata contradiction, hidden DOC, mixing identities, joint
receivers, shared stations and synthetic conservation. Full suite and historical
audit results are recorded in `validation.md`. Figures show real observed data
except the explicitly labelled controlled simulation.
