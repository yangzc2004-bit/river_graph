# Research decision: complete river form and observed tributary overlap

Date: 2026-10-08

## Scientific question and advance

Keep the user's whole-network forms: long, elongated networks; sparse,
mainstem-dominated networks; and broad, tributary-rich networks. The question is
how their pathway arrangement affects the overlap of tributary DOC fluctuations
and the receiving signal. The relevant chain remains:

**whole form -> arrangement of contributing paths -> source-signal overlap -> receiving DOC fluctuations**.

The new advance is to connect complete mapped catchments to **one fixed set of
disjoint monitored upstream inputs per receiver**, including sets of three or
more gauges. The earlier arbitrary two-source connections remain separate.
Observed signal mixing is examined before assigning field peak attenuation to
the preceding controlled routing experiment.

## Observational population

The inventory retains all 297 original station-network instances, representing
295 physical receiving COMIDs. A metadata-selected station represents each
physical receiver in observation summaries; aliases stay in the inventory.
All mapped incremental catchment area is reachable in both routing definitions.

Thirty-two receiving networks have enough common DOC observations for the fixed
frontier analysis: nine elongated, two mainstem-dominated and 21 broad. They
belong to seven complete overlapping-catchment systems. Fifteen receivers have
at least three disjoint inputs; 12 have at least 80% represented catchment area.
Median area coverage is 71.9%, with a substantial range of 3.8–98.8%. Area
coverage and represented path variation are therefore retained for each receiver.

The primary data contain 1,333 common receiver-months and 103 sampled stations.
The role-142/143/144 source-training union is checked against its saved masks;
only those permitted station-month values enter the study. Accepted raw results
reconcile to 9,607 station-month dataset means, maximum absolute difference
0.0000031 mg/L. This is an exploratory ST357 source-role study. Geographic or
external validation predictions and neural models are unchanged.

## What the actual monthly observations show

Remove the shared intercept, annual sine/cosine and linear-year design, retaining
native concentration. Each receiving network has one vote. Intervals resample
complete catchment-overlap systems 5,000 times.

| Quantity | Primary estimate | 95% whole-system interval |
|---|---:|---:|
| Source coherence | 0.264 | 0.213 to 0.313 |
| Calculated mixture variance reduction | 30.5% | 25.7% to 35.4% |
| Asynchronous contribution to that reduction | 23.4 percentage points | 20.0 to 26.4 |
| Observed outlet / upstream-mixture correlation | 0.369 | 0.285 to 0.455 |
| Correlation above calendar-preserving source shuffles | 0.235 | 0.167 to 0.294 |
| log(outlet SD / calculated-mixture SD) | -0.071 | -0.232 to 0.127 |

Tributary departures are positively related but far from perfectly synchronous.
Their fixed area-share mixture has lower variance than the area-weighted average
of individual source variances. Of the mean 30.5-point reduction, 23.4 points
arise from less-than-perfect synchrony, with the remainder from amplitude and
share balance. This is an exact covariance decomposition of observed signals.
Variance reduction under averaging is mathematically constrained; its positive
interval alone is not a field test of river attenuation. It does **not** mean
30.5% lower DOC concentration, removed DOC mass or measured peak reduction.

The mixture still follows receiving DOC more closely than independently shuffled
sources from the same calendar month. This shows contemporaneous information
beyond the retained calendar pattern; shared weather, other inputs and residual
seasonal structure can also contribute. It is not a forecast improvement over
the project's complete model or an isolated physical river effect.

Most importantly, the overall receiving/mixture SD interval spans unity on the
ratio scale. Calculated tributary smoothing cannot be substituted for observed
downstream attenuation.

## Whole shape is not yet separated by these observations

The broad-minus-elongated mixture-buffering difference is only **0.70 percentage
points**, interval **-18.77 to +16.77 points**. The outlet/mixture correlation
difference is -0.012, interval -0.371 to +0.330. All six direct form-contrast
intervals span zero. These use the same system draws for both forms, rather than
comparing separate class intervals. They are unadjusted descriptive contrasts.

All 15 complete/covered-geometry versus signal association intervals also span
zero. For example, entry/branch-length correlation versus source coherence has
point association +0.243, interval -0.110 to +0.731. The two mainstem-dominated
receivers do not establish a general form ranking.

Thus the current observations support studying source overlap and transmission,
but not “broad networks systematically flatten DOC more than elongated networks.”
The original controlled pathway-arrangement results remain a mechanism hypothesis
with identical imposed inputs; they are not replaced by these observational
uncertain associations. More observed receivers within one nested river do not
create more independent systems.

## Routing and actual-date sensitivity

Saved shortest routes retain the same 32 receivers and seven systems, with 1,330
common months. Their mixture reduction is 31.4% [25.9%, 36.6%], asynchronous
contribution 23.7 points [20.0, 26.7], outlet/mixture correlation 0.378
[0.285, 0.467], and real-minus-shuffle correlation 0.241 [0.167, 0.301]. The
receiving/mixture log-SD interval again spans zero. The general signal findings
do not rely on the choice between the two already inspected routing definitions.

The median selected within-month sample span is four days. Exactly common-day
monthly sets yield three eligible receivers in three systems; allowing at most
one, three and seven days yields four, ten and 14 eligible receivers, respectively.
These are different populations and cannot be treated as an alignment-dose curve.

On the same ten receivers and same dates at <=3 days, activity-based mixture
reduction is 36.0%, compared with 33.5% for dataset monthly means on that subset.
Their paired difference is +2.55 points [0.90, 6.39]. This supports retaining the
activity-level follow-up; it does not establish an event delay. All 32 monthly
receivers have fewer than 20 jointly high-source observations, so their Q90
conditional coincidence proportions remain small-sample descriptions.

## A genuine weekly-resolution lead: Loch Vale

The density ledger identifies one network contributing all 81 months in which
every station has at least three distinct sampling days. Its stations are
Icy Brook (401707105395000), Andrews Creek (401723105400000), and The Loch outlet
(401733105392404). The gauges represent 74.5% of mapped area; their mapped
source-to-outlet distances are about 1.06 and 1.42 km. There are **334 exactly
common calendar-day sample sets**, in 116 months and 20 sampled years. Median
same-day UTC span is 2.68 hours. They are not simultaneous or paired water parcels.

The [USGS Loch Vale site account](https://pubs.usgs.gov/fs/fs-164-99/pdf/fs-164-99.pdf)
describes the two tributaries joining above The Loch. This is a lake-influenced
system, not a pure river-routing demonstration. Local NHD receiver/gauge reaches
do not themselves have waterbody tags; station names and site documentation must
therefore remain alongside the geometric inventory. Historical site work also
describes seasonal terrestrial and within-lake DOC sources.
[Baron et al., 1991, USGS publication record](https://www.usgs.gov/publications/sources-dissolved-and-particulate-organic-material-loch-vale-watershed-rocky-mountain).

The density-selected follow-up was recorded after inspecting coverage, before
daily signal calculations. Within permitted common months, select one same-day
activity per station by earliest known timestamp and activity ID. DOC does not
choose the sample. A mean of all accepted activities on each station/day is kept
as a replicate sensitivity.

Calendar/year-adjusted daily source coherence is 0.555, calculated mixture
reduction 20.8%, and outlet/mixture correlation 0.566. The more local within-month
analysis retains **262 common days in 69 months and 19 years**, each month having
at least three common sampling days:

- Source coherence: 0.556.
- Calculated mixture variance reduction: 20.6%, of which 16.5 points are the
  asynchronous component.
- Observed receiving/mixture correlation: 0.462; independent within-year-month
  source shuffles average -0.007.
- Observed receiving SD is **1.60 times** the calculated-mixture SD. In the figure,
  source variance is normalized to 100%, the calculated mixture is 79.4%, and
  observed receiving variance is 204.2%.

Using daily means gives mixture reduction 20.6%, correlation 0.481 and SD ratio
1.59. The basic pattern persists: averaging the tributaries smooths their signal,
but the actual receiving signal is more variable. Unrepresented catchment,
variable flow shares, local inputs and lake processes are plausible follow-up
explanations; these data do not identify their individual effects. No population
CI is assigned to one receiving system and no transit lag is fitted to its outcomes.

The plotted 2010 example is selected solely by its maximum 26 common sampling
days. Lines join actual samples at <=14-day gaps; no high-DOC event is selected
for display. The [USGS quality-assurance account](https://pubs.usgs.gov/publication/ofr20111137)
also records estimated Loch-outlet discharge during October 2005–August 2006;
existing monthly flow ratios are context, not reconstructed mixing shares or
measured residence times.

## Next scientific step

Continue the whole-form mechanism question with an explicit distinction between
**tributary averaging** and **changes occurring on the route to the receiver**.

1. Use the complete-versus-monitored path inventory to prioritize comparable
   elongated and broad systems with high monitoring coverage. Choose systems by
   geometry and sampling opportunity, not by their DOC response.
2. Retain branch-entry positions and branch/mainstem path compensation as the
   structural variables. Ask whether their organization changes observed source
   overlap or the residual receiving response after known tributary fluctuations.
3. Identify lake/reservoir-influenced routes with site documentation as well as
   NHD tags, and analyze them separately from river-dominated cases. The Loch Vale
   weekly sequence is a concrete local follow-up, not a surrogate class comparison.
4. Where synchronized concentration and discharge sequences are available,
   examine flow-weighted inputs and remaining downstream additions before
   estimating event peaks, widths or transit delays. Weekly sampling on 1-km
   routes does not resolve hour-scale water transport.

The study remains centered on **how river form organizes DOC transmission**.
Environmental differences are comparison controls; the new research target is
when a receiving network preserves, smooths or amplifies its measured tributary
signal, and which pathway arrangements explain that behavior.
