# Whole-network form and DOC: independent field comparison

Date: 2026-10-09

## Scientific answer

**Branching organization can provide additional information about river DOC
level, but this replication does not establish a stable DOC ordering of elongated
versus broad networks, or a general buffering effect on DOC variability.**

The positive finding is concentrated in the predeclared subset without shared
upstream channels with ST357. It is not reproduced in the entire new sample
archive or the common-calendar sensitivity. Report these results together; do
not promote the favorable subset to the primary analysis after seeing it.

The study has now completed the bounded real-data test. Another simulation or
another rearrangement of these same records would not resolve the missing
matched shape comparison. No neural model training is required for this question.

## New observations and real geometry

The NEON-authored laboratory-DOC/absorbance archive (Zenodo 20039155) supplies 27
stream/river candidates. Registered S2/buoy locations were obtained from public
NEON location metadata, including sensor alternatives; site centers were not
used as chemistry outlets. All routine-location alternatives for included sites
resolve to the same receiving NHDPlusV2 reach. The farthest registered point from
its mapped channel is under 71 m in the final cohort.

Geometry was measured for 23 sites. The final morphology comparison contains
**13 sites, 11 upstream-overlap groups and 1,015 observed site-month medians in
2018–2025**. It excludes nine networks with fewer than five mapped reaches, one
with a routine location more than 200 m from its mapped channel, two Alaska
sites without NHD position coverage, and two Puerto Rico sites whose COMIDs are
absent from the local CONUS VAA. All candidates and reasons remain in the ledger.

BLWA, MAYF and TOMB share upstream channels, so they form one holdout/bootstrap
group, not three independent river systems. ARIK, KING and MCDI share upstream
channels with the earlier ST357 geography. Their removal leaves **10 sites and
8 groups**. This is a new laboratory sample archive with an explicitly separated
geographic-overlap sensitivity; it is not an external validation of the DOC
reconstruction neural model.

Complete primary-and-secondary upstream channel geometry was required, with
unsimplified basin polygons. Basin/unique-catchment-area consistency, routing
coverage and coordinate-to-channel checks passed for the included cases. The
old geometry database was opened read-only. StreamCat land-cover and climate
coverage are 100% for all final sites. Small unmapped headwaters remain a material
population restriction, not a reason to interpret the selected rivers as a
representative sample of all NEON streams.

## Existing morphology definitions, new outcomes

Frozen geometry-only standardization and class centers were applied to new
features: four elongated/tributary-rich sites, five mainstem-dominated/sparse
sites and four broad/tributary-rich sites. This nearest-center transfer is a
descriptive approximation to the earlier Ward partition. No DOC-dependent
clustering or class reassignment was performed. All features lie within the old
observed ranges, although some small-network sites have extreme standardized
coordinates. Class boundaries remain a description of a continuum.

Replicates were averaged at exact sample occasions before analysis; relocated
RE samples were excluded. Monthly medians prevent intensive sampling from
creating additional river units. DOC level is the median of twelve calendar-month
medians, giving seasons equal weight. Variability is monthly IQR/median. Each
site contributes one outcome per endpoint.

## Fixed predictive comparisons

Ridge(alpha=10) was used with identical contextual inputs and fit-within-fold
standardization. Context includes basin area, wetland/forest/agriculture/urban
cover, climate precipitation/temperature, coordinates and observed-month count.
Whole upstream-overlap groups were held out. Gains below are reductions in
held-out **site-level** MAE relative to context alone; they are neither monthly
DOC reconstruction gains nor percent changes in real DOC concentration.

| Population | Sites / groups | Footprint, DOC level | Branching, DOC level | Paths, DOC level | All form, DOC level |
| --- | --- | --- | --- | --- | --- |
| Primary, 2018–2025 | 13 / 11 | −12.51% | −2.08% | −10.86% | −20.70% |
| Common window, 2020–2024 | 11 / 9 | −4.77% | +3.18% | −14.79% | −16.81% |
| No ST357 upstream overlap | 10 / 8 | +2.73% | **+13.32%** | −2.69% | +2.10% |
| Whole NEON domains held out | 13 / 9 | −12.76% | −2.13% | −11.80% | −21.45% |

Primary context MAE is 1.1780 mg/L; adding branching gives 1.2025 mg/L, with gain
95% interval [−11.56%, +6.59%]. The common-window branching interval is
[−4.78%, +12.45%]. In the non-overlapping subset, context MAE is 0.9015 mg/L and
branching MAE is 0.7814 mg/L: +13.32% gain, interval [+1.48%, +32.41%], positive
in six of eight held-out groups. Removing any one group's scored predictions
retains +7.67% to +17.57% gain; this influence diagnostic does not refit the
remaining training sets.

The earlier ST357 branching signal was +5.59%, interval [+2.02%, +8.42%], on a
different source-role station-median task. The new non-overlapping result is
consistent with branching carrying transferable concentration information in
some populations. The primary and calendar sensitivities show that the strength
of that information depends on the cohort and observation window. No claim of
universal external replication follows from one favorable sensitivity.

## Variability and direct long-versus-broad comparison

Branching did **not** improve the predeclared variability endpoint:

| Population | Branching gain for monthly IQR/median | 95% group-bootstrap interval |
| --- | --- | --- |
| Primary | −10.04% | [−23.66%, −3.62%] |
| Common window | −12.10% | [−21.64%, −4.48%] |
| No ST357 upstream overlap | −3.70% | [−36.18%, +10.98%] |
| Whole domains held out | −9.31% | [−26.30%, −2.57%] |

The common-window footprint gain for variability is +7.57%, interval
[+0.08%, +18.38%]; it is absent in the primary and non-overlapping results.
Retain it as a sensitivity result rather than evidence of general buffering.

**Zero elongated–broad pairs meet the predeclared simultaneous area, land-cover,
hydroclimate and context-distance criteria.** The direct comparable-environment
shape contrast is therefore unidentified in this archive. Do not relax the
matching after seeing outcomes or treat the empty set as a zero DOC difference.
The raw scatterplots describe observations; they are not adjusted causal curves.

## Meaning for the research

1. The original question about actual network morphology has been tested using
   real channels and independent laboratory samples, without changing it into
   a source-landscape study. Environmental variables serve as comparison context.
2. Branch density and mainstem share remain the strongest candidate structural
   descriptors for DOC level. The favorable non-overlapping result supports
   further field work on branching organization, not a rule that every broad
   network has higher DOC.
3. Neither these monthly observations nor the direct shape matching establish
   the predicted peak-spreading/buffering mechanism. Monthly relative IQR is
   not event pulse width. Coincident discharge and incoming branch DOC curves
   were unavailable; one cannot attribute concentration differences to transport,
   mixing or removal from these outlet observations alone.
4. Adding every structural descriptor reduced prediction performance in this
   small cohort. That is evidence about this fixed diagnostic model's transfer,
   not evidence that river morphology physically harms or has no influence on DOC.

## Inference and source limits

Intervals use 5,000 resamples of whole declared holdout groups, preserving paired
site predictions and months within sites. They are conditional on the fitted
out-of-group predictions; they do not include refitting uncertainty. The 32
declared endpoint/block/population comparisons are all reported, with pointwise,
not multiplicity-adjusted, intervals. Small numbers of rivers, correlated
geographic context and source-archive selection limit generalization.

The archive includes DOC samples represented in absorbance processing, not every
NEON water-chemistry sample. Original lab/sample/detection quality flags and exact
individual-sample coordinates were unavailable. Registered routine locations
and receiving-reach geometry are checked, but individual sample positions are
not retrospectively certified. DOC/Q event mass budgets and arrival mechanisms
have not been observed in this analysis.

## Stop point and next substantive advance

Close this comparison with the full tables and maps. The current conclusion is:

> River branching organization has conditional information about DOC level;
> a stable concentration or variability ordering of coarse footprint types
> remains unestablished across comparable independent rivers.

To move from that statement to an explanatory morphology result requires more
independent, environmentally comparable whole networks with original sample QC,
and, for the transport mechanism, synchronized incoming-branch/outlet DOC and
discharge. Extra classifications or simulated pulses on the same records do not
supply those observations. Keep the existing DOC reconstruction paper's model
performance separate from these ecological findings.

## Reproduction

Run through the project's uv-managed environment:

```bash
uv run python scripts/fetch_doc_river_neon_locations_v1.py
uv run python scripts/fetch_doc_river_neon_context_v1.py
uv run python scripts/build_doc_river_neon_forms_v1.py
uv run python scripts/analyze_doc_river_neon_forms_v1.py --bootstrap-draws 5000
uv run python scripts/verify_doc_river_neon_forms_v1.py
uv run python scripts/plot_doc_river_neon_forms_v1.py
```

Original DOC acquisition and replicate profiling are preserved in
`doc_river_wholeform_evidence_v1`. Raw public metadata/geometry/context downloads
are retained in the new gitignored cache. The first geometry pass encountered
the existing routing helper's edge-free one-reach limitation; the new builder
now inventories that case analytically, then excludes it under the unchanged
five-reach rule. Completed multi-reach cases were retained and no scientific
endpoint was altered. An initial 23-COMID coverage query was preserved; repeating
the full 25-COMID registered-location request returned the same 23 CONUS values
exactly, with two Puerto Rico misses recorded.

Both final figures were rendered and visually inspected. A minor overlapping
site-label issue was corrected without changing data or scales. Verification
recomputed all 32 reported gains from the 470 held-out site prediction rows.

Full pytest: **1,421 passed, 2 skipped**, with eight existing warnings. Ruff:
all checks passed. Historical `audit_artifacts.py --verify`: exit 0, retaining
the known G0 conflict exclusion, historical missing dataset qualifications and
85 historical predictions without sidecars. Those historical limits have not
been replaced with a claim of new verification.

## Primary sources

- [NEON-authored laboratory DOC/absorbance archive](https://zenodo.org/records/20039155)
- [NEON routine location metadata endpoint](https://data.neonscience.org/api/v0/locations/BLUE.AOS.S2)
- [NEON water-chemistry user guide, revision F](https://data.neonscience.org/documents/10179/2838366/NEON_waterChem_userGuide_vF/de1c052b-cce3-ecdb-4459-48c60fbbb5d3)
- [USGS NLDI feature sources and coordinate queries](https://api.water.usgs.gov/docs/nldi/feature-sources/)
- [EPA StreamCat watershed and coverage definitions](https://www.epa.gov/national-aquatic-resource-surveys/streamcat-dataset-readme)
