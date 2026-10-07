# Measurement definitions after inspecting source tables

## Field support

There are three additions to one stream, each with two locations catalogued at
41 and 61 m from release. These are six pulse/location records and three paired
comparisons within one short segment. They do not replicate the three network
morphology classes. Both specific conductance and laboratory DOC/isotopes are
measured in the original analysed CSVs. The 15 August CSV additionally contains dense
conductivity-only rows: an unassayed DOC row is not a failed DOC assay.

The catalogue/paper distances are 41/61 m. The leachate file also has pre-addition
spatial-survey labels `50.0m.upstreamStn` and `75.5m.downstmStn`. Their mapping
to the catalogue is unresolved. Preserve the catalogue's nominal 20 m difference
as metadata, but do not estimate a velocity or uptake per metre from that distance.
The release-referenced clock and concentration ratios do not require the distance
to be resolved; figures identify the two locations without adding a scale claim.
The study plan's initial 20 m wording therefore describes the catalogue difference;
the raw-table distance discrepancy qualifies its physical interpretation.

Release-referenced time is reconstructed from the original sample date/time and
the recorded release time. Retain actual clock differences; do not reset each
location to its first sample. The author transit summary and manually supplied
travel times are retained as separate columns, not substituted for these clocks.

## Original-point primary analysis

No laboratory value is imputed. Preserve all rows and source record IDs. A
same-clock laboratory replicate is averaged after calculating each replicate's
isotope excess, with its replicate count retained. Conflicting same-clock
conductivity records are flagged and left unresolved, not silently averaged.

Use the author isotope equation R=0.0112372*(1+delta/1000), AF=R/(1+R).
Excess labelled DOC is measured DOC concentration times (AF-AF_background).
It differs from total DOC, whose ambient component can vary substantially.

The following pre-front background ranges are selected from the original clocks
and source processing notes before calculating response contrasts:

| Pulse/site | Conductivity background through minute | Isotope background through minute |
|---|---:|---:|
|8 August upstream|25|25|
|8 August downstream|75|75|
|9 August upstream|30|30|
|9 August downstream|75|75|
|15 August upstream|35|35|
|15 August downstream|75|50|

The downstream 8 August series has only one initial background sample. Record
this and the late drift explicitly. Primary backgrounds are arithmetic means of
the available original measurements, with median backgrounds as a sensitivity.
These ranges are not claimed to be measurements before the actual injection;
they are sampled before the discernible local pulse front.

The author's conservative-carbon reference is the conductivity anomaly times
`doc_mass_13C_mg / (salt_mass * 2100)`, using their archived calibration. The
source code does not provide the underlying conductivity calibration measurements;
the coefficient is an inherited calibration, not independently estimated here.

Primary pulse-core comparison uses original co-observed points within 0–300 min
whose conservative reference is at least 25% of its series maximum. Fit a slope
through the origin: sum(reference*labelled_DOC)/sum(reference^2). Also report
the pointwise ratios, their median, and 10% and 50% pulse-core sensitivities.
This is a concentration-response fraction under the author's calibration,
**not an integrated mass recovery or a biological removal rate**. Each date and
location is reported; no population bootstrap is attached to three dates in one
reach. Median-background results and author-processed results are separate.

## Travel and spread

Conservative pulse shape is summarized on the release-referenced clock, within
the recorded portion of 0–300 min. For each adjacent pair of finite conductivity
measurements separated by no more than 30 min, integrate the nonnegative anomaly
as a piecewise-linear segment. Also check 20 and 45 min limits. Do not connect
through an explicitly missing or conflicting source record. Report the interval
coverage, omitted gaps, boundary/peak ratio, centroid and central 80% duration.
The same algorithm on lab-scheduled co-observed points provides a bounded-window
DOC/reference ratio diagnostic; the conductivity-only rows do not create
laboratory measurements.

Piecewise integration is a stated quadrature assumption between observations,
not generated observational values. No exponential tail is extrapolated and no
complete-pulse recovery is claimed. Raw background drift and sampling endpoints
can affect these bounded-window measures. Timing based on an observed sampled
maximum is distinguished from the interpolated integrated quantiles.

## Author-processed comparison

The pinned `data_process.R` and `data_doc.csv` include substitutions for missing
conductivity, manually edited conductivity/labelled DOC, baseline drift
corrections, interpolation of three upstream leachate DOC points, and forced
late residual values. Some original rows were also filtered by CO2 availability.
Display these separately and count changed, added and missing original values by
matching pulse/site and actual release-referenced time. Their disagreement with
the raw-point derivation is not by itself a code defect; some reflects an explicit
and different processing policy. Never count an author-filled point as an extra
independent field sample.

Upstream/downstream ratios of the core slopes cancel the shared conductivity–carbon
coefficient within each addition. Absolute core fractions still inherit that
coefficient and the background policy.

## Sources

- Hall et al. (2026), [article](https://doi.org/10.1007/s10021-025-01030-2).
- Plont et al. (2025), [CC BY 4.0 observations](https://doi.org/10.4211/hs.988f0d0aa46249b2b654145cf5fbf895).
- [Author processing](https://github.com/robohall/DOC_uptake).
- The article supplement explicitly contrasts measured labelled DOC and its
  conservative reference; our original-point/clock analysis is a new reanalysis,
  not a claim to regenerate the authors' respiration model or all published fits.
