# Field observations for river form and DOC response

## Research result

We acquired two public high-frequency chemistry archives and audited the other
two identified source families. This adds real subdaily outlet traces to the
river-form study. The Arctic archive supplies **69 author-defined storms**;
**58** have sufficiently dense DOC records under the stated record-coverage
screen, and **49** remain provisional DOC record candidates after timestamp
conflicts and unexplained cross-year sequences are set aside. These are repeated
events at two outlets, not 49 independent river networks.

The most consequential finding is a source-record problem: **6,783 ordered
finite DOC values from Oksrukuyik in 2017 appear exactly, in the same order,
under 2022 timestamps**. The same values span about 70.65 days in the earlier
clock and 23.55 days in the later clock. Such a difference could masquerade as a
shorter hydrological or DOC response. We retain the original values and do not
interpret this as physical buffering or a morphology effect. The cause of the
archive assembly is unresolved.

No acquired archive yet supplies the complete combination needed for an
independent elongated-versus-broad event comparison: mapped whole networks,
comparable independent river systems, continuous DOC and discharge, and adequate
coverage of the event rise and recession. No new form ranking is released.

## Arctic records

The [high-frequency chemistry archive](https://doi.org/10.18739/A2JH3D482)
contains 116,266 combined-logger rows. Canonical site/timestamps number 115,858:
408 extra rows arise at 40 duplicate timestamps. Twenty-four timestamps have
conflicting finite DOC values and remain missing in the canonical record;
equal alternatives are coalesced without averaging. There are 79,885 finite,
unambiguous DOC records after that operation. These are calibrated optical
estimates, not individual laboratory DOC measurements.

The [storm archive](https://doi.org/10.18739/A2C53F37J) supplies 39 Kuparuk and
30 Oksrukuyik intervals. All 69 are retained in `author_storm_audit.csv`.
Elapsed and published durations agree within one minute. Six intervals use a bare date at one
boundary; these retain their midnight interpretation and are explicitly flagged.
The chemistry clock is labelled AKDT; the storm clock is not independently
documented, so their alignment remains provisional.

Record coverage uses occupied nominal 15-minute bins, not total row count.
Events must span at least 24 hours, have at least 90% occupied bins and no DOC
gap longer than one hour, including the boundaries. Combined-logger grids vary
between 5, 10 and 15 minutes, and their frequency is not necessarily the DOC
measurement frequency. No new measurements are interpolated. Record-qualified
events need not contain a complete DOC rise and recovery.

The repeated sequences affect Oksrukuyik 2022 and 2023. Beyond the 6,783-value
sequence, the archive repeats 5,384 values from 2018 in 2022, 570 values from
2019 in 2022 and 2,703 values from 2019 in 2023. These are longest exact ordered
sequences with varying values, not similarities of rounded means or ranges.
The two later site/years are set aside for source resolution, while all their
records and storm intervals stay in the audit. Large concentrations are
preserved; no arbitrary concentration ceiling was applied.

This leaves 35 Kuparuk and 14 Oksrukuyik provisional DOC record candidates.
The figure shows the earliest two candidates at each outlet, selected by date
and coverage rather than by peak appearance. They are different storms and
do not establish a difference between river forms.

The chemistry package has **no discharge column**. Its methods and the
[source study](https://doi.org/10.1029/2025JG009594) refer to a separate Arctic
LTER discharge archive. The attempted legacy metadata and current catalog
public-search endpoints returned HTTP 403. Published peak discharge and storm
water yield are summaries and cannot replace the hydrograph. Thus none of the
49 candidates is labelled a complete joined DOC-flow event in this release.
The source study also reports that the Kuparuk outlet and catchment area changed
in 2022; future network extraction must respect the two outlet periods.

## Other public sources

The [Kervidy-Naizin archive](https://doi.org/10.57745/OFOUWE) was retrieved with
its measurement description and checked against the repository file checksum.
It has 315,293 rows, of which **81,739** contain laboratory-corrected optical
DOC, from 14 October 2020 to 6 September 2023. Earlier raw optical estimates
remain distinct from corrected DOC. The source excludes earlier corrected DOC
because of sensor drift. The distributed corrected series has modal spacing
of 15 minutes despite the README describing a 10-minute instrument setting
after 2016. This discrepancy needs clarification before claiming the native
measurement resolution. The file contains no discharge and represents one
catchment outlet.

[NEON's integration guidance](https://www.neonscience.org/resources/learning-hub/tutorials/aquatic-data-product-integration)
distinguishes grab-sample chemistry, fDOM sensors and discharge. Download access
requires an API token under the current guidance. No credential was sought.
Sensor fDOM would need DOC calibration and paired mainstem sensors do not close
all tributary inputs.

The [Krycklan and Yli-Nuortti Dryad source](https://doi.org/10.5061/dryad.wpzgmsbp9)
distributes daily aggregates of calibrated optical DOC. Its four listed sites
include three nested Krycklan catchments and a Finland catchment, without the
C7 receiving sensor required by our branch configuration. The distributed
resolution cannot identify subdaily peak width. This source was assessed from
its public file and method description; its numeric file was not retrieved in
this release.

## Consequence for the morphology study

The mechanistic question remains **how real branch-path differences,
confluence position and shared downstream routes organize DOC arrival**.
Lake-versus-river labels are not substitutes for our mapped elongated and
broad forms. The first four displayed outlet traces do not support a new
classification or a claim that one form smooths DOC more strongly.

The useful next step is a geographically explicit event comparison: obtain
the actual network geometry for each monitored outlet period and the separately
published hydrograph, then select concurrent events by flow and coverage.
Compare response timing and duration against measured path dispersion and
shared-route length, while retaining event input intensity and lake storage
as background variables. A mainstem outlet comparison cannot separate branch
mixing; that part continues to use the observed confluence configurations and
their measured water coverage. Kervidy adds a separate outlet-response case
once its flow companion and distributed-clock processing are documented.

This work narrows the next data acquisition to specific missing objects and
prevents a documented timing artifact from entering the morphology result.
It does not overturn the earlier real-network controlled experiments or the
monitored mixing analysis. No model was trained or tuned.

## Reproduction

Run the fetch script separately for `--source arctic` and `--source kervidy` to
record public package retrievals. Existing retrieval manifests and raw files
are preserved. Then run the analysis script followed by the plot script, with
`--chinese` for the Chinese figure set, in the project's uv environment.
Raw objects are local and gitignored; compact record audits, summary tables,
figures and source identities are stored in this experiment directory.

## Follow-up: actual Kervidy geometry and public discharge companion

The separately versioned `doc_river_kervidy_geometry_v1` follow-up now supplies
the public mapped river network, official 4.890 km² catchment boundary and actual
outlet location, together with 100,685 quarter-hour discharge records.
81,302 corrected DOC records have an exact UTC flow match. Four unchanged
flow-selected windows are displayed, three with dense joint records; one has
a DOC gap and another a censored flow maximum. The chemistry-only audit above
remains intact. This follow-up fills a missing companion for one catchment,
not independent river-form replication or an Arctic flow acquisition.
