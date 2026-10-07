# Hourly response: distinguish a shifted wave from DOC buffering

## Research result

The preserved Turbolo data provide a later downstream flow maximum, but do not
show a consistent broadening of the flow waveform. Published optical-DOC maxima
do not follow a common upstream–downstream offset. This supports measuring water
and carbon timing separately, rather than assigning DOC a travel lag from the
flow maximum. It does not rank the three original river-form classes.

This analysis extends the preceding laboratory branch-mixing comparison. That
comparison showed a smaller calculated mixture CV in seven repeated C7 years;
hourly timing now tests a different part of the proposed mechanism. No DOC
prediction model or training experiment is changed.

## Observed-clock and waveform results

There are 422 paired hourly records in 15 continuous observation segments. Seven
span at least 24 elapsed hours; three of these concatenate several original
author event IDs. Segments and hours are not independent storms or catchments.
The unreported source-clock timezone is retained without assigning UTC.

| Segment | Start | Elapsed hours | Downstream flow-peak offset (h) | Half-height flow lobes | Downstream − upstream dominant-lobe width (h) |
|---|---|---:|---:|---|---:|
| 3 | 2019-11-05 | 51 | 1 | single at both sites | −0.050 |
| 4 | 2019-11-13 | 25 | 2 | multiple | −1.609 |
| 5 | 2021-01-17 | 28 | 1 | single at both sites | +0.008 |
| 6 | 2021-01-23 | 52 | 1 | multiple | +0.004 |
| 8 | 2021-01-31 | 24 | 1 | multiple | approximately 0 |
| 10 | 2021-02-08 | 48 | 1 | multiple | −0.006 |
| 15 | 2021-03-20 | 27 | 1 | multiple | −0.011 |

The two single-lobe long segments have effectively equal widths at hourly
sampling resolution. Small interpolated crossing differences are not
sub-hourly travel-time measurements. Other rows describe the lobe around the
largest sampled flow maximum, not the duration of a multi-storm event.

After aligning at the sampled flow-peak offset, median correlation across the
seven long segments is 0.999986, and median range-normalized regression RMSE is
0.001269 (0.127% of the downstream flow range). Several archived series are
almost exact scaled, one-hour-shifted copies. The data README and accessible
main paper do not explain this numerical regularity or give rating-curve / row
derivation information. The supplemental download returned HTTP 403 during this
follow-up. Therefore the result is a property of the published flow series,
not independently verified hydraulic transit evidence. The diagnostic is
reported rather than assuming two independent input waves.

## Optical-DOC timing and measurement dependence

The [source paper](https://doi.org/10.1029/2022WR034397), section 2.3, states that
DOC above 600 FNU was obtained from flow and preceding rainfall regressions.
The archive lacks explicit reconstructed-row flags. Below that threshold the
values are still corrected fDOM estimates. All 30 site-segment maxima exceed
the source laboratory calibration maximum of 3.88 mg C/L; their amplitudes are
not validated by the threshold screen.

The following are descriptive archive calculations, not additional experimental
validation of the source optical method:

- 82/422 upstream and 107/422 downstream hours enter the reported >600-FNU
  retrieval regime.
- Both published peak samples are outside that regime in 10/15 segments,
  including five of the seven long segments. This does not certify peak timing:
  an estimated neighbor can still change which sample is the maximum.
- Both full half-height support intervals avoid the regime in only **one of
  15 segments**, and in **none of the seven long segments**. Support includes the
  lobe and samples bracketing both crossings; an excluded maximum is never
  replaced by a low-turbidity maximum.
- Between-site DOC maximum differences vary from −23 to +10 hours in the full
  archive. Large offsets in multi-wave records can compare different waves;
  these are not DOC transport-time estimates.
- In the 2021-01-17 segment, flow maxima differ by one hour but published DOC
  maxima are simultaneous. The DOC width-support samples nevertheless enter
  the retrieval regime.
- In the shorter, 23-hour 2021-03-14/15 segment (#13), neither site enters
  >600 FNU. Both published DOC maxima are simultaneous, while flow maxima differ
  by one hour. DOC half-height widths are 2.53 h upstream and 3.63 h downstream.
  This is one useful waveform example, not replication across network forms;
  peak turbidity is 440–480 FNU and peaks exceed the laboratory calibration
  range, so it is not a laboratory-confirmed broadening result.

## What this changes in the river-form study

The geometry hypothesis remains specific: the spread of branch path lengths,
the junction sequence, and shared downstream storage can change which incoming
carbon waves overlap. The present pair does not observe both incoming branches.
It cannot distinguish upstream translation, a second branch contribution and
downstream storage by itself. It also cannot identify carbon removal from a
concentration peak.

Use the controlled routing results as predictions and the laboratory mixing
comparison as field evidence for concentration combination. Use this hourly
case to demonstrate why the missing middle measurement is **branch–branch–outlet
carbon-wave timing**, with flow measured alongside it. Avoid returning to a
land-cover-only explanation or presenting the paired reach as a morphology
classification experiment.

## Next research step

Build a targeted public-observation inventory of synchronized branch A, branch B
and receiver DOC/flow, tied to actual mapped confluences. Prioritize archives
with instrument-quality flags or event laboratory samples and complete rising
and falling limbs. A single upstream–outlet pair is supporting context, not a
substitute for observing both branch signals.

For qualifying configurations, calculate branch peak separation and overlap,
then compare the observed outlet waveform with the discharge-weighted branch
mixture. Relate those quantities to independent branch-path difference and
shared-trunk length. Compare repeated events within a confluence first; across
confluences, retain flow, input amplitudes and coverage as explicit conditions.
This directly tests *which arrangement combines incoming waves and how*, while
keeping the original elongated, broad-branching and sparse forms as context.

## Products

- [Hourly result and quality figure](figures/hourly_response_and_quality.png)
  ([Chinese](figures/hourly_response_and_quality_cn.png)).
- [Chronological examples](figures/hourly_waveform_examples.png)
  ([Chinese](figures/hourly_waveform_examples_cn.png)).
- Complete three-page [diagnostic atlas](figures/hourly_diagnostic_atlas_1.png),
  with every observation segment retained.
- `analysis/segment_comparison.csv`, `site_waveforms.csv`,
  `optical_quality_summary.csv`, `paired_hourly_quality.parquet` and `summary.json`.
- Source-linked replay and synthetic tests distinguish missing crossings,
  multi-wave segments, tied maxima and the reported DOC retrieval regime.

Data: [CC BY 4.0 public Turbolo archive](https://researchdata.cab.unipd.it/803/).
