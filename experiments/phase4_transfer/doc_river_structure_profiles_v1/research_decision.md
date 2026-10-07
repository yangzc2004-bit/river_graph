# River form organizes DOC arrival, mixing and storage

Date: 2026-10-07.

## Scientific conclusion

**The useful ecological meaning of river form is its organization of flow paths:
independent tributaries disperse arrivals, a shared trunk carries their combined
signal, and storage location changes how those signals overlap.**

The three original real forms now have quantitative structural portraits. They
are recognizable outlines with overlapping internal mechanisms. The strongest
matched structural difference is greater arrival dispersion in elongated than
in broad networks. Controlled identical-input routing turns that difference
into broader, lower pulses. Available monthly DOC observations establish real
variation and flow responses, while the form-specific event response remains
a question for finer temporal observations.

This synthesis retains the research focus on river morphology. Land cover,
climate and basin area provide the original matched comparison context; they
do not replace river organization as the scientific variable.

## 1. Three original forms, continuous mechanism profiles

All 297 source-role station instances remain: 102 elongated tributary-rich,
30 mainstem-dominated sparse and 165 broad tributary-rich. They represent
295 distinct receiving COMIDs across 62 HUC4 regions. No classification was
changed using DOC. Two sparse cases lack an eligible two-tributary major
confluence; their whole-network and DOC descriptors remain in the analysis.

| Structural descriptor, station-equal mean | Elongated | Mainstem dominated | Broad |
|---|---:|---:|---:|
| Whole-network path SD / mean | 0.474 | 0.407 | 0.435 |
| Shared trunk / weighted selected source-to-outlet path | 40.5% | 56.6% | 32.1% |
| Whole-network weighted path length within mapped waterbodies | 4.65% | 8.35% | 5.17% |
| Area-proxy source share with some mapped storage along its route | 39.7% | 30.8% | 43.6% |

Whole-network metrics use every positive incremental catchment. Shared-path
metrics use the previous geometry-selected major junction and two representative
source midpoints. Selected tributaries cover a median 83.15% of basin area,
but their two midpoint paths are not a substitute for the full path distribution.
The shared-trunk denominators are 102, 28 and 165 selected corridors.

Interpretation:

- **Elongated, tributary-rich:** more dispersed arrival paths provide a mechanism
  for spreading a simultaneous input over time.
- **Mainstem-dominated, sparse:** the selected common path is a larger fraction
  of the total route. A common deterministic path shifts both signals together;
  it becomes a broadening mechanism when storage or other time-varying transport
  is introduced.
- **Broad, tributary-rich:** more concentrated whole-network arrivals can
  produce sharper controlled outlet pulses. Storage on independent paths can
  alter their overlap before mixing.

The within-class distributions matter. Storage path-share medians are only
0.66%, 0.0085% and 1.12%, far below their means. The large sparse-class mean
is therefore not a typical sparse river's storage exposure. The original broad
display example also has a higher path CV (0.530) than the elongated display
example (0.497), despite the opposite cohort-mean ordering. Individual rivers
need their measured internal structure as well as their outline label.

## 2. Where mapped storage actually occurs

LakePond/Reservoir reaches were traced along all 295 selected station-corridor
instances, representing 293 distinct receiving COMIDs. Two pairs of source
stations share the same receiving reach; they retain their station-instance
weights and are not counted as additional independent river geometries.
Source reaches retain the original midpoint crop; shared reaches occur once.
The branch allocation uses fixed drainage-area shares. A waterbody spanning a
junction may occupy both segments, with one physical waterbody ID retained.

| Mapped position on selected paths | Elongated, n=102 | Mainstem dominated, n=28 | Broad, n=165 |
|---|---:|---:|---:|
| No mapped storage on selected paths | 47 (46.1%) | 17 (60.7%) | 60 (36.4%) |
| Independent paths only | 36 (35.3%) | 5 (17.9%) | 75 (45.5%) |
| Shared trunk only | 4 (3.9%) | 2 (7.1%) | 6 (3.6%) |
| Independent paths and shared trunk | 15 (14.7%) | 4 (14.3%) | 24 (14.5%) |

Storage occurs on 171 selected corridor instances, including 55 with some storage on
the shared trunk. Absence on a selected corridor does not mean absence from
other source routes. These measures are actual mapped channel lengths, not
measured water residence times or volumes.

The preceding placement experiment explains why this position matters:
branch storage changes each component waveform and their overlap, whereas
post-mixing shared storage smooths the combined waveform. Its strength-matching
sensitivities remain part of that experiment; this map inventory does not
assign its simulated peak reductions to actual lakes or reservoirs.

## 3. The same environment-matched rivers connect geometry and response

The original 22 elongated/broad pairs in 12 HUC4s are reused without rematching.
Differences below are broad minus elongated. Intervals use 5,000 paired whole-
HUC4 resamples with equal weight per pair.

| Outcome | Mean paired difference | 95% interval |
|---|---:|---:|
| Whole-network path CV | −0.0432 | −0.0851 to −0.0164 |
| Selected common-path share | −4.51 percentage points | −17.31 to +10.58 |
| Whole-network mapped storage path share | −1.24 percentage points | −3.33 to +0.36 |
| Controlled outlet/input pulse peak | +0.0452 | +0.0201 to +0.0866 |
| Controlled pulse SD, relative time | −0.0409 | −0.0808 to −0.0154 |
| Observed monthly DOC CV | +0.0264 | −0.0582 to +0.0916 |
| Observed station median DOC | +0.406 mg/L | −0.371 to +1.159 |
| Observed seasonal log1p DOC peak-to-trough amplitude | +0.0343 | −0.0729 to +0.1199 |

The controlled response is the saved all-source experiment with an identical
Gaussian concentration pulse, SD0.15, and mean path delay one in each network.
It isolates relative arrival organization. This normalization differs from
earlier area-normalized or maximum-delay-normalized scenarios; absolute pulse
numbers from those experiments are not merged. Sparse-network responses remain
sensitive to within-reach input discretization, as already examined in the
complete-network storage study.

The matched evidence chain therefore establishes **form → path organization →
controlled timing/peak response**. It does not yet establish a form-specific
ranking of actual monthly DOC variability or median concentration. The native
CV contrast is positive even though the unmatched broad-class CV mean is lower;
the matched comparison is the relevant estimate of a between-form contrast.

Conservative routing preserves the integrated anomaly. These peak differences
describe timing and waveform redistribution rather than persistent DOC gain or
removal. The +0.0452 pulse transmission difference is neither a prediction-MAE
improvement nor a measured 4.52% DOC increase.

## 4. Field observations at their actual temporal and spatial scale

The existing season/year-adjusted high-minus-low flow responses are positive
in all three original forms: +0.1165, +0.1493 and +0.1285 log1p DOC, with their
respective HUC4 intervals above zero. The fixed adjusted matched form contrast
in the earlier flow-response study remains uncertain; separate positive class
responses are not evidence for a class difference.

The 59 actually monitored tributary corridors cover 22 receivers and 11 shared
monitoring systems. Their saved continuous associations of branch dispersion,
branch balance and common-trunk share with outlet/mixture variability retain
intervals spanning zero. Those corridors are a different structural footprint
from the complete network; their observations are kept as separate context.

The date audit provides the practical explanation for the next measurement
step: median station sampling interval is 28 days, and three-station samples
span a median four days. Only 20.4% of the 3,026 matched records share the same
calendar day. The five monitored connections with common-trunk storage have
no common month with three or more sampling days at all three stations.
Monthly records can characterize DOC variation and flow-associated states;
they cannot reconstruct an event's arrival sequence and peak width directly.

## 5. Research direction

The scientific story is now concrete:

> River form organizes the arrival and overlap of DOC signals. Path dispersion
> spreads a pulse, shared downstream transport carries the mixture, and storage
> position controls whether component damping outweighs increased overlap.

The next experiment should target this mechanism directly, keeping the current
forms fixed. Select comparable real corridors using geometry, storage position
and synchronized DOC availability before examining event outcomes. A useful
field comparison needs source and outlet DOC on a cadence shorter than the
candidate arrival separation, coincident discharge, and observations spanning
the pulse rise and recession. Compare source pulse area, outlet peak, arrival
centroid and duration on the same events. If existing open data cannot supply
those sequences, identify the missing sampling design explicitly rather than
infer them from monthly DOC and daily flow alone.

For the reconstruction model, these findings motivate a directed temporal
filter conditioned on path dispersion, shared-path proportion and storage
placement, with nonnegative causal weights and unit steady-signal gain. A
matched local-descriptor/no-message control would distinguish the value of
geometry as a covariate from the value of transporting actual upstream signals.
That model experiment is a later use of the structural mechanism, not a new
model-performance result produced by this synthesis.

## Reproducibility

The integrated station table, storage partition inventory, fixed matched
contrasts and both observational contexts have dedicated analysis, plotting
and replay scripts. Figures are available in English and Chinese as PNG/PDF.
All source classes, existing DOC data and original experiment results are
preserved. Canonical station geography comes from the original morphology
panel, keeping four-digit HUC4 codes intact; geographic display metadata from
the old relative-time routing table is not used in grouping.

Validation: 295 real storage partitions and all statistical tables replayed
from inputs; full suite **1,200 passed, 2 skipped**; Ruff and the historical
artifact audit passed. The historical audit retains its previously documented
parquet-only provenance limitations and known corrupt-artifact exclusion.
No new neural training or external-basin validation occurred in this step.
