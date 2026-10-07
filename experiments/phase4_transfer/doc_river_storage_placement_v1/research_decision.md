# Storage position and DOC pulse overlap in real river networks

River structure changes DOC peaks through two coupled effects: each tributary
pulse is attenuated, and the receiving reach combines the resulting waveforms.
A conservative storage response can weaken both individual pulses while
increasing their combined maximum through greater overlap. This supplies a
specific structural explanation for the peak increases in the preceding whole
network experiment.

## Real geometry and controlled comparison

Of the original 297 station network instances, 295 contain an eligible real
confluence: 102 elongated tributary rich, 28 mainstem dominated sparse, and 165
broad tributary rich networks. Two sparse instances have no two positive area
tributaries and remain explicit exclusions. The geometry rule selects the
largest balanced upstream area junction and actual representative reach
midpoints in its two largest tributary subtrees. These selected subtrees cover
a median 83.15% of the represented basin area. Original morphology labels and
their three mapped examples are retained.

The two branch inputs have identical DOC forcing. Positive constant flow shares
use incremental area; equal flow is a sensitivity. All path means remain fixed,
with pair weighted mean time normalized to 1. Storage redistributes a feasible
part of a segment's translation into a unit gain exponential response. The
locations are controlled placements on actual paths. They do not estimate
the residence times of mapped lakes or chemical DOC removal.

Two representative branch inputs describe a confluence response. They differ
from the preceding experiment's input over every incremental catchment, so the
absolute peak heights of the two studies describe different source scopes.
The same placement experiment is repeated on all 59 existing monitored
footprints spanning 22 receiving stations and 11 shared monitoring systems.

## Matched variance result

For the middle input pulse, SD 0.15, and 50% of the common feasible strength, all
three placements add exactly the same mixture variance. Their outlet means
and SDs are also identical. The following results give each of 295 real
confluences equal weight. Intervals use 5000 HUC4 block bootstrap draws over 62
receiver HUC4 groups.

| Storage position | Mean peak reduction | 95% interval | Central 80% duration change | Instances with peak increase greater than 1% |
|---|---:|---:|---:|---:|
| Earlier arriving branch |4.60%|3.65 to 5.52%|+10.03%|69 of 295|
| Later arriving branch |7.04%|6.39 to 7.64%|+8.97%|9 of 295|
| Shared downstream trunk |9.01%|8.15 to 9.81%|+11.19%|0 of 295|

The paired shared trunk minus earlier branch difference in reduction is 4.41
percentage points, interval 3.82 to 4.98; shared trunk minus later branch is 1.97,
interval 1.36 to 2.55. A common response acts on the complete mixture and cannot
increase its maximum. Branch specific responses can change the overlap of
the constituent waveforms. For the early branch, 148 instances reduce the peak
by more than 1%, 69 increase it by more than 1%, and 78 change by at most 1%.

These distributions also differ within the original forms. Mean shared trunk
reductions are 7.39%, 8.52%, and 10.10% for elongated, sparse, and broad forms.
Early branch reduction has a median of minus 0.07% in elongated networks,
despite a positive class mean 3.22%. Individual geometry and flow balance
remain necessary alongside a form label. Each geometry uses its own feasible
capacity; the class means do not hold absolute added variance constant across
different basins.

## The strength definition matters

At equal flow weighted allocated storage time, the means are 4.97%, 5.15%, and
5.12% for early branch, late branch and shared trunk. All three paired placement
intervals include zero. This sensitivity prevents a universal position ranking.

With variance matched, mean allocated time is 0.04955, 0.06674, and 0.08402
relative units across those positions. With allocated time matched, each
receives 0.05604, but added variances become 0.01140, 0.00785, and 0.00435.
Location comparisons therefore need an explicit strength definition. The
reproducible mechanism is the competition between component attenuation and
mixture overlap.

Source balance also matters. With equal branch flows, the variance matched
late branch mean reduction is about 0.80%, compared with 7.04% under the
incremental area proxy. The area proxy places a larger flow share on the later
branch in 257 of 295 selected confluences. Flow shares are assumptions in this
controlled experiment, and their sensitivity is reported with the geometry.

## Exact decomposition and mapped example

Let E be the sum of the two weighted individual pulse maxima and A the combined
peak divided by E. Then peak equals E times A, and its log change is exactly the
sum of the envelope log change and overlap log change. Conservative kernels
cannot raise E. A higher combined peak requires an increase in A that outweighs
the attenuation of E.

Station 06697100 was selected before outcomes by arrival dispersion times
feasible storage capacity. Its real independent branches are 17.45 and 7.49 km,
with 3.59 km of common downstream path. In the variance matched middle pulse:

| Position | Combined peak change | Envelope log change | Overlap log change |
|---|---:|---:|---:|
| Early branch |+2.66%|−0.1160|+0.1422|
| Late branch |−9.48%|−0.1219|+0.0223|
| Shared trunk |−13.50%|−0.1614|+0.0164|

The early pulse's extended tail overlaps the later pulse more strongly. It
raises the outlet maximum while widening central duration by 6.18%, even
though individual peak capacity falls. This is a direct waveform explanation
on an actual river structure with unchanged inputs and path means.

## Replication and pulse duration

Across the 59 monitored footprints, first averaging within each receiver, the
variance matched means are 2.23%, 1.93%, and 3.50%. Shared trunk minus early is
1.26 percentage points, interval 0.57 to 2.80; shared trunk minus late is 1.57,
interval 0.03 to 2.59. The late minus early interval includes zero. The monitored
cohort retains support for common mixture attenuation but does not establish
the same branch ranking as the larger geometry cohort. Its single sparse
connection and single sparse monitoring system cannot define a replicated
form comparison.

In the larger cohort, common trunk mean reductions are 19.88%, 9.01%, and 3.48%
as input SD increases from 0.075 through 0.15 to 0.30. Longer input pulses already
overlap more broadly, leaving less peak sensitivity to the same storage rule.
All pulse durations, strength fractions and both flow assumptions remain in
the saved tables.

## Research decision

Use this mechanism to describe what river form does: it sets the distribution
of source to outlet paths, the location of shared mixing, and the degree of
pulse overlap after transport. Elongation, tributary abundance and downstream
shared path length become measurable controls on a response, rather than a
standalone categorical explanation.

The next study should assemble these continuous structural controls into a
mechanism profile for each original form, including source arrival dispersion,
shared path fraction, and mapped storage position. Compare these profiles
with the existing DOC fluctuation evidence at the temporal resolution that
the observations support. Preserve source forcing controls in the mechanism
experiments; measured ecological source contrasts can enter a separate
observational adjustment.

## Products and verification

The analysis contains 26904 scenarios, matched placement contrasts, both
cohort and class summaries, the complete inclusion inventory, and representative
source response curves. English and Chinese PNG/PDF figures show actual
projected flowlines and the two response components. All 354 geometries were
replayed; source hashes, snapshots, products and both figure receipts were
verified. Maximum errors are 2.70e-14 for centroid, 8.89e-16 for integrated
anomaly, 2.07e-12 for SD, and 3.34e-16 for log decomposition.

Continuous peak refinement resolves grid switching between symmetric double
peaks. Over the eight geometry selected examples, halving the numerical step
changes peak height by at most 3.45e-15, peak time by 2.78e-8, and central
duration by 8.78e-6. Near equal double maxima retain an ambiguity flag and both
times. See [numerical note](numerical_note.md).

The full suite passes 1189 tests with 2 data dependent skips and 8 existing
warnings; Ruff passes. Historical artifact audit exits 0, retaining its 83
parquet only checks, one reported zero coverage result, and known excluded
corrupt G0 artifact. No neural training was performed in this study.

Reproduce from the repository root with the existing uv environment:

```bash
uv run python scripts/analyze_doc_river_storage_placement_v1.py --bootstrap-draws 5000
uv run python scripts/plot_doc_river_storage_placement_v1.py
uv run python scripts/plot_doc_river_storage_placement_v1.py --chinese
uv run python scripts/verify_doc_river_storage_placement_v1.py --full-replay
```

Primary numerical evidence: [cohort summary](analysis/cohort_summary.csv),
[paired contrasts](analysis/contrast_summary.csv),
[effect counts](analysis/primary_effect_counts.csv),
[scenario metrics](analysis/scenario_metrics.parquet),
[selected real geometries](analysis/selected_footprints.csv).
