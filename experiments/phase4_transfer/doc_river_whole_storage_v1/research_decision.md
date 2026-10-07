# Whole river form and conservative DOC signal reshaping

## What this extension answers

The analysis returns to complete real upstream networks and the original three
planform classes. It compares **297 station-network instances, 295 receiving
reaches and 62 HUC4 regions**: 102 elongated/tributary-rich, 30
mainstem-dominated/sparse, and 165 broad/tributary-rich instances. This is the
unchanged eligible panel, not a new classification of all 322 classified networks.

All incremental catchments receive identical DOC forcing; constant flow shares
are proportional to unique incremental area. Thus differences here arise from
measured channel paths and their mapped internal storage arrangement, rather
than differences in terrestrial DOC sources. Real paths include the secondary
links already allowed by the preceding analysis. One successor per reach
reproduces every saved shortest distance (maximum discrepancy 7.64e-13 km).

Mapped lake/reservoir reaches with one contiguous physical waterbody ID are
combined into one element on each path; separate waterbodies remain serial
elements. Source entry position determines the traversed portion. The model
allocates f of mapped waterbody travel length to an exponential response, while
retaining the path's original mean time. This is a pathwise response experiment,
not a coupled multi-inlet lake hydraulic model or a residence-time estimate.

## 1. Real shape differences survive the storage extension

The following are equal-instance means for the middle input pulse (SD 0.15),
with reach-midpoint inputs. Peaks are fractions of the input peak; duration
uses channel length / square-root basin area at conceptual common velocity.

| Existing form | Instances | Translation-only peak | Translation-only central 80% duration | Peak with f=0.5 mapped storage | Duration with f=0.5 |
|---|---:|---:|---:|---:|---:|
| Elongated / tributary-rich | 102 | 0.189628 | 2.450863 | 0.188244 | 2.458260 |
| Mainstem dominated / sparse | 30 | 0.454928 | 1.292110 | 0.440305 | 1.305298 |
| Broad / tributary-rich | 165 | 0.245853 | 1.792049 | 0.244345 | 1.799314 |

The **22 original environmentally matched elongated/broad pairs in 12 HUC4s**
provide the direct comparison. Broad minus elongated peak is 0.049577
[HUC4-block bootstrap 95%: 0.033066, 0.079419] without storage and **0.050636
[0.034204, 0.080297]** with f=0.5 storage. The broad response is shorter by
0.526197 [−0.770861, −0.369984] duration units in the latter comparison.
The same 19/22 pairs have the positive peak direction.

The physical interpretation is that different real path arrangements spread
the arrivals of the same input differently. The elongated examples/cohort
produce a lower, longer response than broad networks. These between-class
comparisons preserve the preceding basin-size normalization; they do not
equalize the mean travel time between different basins. Within each network,
the storage operation does preserve every source path's mean.

## 2. Lake/reservoir opportunity varies much more within classes

261/297 instances include some source flow whose selected route traverses
mapped storage. Nevertheless, route-specific storage response is often small.
For f=0.5, the mean percentage peak reductions are **0.8518%, 2.7841%, and
0.6796%** for elongated, sparse and broad forms respectively. Their medians
are approximately zero, zero and 0.0013%. The sparse-class mean is influenced
by a few strong cases: its 95% HUC4 interval is [0.2512%, 7.6513%]. This is
not evidence that every sparse network has a stronger storage response.

Across the complete cohort, **41 instances reduce the peak by more than 1%**,
15 by more than 5%, and 12 by more than 10%. Six increase it by more than 1%.
The largest reduction is 34.05% at 03438500 (elongated), followed by 27.56% at
05357215 (sparse) and 20.12% at 06926510 (broad). Thus strong responses occur
in all three types. The largest increase is 3.34% at 06893890 (elongated).
All percentages compare that network to its own translation-only response.

Added storage variance is nonnegative in every case, but peak height and the
central 80% duration need not change monotonically in a heterogeneous mixture.
A common post-mixing kernel cannot increase the maximum of a nonnegative
signal. Here different source paths traverse different kernels, and fixed-mean
translation/storage redistribution can rearrange the overlap of their modes.
The peak-increase cases therefore motivate a specific follow-up: isolate the
relative position of storage along different tributaries, instead of treating
all lakes as an identical downstream smoother. Peak time is distinct from
mean arrival time; these scenarios do not imply that real reservoirs speed up
river transport.

## 3. Source discretization does not remove the elongated/broad contrast

Every network is also rerun with five equally spaced source positions per
reach, retaining each incremental catchment's total weight. With f=0.5,
the mean peaks become 0.183973, 0.322755 and 0.239749. Sparse networks are
sensitive to where input is placed on their few long reaches, as found in the
preceding experiment. Consequently their midpoint peak should not be presented
as a resolution-independent physical signature.

The fixed elongated/broad difference remains **0.047782 [0.033748, 0.071450]**
with five-point inputs; duration difference remains −0.521267
[−0.765763, −0.363544]. Their contrast therefore survives both mapped storage
and the tested reach-input refinement.

## 4. Why this differs from the two-branch storage experiment

The preceding 59-footprint experiment assigned storage to the complete common
trunk and used a common kernel after mixing. Here storage is assigned only to
actual mapped waterbody portions on every route. Mean mapped-storage share of
the route-time budget is 4.65%, 8.35% and 5.17% in the three classes, with
highly skewed distributions. Serial distinct elements also contribute a sum
of squared times, rather than the square of their combined time. The smaller
cohort-average response is therefore not a contradiction of the earlier
29.26% controlled common-trunk reduction. Neither experiment estimates real
hydraulic retention from mapped length alone. A short mapped reservoir can
have a long actual residence time, which requires independent hydraulic data.

## 5. Research direction

The main idea is now more precise: **river shape reorganizes the timing of DOC
inputs, while storage modifies that timing in a location-dependent manner.**
Shape labels summarize network organization; route dispersion and storage
placement explain the internal variation. Under conservative steady forcing,
all outputs retain the same concentration and integrated anomaly. To study
changes in mean DOC, subsequent work needs an independently specified reaction
or exchange process, rather than interpreting peak smoothing as chemical loss.

Next, test tributary storage position through matched structural interventions:
preserve the same real paths, mapped storage opportunity and total time budget,
but move the same response between early-arriving, late-arriving and shared
downstream paths. Compare arrival overlap and peak changes within networks and
then within each of the original forms. This directly follows the peak-increase
cases without returning to a landscape-source classification. All exploratory
placement experiments should be reported together, rather than selecting the
largest result as a new class definition.

For eventual predictive integration, an upstream time operator can use
measured path delays plus contiguous-waterbody response states, with separate
source support and conservative kernel gain. Its parameters would be trained
and evaluated using hidden-station protocols, not taken from these scenario
values. Targeted event observations around contrasting mapped storage positions
would identify the physical lag and residence scales that monthly DOC cannot.

## Evidence and replication

The 4,158 scenarios comprise 297 × (4 allocations × 3 pulse SDs + 2 allocations
with five-point inputs). Exact identities use the mixture's mean and
input variance + between-path variance + within-path serial storage variance.
Maximum numerical centroid/SD errors are 1.94e-12 / 6.71e-11; integrated
anomaly error is 2.93e-13. Half-step checks on the three fixed examples and the
largest geometric storage-variance instance give peak error at most 1.33e-5
and duration error at most 3.05e-6. Prior midpoint peaks are reproduced within
0.000121, consistent with replacing binned routing by analytic Fourier kernels.

Class and fixed-pair intervals resample HUC4 blocks 5,000 times. They describe
variation in this retained geometry cohort; nested outlets and two repeated
receiving reaches are not independent hydraulic experiments, and the intervals
do not estimate uncertainty in the chosen storage fractions. Within-class
quantiles are distributions of deterministic scenarios, not confidence limits.

Inspect `analysis/scenario_metrics.csv`, `network_descriptors.csv`,
`class_summary.csv`, `matched_contrasts.csv`, `within_class.csv`, and the two
bilingual figure sets. The execution preserves all preceding classifications,
DOC/model predictions and endpoint files. See README for reproduction and
`validation.md` for software checks and historical audit qualifications.
