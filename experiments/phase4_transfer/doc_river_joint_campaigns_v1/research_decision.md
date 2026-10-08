# Actual river arrangement and DOC buffering across water states

Date: 2026-10-08.

## Main finding

**A receiving stream can smooth the seasonal DOC pattern while retaining, or
amplifying relative to the measured inputs, fluctuations that depart from that
pattern. The outlet response also changes between low and high flow at the same
mapped confluence.** These observed distinctions make a fixed “broad river equals
strong buffer” rule inadequate for the current research question.

The useful next structural representation is the arrangement of separate paths
and the common corridor, considered together with water state and incoming
signal coordination. A long common corridor alone does not determine the observed
response. This follows the original morphology question through actual river
arrangements; it introduces no new land-source classification.

## Observation population

The complete saved laboratory archive supplies 1,052 matched three-site campaigns
across four real Krycklan configurations. Of these, 563 have complete daily flow;
excluding both receiving-date duplicates at C7 leaves **561 eligible
configuration-campaigns on 311 calendar dates**, representing 1,622 distinct
site/timestamp laboratory sample identities at 11 sites. The former numbers count
comparisons, not independent sampling activities or rivers.

The exact intersection gives **47 shared dates in 2014–2017 at all four
configurations**. C7 is both a receiver and a source to C9; these are nested
arrangements in one connected catchment. The earlier spring comparison and this
full-archive follow-up share observations and are not independent replications.
We preserve actual sample times, require a three-sample span of at most 12 hours
and a common fixed-UTC+1 date, and never interpolate DOC or missing discharge.

Daily discharges provide the two-branch mixture
`(Q_A C_A + Q_B C_B)/(Q_A+Q_B)`. Each concentration series is projected onto the
same intercept, annual sine/cosine and linear calendar-time design. The primary
ratio is adjusted outlet SD divided by adjusted mixture SD. It describes the
variation around those fitted patterns, not hourly event peaks or DOC removal.
Five thousand bootstrap draws resample complete years jointly across sites and
refit that projection. Intervals describe temporal repetition at these sites,
not replication across independent morphological classes.

## Four measured arrangements on identical dates

| Receiver | Source paths, km | Common corridor, km | Outlet/mixture adjusted SD | 95% year interval |
|---|---:|---:|---:|---:|
| C12 | 1.102 / 2.856 | 0.096 | 0.860 | 0.843–0.886 |
| C16 | 8.400 / 8.169 | 3.132 | 0.959 | 0.856–1.061 |
| C7 | 0.015 / 1.090 | 0.014 | 1.471 | 1.154–2.217 |
| C9 | 2.085 / 2.035 | 1.708 | 0.881 | 0.794–1.071 |

C12 has lower adjusted outlet variation on this shared calendar. The C16 and C9
intervals cross one. C7 has greater adjusted outlet variation than its partial
upstream mixture; its paired ratio difference from each other configuration also
remains positive in the year-bootstrap comparison. These contrasts compare four
specific monitored arrangements, rather than identify a geometry coefficient.

The common share is only 4.8% at C12 and 82.9% at C9, yet their adjusted ratios
are close: C9-minus-C12 is +0.021 [−0.055, +0.222]. Thus the present observations
do not give a simple ordering by common-path fraction. They do not establish
that common-path length has no effect.

### Seasonal smoothing and remaining fluctuations differ

On the same 47 dates, C7's **raw** outlet/mixture SD ratio is 0.783, while its
adjusted ratio is **1.471**. The seasonal/time design accounts for 83.2% of the
mixture variance and 40.5% of outlet variance. A smaller overall SD therefore does
not imply smaller variation around the calendar pattern.

On all 306 C7 dates, the adjusted ratio is 0.753 [0.659, 0.871]. Changing to the
joint calendar changes the represented years and dates; it must not be described
as a measured change to channel geometry. Both populations are retained.
Omitting each joint-calendar year in turn leaves the C7 ratio at 1.224–1.688;
the joint result is not supplied by one anomalous year. C12 likewise remains
below one after every year omission (0.850–0.869).

## Higher flow changes the response within a fixed arrangement

Receiver daily positive-flow terciles use all available flow days within each
comparison span, including dates without chemistry. The seasonal projection
remains the full-population projection when examining a flow state.

| Joint-calendar receiver | Low-flow ratio | High-flow ratio | High minus low | 95% year interval |
|---|---:|---:|---:|---:|
| C12 | 0.875 | 0.824 | −0.051 | −0.191 to +0.166 |
| C16 | 0.632 | 1.030 | +0.399 | +0.155 to +0.759 |
| C7 | 0.800 | 1.497 | +0.697 | +0.242 to +1.100 |
| C9 | 0.812 | 0.813 | +0.001 | −0.100 to +0.611 |

The larger high-flow ratio recurs at C16 and C7 in the all-calendar sensitivity:
contrasts +0.274 [0.098, 0.519] and +0.280 [0.147, 0.407], respectively. Water
state is therefore an empirically useful condition on the response at those
configurations. A shorter transport/storage exposure at higher discharge is a
testable explanation; these data do not measure residence time to identify it.

Flow states are local terciles, so different sites need not enter the same state
on a shared date. The paired layout contrast is the joint-calendar result above;
these state rows are within-site comparisons.

## What branch coordination contributes

On the common dates, the adjusted branch correlations are 0.911, 0.738, 0.404 and
0.825 at C12, C16, C7 and C9. Fixed-mean-flow weighting yields variance reductions
of 2.0%, 12.8%, 24.9% and 13.0%, relative to the same-date flow-share-weighted
individual variances. Their exact covariance decompositions are preserved.
The mixture identity itself is established mathematics; the observed contrasts
in coordination and weighting supply the case-specific empirical information.

C7 can have a less variable calculated combination of branches and still a more
variable observed outlet residual on the joint calendar. That separates mixing
of measured inputs from what happens between those inputs and the receiver.

## Water coverage and the remaining structural inference

Median measured upstream/receiver flow shares on joint dates are 0.630, 0.229,
0.625 and 0.610. No configuration has enough joint dates with a share of 0.8–1.2
to repeat the full adjusted comparison: counts are 7, 0, 1 and 2. The all-calendar
screen allows C12 (31 dates) and C7 (41), with adjusted ratios 0.857 and 0.476,
but changes their dates. These sensitivities cannot close the unmonitored carbon
and water budget.

Consequently, greater outlet residual variation is an observed signal relation,
not proof of carbon production; lower variation is not proof of carbon removal.
The original three whole-network classes remain intact and are not assigned to
these four partial monitored arrangements. Their direct event-scale class contrast
still needs coordinated branch and outlet sampling in independent catchments.

## Research decision

Retain this as the measured-process extension to the real-geometry timing study.
It adds a concrete result beyond the intuitive statement that “more tributaries
mix”: **the same arrangement can smooth the seasonal pattern and behave
differently for departures from it, while higher flow weakens the apparent
buffering at two configurations.** Keep geometry, incoming coordination and
water state separate in the paper's mechanism diagram and observed results.

Do not launch another geometry-only readout or retune the completed four-procedure
DOC prediction comparison on these observations. The next decisive empirical
design measures both tributaries and the outlet during the same events, with
continuous flow, event DOC and a conservative reference. This would distinguish
arrival overlap, spreading and unmonitored input along a measured common corridor.

Sources: [SITES laboratory chemistry](https://meta.fieldsites.se/collections/368_zFIGZFTzg1Ynt5tNEGPv),
[SITES daily discharge](https://meta.fieldsites.se/collections/2M74g6oPabfeXXoOkaiGbFs2),
and the [SLU-linked Krycklan stream map](https://ttiwarir.github.io/krycklan-map/).
The saved parent retrieval manifest identifies the original object versions.
This study has been made possible by data provided by the Swedish Infrastructure
for Ecosystem Science (SITES).
