# Research decision: river structure redistributes a DOC pulse in time

## Main finding

**Independent tributary paths disperse arrivals; a common-trunk storage
response spreads each arrival through time. These are distinct structural
operations, and their strength depends on the duration of the DOC input.**

On the 59 measured river footprints, identical tributary inputs and fixed
mixture shares produce different outlet responses even when the mean arrival
time is kept identical. This gives the river-form story a specific process
explanation: geometry organizes the arrival distribution, while temporary
storage can broaden that distribution without removing DOC.

This is a controlled routing experiment on real station-to-station geometry,
not an estimate of observed DOC peak attenuation. It uses 22 receivers in 11
connected monitoring systems and introduces no new prediction-model training.
The three original whole-network outline classes remain unchanged. Every pulse
duration, storage fraction and branch condition is reported, without selecting
parameters from measured downstream DOC.

## What is controlled

Both tributaries receive the same Gaussian concentration anomaly, peak one.
Existing drainage-area proxy weights remain fixed, with constant positive flow.
The independent branch lengths and shared trunk come from the measured NHDPlus
station-to-station footprints. Divide lengths by their weighted mean complete
path once. This gives a reference mean arrival of one, in relative time units.

Compare actual branches with equal branch delays at the same weighted mean.
Then replace fractions 0, 0.25, 0.5 and 1 of the common mean-time budget by a
nonnegative, unit-gain exponential storage response. The remaining deterministic
translation decreases by exactly the storage kernel's mean. Thus the total
mean arrival stays one in every case. This is a matched response-shape
comparison; it does not add storage to a calibrated river while also keeping
its measured velocity fixed.

Storage follows the constant-discharge, constant-volume conservative tank
equation `tau * dC_out/dt + C_out = C_in`. Its impulse response is
`exp(-u/tau)/tau` for `u >= 0`. Constant concentration and integrated anomaly
are preserved. A lower concentration peak means redistribution in time, not
chemical DOC loss. Mapped lake/reservoir path fractions identify actual channel
context; they are never converted into residence time or a storage fraction.

The conceptual separation of translation and temporary-storage attenuation
follows the [USACE HEC-HMS Clark routing description](https://www.hec.usace.army.mil/confluence/hmsdocs/hmstrm/transform/clark-unit-hydrograph-model).
That source describes runoff routing. The DOC-anomaly kernel here is separately
derived for conservative concentration under the stated constant-flow scenario.

## 1. Branch differences and storage both reshape the pulse

For the middle input duration, SD 0.15, summaries first average connections
within receiver and then give the 22 receivers equal weight:

| Common time allocated to storage | Peak reduction from branch differences, relative to equal branches at the same storage | Additional peak reduction from storage, relative to actual-branch translation | Change in central 80% duration from storage |
|---|---:|---:|---:|
| 0% | 12.37% | 0% | 0% |
| 25% | 11.44% | 15.10% | +19.77% |
| 50% | 10.40% | 29.26% | +58.36% |
| 100% | 8.95% | 46.09% | +151.16% |

At the 50% setting, combined peak reduction relative to equal-branch pure
translation is **38.78%**. It is not the sum of 12.37% and 29.26%: those
percentages have different denominators, and branch separation and storage
interact. Broadening an already split pulse overlaps partly with the operation
that split it in the first place.

The centroid remains one although the peak time changes. A peak is the instant
of highest concentration; a centroid is the mean timing of the entire anomaly.
They are distinct measures. The central 80% duration is t90 minus t10 of the
integrated anomaly, not the time above an arbitrary concentration threshold.
In this fixed-mean comparison, storage replaces some deterministic translation
with a residence distribution. Its largest peak can occur earlier while its
late tail grows; this follows the changed response shape and is different from
adding storage while leaving every advective delay unchanged.

## 2. Input duration changes the same network's response

At 50% common-time storage, compare all three fixed input durations:

| Input pulse SD | Additional peak reduction from storage | Change in central 80% duration | Combined peak reduction from branch separation and storage |
|---|---:|---:|---:|
| 0.075, short | 46.59% | +132.74% | 55.10% |
| 0.15, middle | 29.26% | +58.36% | 38.78% |
| 0.30, long | 15.13% | +22.28% | 23.56% |

The additional storage peak reduction is short >= middle >= long in every one
of the 59 footprints for this fixed setting. A sharp input is more easily
smoothed by the same residence distribution. A broad, persistent input changes
more slowly than that response, so its peak is less altered.

This is why a class label cannot imply one fixed DOC-buffering percentage.
The relevant quantities are branch arrival SD divided by input duration and
storage mean time divided by input duration, together with branch contribution
balance. A network can strongly smooth event peaks while transmitting slower
changes much more closely.

## 3. A useful exact decomposition

With normalized independent branch delays b_a and b_b, shares w and 1-w,
input SD sigma and common storage mean tau:

    output timing variance = sigma^2 + w(1-w)(b_a-b_b)^2 + tau^2

The three terms represent input duration, branch-arrival dispersion and
common-trunk storage. Unlike percentage peak reductions, these variance
contributions add exactly in the experiment. This supplies a clean process
representation for a future directed temporal graph operator.

In 44 of 59 geometric footprints, storage could exceed the branch-arrival
variance within the full common-time budget. The median fraction required for
equal storage and branch variance is 0.303. These are conditional geometric
calculations, not measured storage strengths. They describe where a shared
response could matter if hydraulic information established it.

## 4. Connect the result to real river forms

The user's elongated, mainstem-dominated and broad tributary-rich networks are
whole upstream structures. The present controlled experiment resolves the
pieces actually covered by a tributary pair and downstream receiver. It
clarifies the operations inside a river form rather than redefining its outline.

The monitored subset contains 22 elongated-class connections at six receivers
in one system, one mainstem/sparse connection at one receiver, and 36
broad-class connections at 15 receivers in 11 systems. `outline_context.csv`
retains their descriptive scenario summaries. This uneven coverage cannot
establish a replicated storage ranking among the three whole-network classes.

The previous whole-network routing study already showed that real path
distributions reshape identical inputs, including an elongated–broad comparison.
This follow-up supplies the missing conservative storage operation. Together,
they motivate describing each real river by a **response fingerprint**:
arrival centroid, path dispersion, contribution balance and storage broadening.
The fingerprint explains why two similarly shaped outlines can transmit DOC
differently and why a visually broad network is not automatically the strongest
buffer.

The five geometry-selected examples in the figure are display examples, not
five new shape classes. The fifth has the largest measured common-trunk
waterbody-path fraction, 89.17%, but only a 13.07% common share of its weighted
complete path. Under the identical 50% common-time allocation its additional
peak reduction is only about 6.2%. The illustration shows what the fixed
allocation means; it does not infer that this actual waterbody has weak storage.
Length fraction alone supplies neither water volume nor hydraulic residence.

## 5. What the observations can test now

The sampling-date follow-up retained substantial asynchronous tributary
variability after close-date alignment. That supports studying how the measured
branches combine varying DOC signals. It does not identify a storage kernel:
the median station sampling interval is 28 days, and none of the five mapped
common-storage connections has a common month with at least three distinct
sampling days at all three stations.

Consequently, keep the controlled peak/width result alongside the date-aligned
source-variability result. Do not translate the relative time axis into days
or fit event residence parameters from a monthly concentration sequence.
Mapped storage context and existing measured DOC remain visible, with no
landscape-source contrast introduced into this experiment.

## Next research step

**Return this mechanism to the complete real-network shape comparison.**

1. Extend the existing whole-network conservative routing experiment with a
   separate storage response on actual mapped waterbody segments. Keep input
   concentration identical everywhere, preserve mean source-to-outlet arrival
   within each comparison, and retain the original three outline classes.
2. Compute response fingerprints across pulse durations, then compare both
   between-class and within-class variation. Separate the contributions of
   path dispersion, branch balance and storage exposure. Check reach/source
   discretization and contiguous waterbody grouping so a finely split lake
   is not treated as many independent reservoirs.
3. Map high-resolution observation opportunities onto contrasting fingerprints:
   a path-dispersed network and a storage-dominated network with similar overall
   mean arrival. Joint upstream/outlet concentration and flow observations would
   distinguish a split arrival from a storage-broadened tail.
4. In the DOC model, represent the river contribution with directed mixtures of
   nonnegative delay/storage kernels, retaining the local environmental and
   temporal prediction as background. Estimate any physical time scale with
   suitable source observations; evaluate added river information separately
   from source-station ecological similarity.

The scientific centre stays river structure: **which paths assemble a DOC
signal, how their arrivals combine, and which internal channel environments
spread that signal in time.**

## Deliverables and numerical checks

All 1,416 scenarios (59 footprints x two branch conditions x four storage
fractions x three durations), 708 matched contrasts, receiver summaries and
geometry-selected traces are saved. No scenario is discarded.

The analytical and integrated centroids agree within 5.2e-13; integrated anomaly
fractions differ from one by at most 1.7e-14; numerical and analytical SDs differ
by at most 8.3e-12. A preliminary time step of 0.0025 gave a maximum narrow-pulse
peak difference of 0.000119 on refinement, so the final step is 0.00125 and the
full cohort is repeated at 0.000625. Maximum peak difference is now 0.0000323
of the unit input peak; central-duration difference is at most 0.0000103.

English/Chinese figures show actual cropped NHDPlus geometry, mapped waterbody
segments, matched responses and the complete parameter comparisons. Full replay
and project checks are recorded in `validation.md`.
