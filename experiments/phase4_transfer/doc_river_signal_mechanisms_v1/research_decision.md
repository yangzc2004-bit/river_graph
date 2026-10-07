# Research decision: integration, arrival opportunity and channel budgets

## Main result

**Tributary integration has a substantial capacity to suppress asynchronous DOC
fluctuations. In the existing monthly representation, separating the two path
delays changes the upstream concentration input very little. A field estimate
of channel DOC loss remains unsupported by the available water budgets.**

The structural question is becoming more concrete: branch number alone does
not determine buffering. The relative contribution of branches and whether
their fluctuations coincide determine the mixing operation; path differences
matter when the tributaries change differently through time. The whole-network
outline is an organizing description of these operations, not a sufficient
prediction of their net DOC effect.

## Observations and comparison population

Reuse all **59 independent tributary-pair connections, 22 receiving stations,
11 connected monitoring systems and 15 HUC4 regions** in the completed observed
transport study. Their **3,026 connection-months represent 1,092 unique receiving
station-months**. Each source has measured DOC in the current and immediately
preceding month; the receiving DOC is measured on the same current monthly grid.
All DOC cells belong to the existing permitted source-role union 142/143/144.

No neural model or calibrator was trained or selected. This is an exploratory
mechanism diagnostic after seeing earlier results, not independent confirmation
of them. Pair anomalies are connection-specific because different pairs have
different common dates. Dates are equal within a pair, pairs equal within a
receiver, and receivers equal in summaries. Intervals use 5,000 whole-system
bootstrap draws; HUC4 sensitivity is separately saved.

## 1. What the mixing operation does

Project both sources, their fixed area-weighted mixture and outlet onto exactly
the same calendar sine/cosine and year design. Native concentrations preserve
linear mixing, so the source/mixture variance decomposition is exact.
The reference is the area-share-weighted average source anomaly variance,
not the average concentration and not a measured load.

| Mixing calculation / scenario | Variance reduction | Whole-system 95% interval |
|---|---:|---|
| Observed signals, original area-share weights | 21.03% | [14.67%, 27.39%] |
| Contribution from source amplitude imbalance | 4.78 percentage points | [2.72, 6.77] |
| Contribution from asynchronous changes | 16.25 percentage points | [11.50, 21.67] |
| Equalized amplitudes, original branch weights | 16.30% | [12.16%, 20.91%] |
| Equalized amplitudes and balanced 50:50 weights | 35.81% | [29.22%, 41.76%] |

Asynchronous changes account for **77.3% of the mean calculated reduction**
(16.25/21.03); this is a ratio of the reported mean contributions, not a mean
of per-connection percentages. The mean source correlation is 0.284
[0.165, 0.416]. Thus the branches often do not fluctuate together.
Equalizing amplitudes retains a meaningful modeled buffer. Equalizing their
contributions increases that potential; perfectly synchronized equal-amplitude
sources have zero mixing buffer under any weight.

This does not establish a new universal mixing law: the nonnegative variance
reduction follows from the fixed mixing identity. The empirical information is
the observed correlation, balance and contribution magnitudes. Area is a proxy
for branch flow share. The 21% describes a two-source mixture calculation,
**not 21% less DOC concentration, whole-river removal or net carbon export**.

The calculated reduction is stable to leaving out one entire monitoring system:
19.65–22.76%. The equal-amplitude scenario ranges from 15.32–17.68%.

## 2. Why branch-specific monthly paths gave little extra prediction

The exact branch-minus-mean-path concentration difference is

`w(1-w) * (pathA-pathB)/max_path * [(Aprev-Anow)-(Bprev-Bnow)]`.

Three ingredients must coincide: both branches contribute, their paths differ,
and their DOC changes differ. Equal paths or parallel changes give exactly zero
additional signal. This connects a genuine structural property to an observed
temporal condition without confusing a source mean difference with routing.

Using the original maximum path scale (861.43 km) and fixed interpolation
fraction 1, the mean absolute input difference is **0.01475 mg/L**
[0.00707, 0.02011]. Its mean stabilized relative magnitude is **0.278%**
[0.135%, 0.385%], with denominator 1 mg/L plus the same-month mixture.
The hierarchically weighted fraction of connection-months exceeding a 1%
difference is 6.33% [1.65%, 9.25%]; exceeding 5% is 0.229%
[0, 0.484%]. These are not pooled counts of independent outlet observations.

Apply the *same saved mean-delay model* to each input, without refitting or
changing its selected setting: predictions differ by **0.00858 mg/L** on average
[0.00376, 0.01161]. The mean-path predictions reproduce the original saved
predictions. This isolates sensitivity to input geometry; it does not introduce
another fitted-model performance claim.

The input difference remains 0.01264–0.01615 mg/L when omitting one system.
It explains why the previous branch-arrival operator supplied so little extra
information over a mean path **in this representation**. It does not show that
actual travel-time differences are small: the old interpolation scale is not a
measured travel time, and monthly observations cannot resolve submonthly events.

## 3. What the measured outlet adds

On these same common dates, the geometric mean outlet/mixture anomaly SD ratio
is **0.768 [0.599, 0.983]** in native concentration and **0.840
[0.730, 0.970]** in log1p concentration. Native scale is the primary identity;
log scale was added after inspecting native summaries as a scale diagnostic.
Whole-system omission retains ratios below one on both scales.

The outlet/source arithmetic variance ratio is **1.89 [0.49, 5.58]**.
Connection 03216600–03267900–03303280 has a ratio of 28.43, illustrating that
the geometric mean and arithmetic variance mean answer different questions.
All concentrations and connections remain included; measured values span
0.3–40 mg/L in the sources and 0.9–22.9 mg/L at the outlets.

Smaller typical outlet variability is compatible with integration and routing,
but cannot be attributed entirely to the two gauged branches or to removal.
Unmonitored tributaries, changing water shares and monthly sampling remain part
of the outlet comparison. The previous larger 38-receiver process cohort and
this 22-receiver current/previous-month cohort are different; this is not an
independent replication or a replacement of the earlier result.

## 4. Does the water budget support a channel-loss conclusion?

| Availability condition | Connection-months | Unique outlet-months | Receivers | Systems |
|---|---:|---:|---:|---:|
| Common DOC input population | 3,026 | 1,092 | 22 | 11 |
| Measured positive flow at all three stations | 906 | 563 | 18 | 10 |
| Source area covers at least 80% of receiver drainage | 147 | 147 | 5 | 4 |
| Area condition plus source/receiver flow ratio 0.8–1.2 | 101 | 101 | 5 | 4 |

Among all three-flow cases, the receiver-equal source/receiver flow ratio is
only 0.481 [0.374, 0.599]. Much of the receiver water is unrepresented by the
two sources. In the screened five receivers, the ratio is 0.937
[0.921, 0.976]. Three of these receivers are broad and two elongated; no sparse
receiver passes. Cases remain inspectable at 03290500, 05355250, 06342500,
06889000 and 06892350.

The outlet-minus-flow-weighted-source apparent concentration departure is
+0.210 mg/L [-0.450, +0.535] in the three-flow cases and **-0.162 mg/L
[-0.987, +0.291]** after the area/water screen. Both intervals include zero.
The screened individual departures range from -1.278 to +0.521 mg/L.
Monthly averaged DOC and flow do not form contemporaneous DOC loads; no
retention coefficient or physical degradation rate is estimated.

## 5. Relation to the user's real river forms

The observed subset contains six elongated receivers in **one** monitoring
system, one mainstem/sparse receiver in **one** system, and 15 broad receivers
in 11 systems. Descriptive mixture reductions are 16.9%, 63.8% and 19.8%,
respectively. The sparse value belongs to one receiver and is not a class effect.
The broad and elongated numbers do not establish a replicated class contrast.
Single-system class intervals are unavailable even in the HUC4 sensitivity;
counting three HUC4 inside one shared system does not create independent systems.

The more useful structural description is now **branch balance × asynchronous
variation × path contrast**. A wide, branch-rich outline can contain a dominant
trunk; a narrow outline can contain a balanced junction. That is why labeling
every broad river as a strong buffer would miss the operation being tested.
The complete real-geometry routing study still supplies the whole-network,
identical-input comparison; the observed pairs supply local process evidence.

## Next research step

1. **Expand the junction representation.** Build balanced-versus-dominated and
   similar-versus-dispersed-path categories from real NHD geometry, independent
   of DOC. Nest these internal structural categories inside the three fixed
   outline classes rather than recluster using the present outcomes.
2. **Map monitoring coverage onto those structures.** Identify the additional
   branches entering between gauges and outlet. Prioritize near-complete,
   independently monitored junctions and additional elongated/sparse systems.
3. **Audit original DOC sampling dates at those junctions.** Establish which
   examples have same-date or event-scale upstream/downstream observations
   before estimating peak timing, hysteresis or physical retention. Daily flow
   alone cannot replace missing high-frequency DOC.
4. **Retain the controlled input experiment as the mechanism bridge.** Vary
   branch balance, path dispersion and arrival synchrony on real geometry with
   input amplitudes held fixed. Tie its predictions to the observed signal
   conditions above; do not retune the monthly arrival proxy on scored outlets.

The emerging narrative is: **river shape organizes how many paths contribute,
how evenly they contribute and when their signals meet; these operations reshape
DOC variability, while concentration change requires an additional budget.**

Replay commands and verification are in README.md and validation.md. No earlier
model, classification, protocol or prediction has been overwritten.
