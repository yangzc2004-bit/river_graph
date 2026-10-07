# Research decision: tributary dispersion and shared-trunk translation

Date: 2026-10-07.

## Scientific advance

River structure now has a concrete interpretation at the scale actually
observed: **independent branches separate arrival times, while their common
downstream trunk shifts both signals together**. This distinction explains why
river length alone is not an adequate descriptor of DOC buffering.

The previous three complete-network outlines remain unchanged. This study does
not rank terrestrial source regions or create another morphology classification.
It resolves the structural operations within the measured river corridor.

## 1. Actual monitored river footprints

All previous 59 eligible tributary pairs, 22 receivers and 11 connected
monitoring systems are retained, with the same 3,026 connection-month records
and 1,092 unique receiver-month truths. The primary downstream NHDPlus route is
partitioned at the first common reach into branch A, branch B and a single
shared trunk. Distances include the actual linear-reference positions of all
three stations. No station measure is imputed in this cohort.

The inventory contains 10,175 pair-specific reach records and 3,509 distinct
COMIDs across overlapping corridors. All source-to-receiver and confluence
distances reproduce the original inventory. Shared reaches occur once per pair.

Geometry-only display cuts are independent-branch CV 0.3008 and common-trunk
share 0.5057. The four quadrants contain 12, 17, 17 and 13 connections. Their
representatives are chosen near the geometric median, without DOC outcomes.
These are four examples, not four natural classes. Mapped channels are cropped
at station measures after inferring downstream coordinate orientation. All
386 distinct mapped reaches have cached real geometry and zero endpoint gaps.

| Example | Source A / source B / receiver | Independent A / B (km) | Common trunk (km) | Branch CV | Common share |
|---|---|---:|---:|---:|---:|
| Lower dispersion, lower common share | 05520500 / 05540500 / 05543500 | 76.8 / 23.8 | 43.0 | 0.248 | 37.9% |
| Higher dispersion, lower common share | 06893890 / 06893970 / 06894000 | 12.5 / 2.7 | 3.5 | 0.370 | 24.6% |
| Lower dispersion, higher common share | 06889000 / 06890900 / 06892350 | 28.5 / 2.5 | 54.3 | 0.132 | 66.0% |
| Higher dispersion, higher common share | 05449500 / 05451210 / 05453100 | 120.3 / 8.6 | 162.3 | 0.650 | 66.5% |

Balance is `4w(1-w)`, using the same positive source drainage-area shares as
before. These shares are not measured instantaneous discharges. A small
tributary can travel a very different distance while contributing little to
the weighted dispersion, as in the third example.

Let A and B denote independent lengths and C the common length. The exact
identity is:

`total-path CV = (1 - common share) × independent-branch CV`.

A long shared trunk reduces the relative CV by increasing the mean path. It
does not itself increase the difference between the two arrival times. This
distinguishes common residence from tributary separation.

## 2. Controlled structural comparison

Both sources receive an identical Gaussian concentration pulse, SD 0.15.
Weights remain fixed. Actual total paths are normalized once to weighted mean
delay one. Then we either equalize the independent branches at their original
weighted mean, or remove the common trunk without renormalizing time.

Across all 59 connections:

| Operation | Mean peak / identical input peak | Median peak | Mean output SD | Mean relative delay |
|---|---:|---:|---:|---:|
| Actual branches and common trunk | 0.8406 | 0.8777 | 0.3108 | 1.0000 |
| Equal independent branches, same common trunk | 1.0000 | 1.0000 | 0.1500 | 1.0000 |
| Actual branches, common trunk removed | 0.8406 | 0.8777 | 0.3108 | 0.5120 |

Unequal branches lower the peak in every connection; mean peak reduction is
**15.94%**, with individual reductions from 0.013% to 49.26%. Removing the
shared trunk preserves the exact pulse shape, peak and width, and shifts its
centroid earlier by the original common share. Integrated anomaly mass stays
one and steady concentration stays five in all 177 scenarios.

Thus relative arrival and shared translation are distinct operations. This is
a controlled, constant-velocity routing experiment, not a DOC removal model or
an estimate of travel days. Storage, changing flow and processing can add
further effects to an actual shared trunk; those effects are not fitted here.
Five of the 59 common trunks include mapped storage reaches.

## 3. Observed DOC on the matched footprint

All six exploratory continuous associations are reported, adjusting for the
other two structural terms, log mean path length and source drainage coverage.
Dates are summarized within connection and connections within receiver.
Intervals use 5,000 whole-monitoring-system bootstrap draws.

Coefficients below are changes in log(outlet/mixture SD ratio) per one
receiver-population SD of the structural descriptor:

| Structural term | Native DOC coefficient (95% interval) | Log1p DOC coefficient (95% interval) |
|---|---:|---:|
| Branch balance | −0.173 (−0.902, 0.414) | −0.026 (−0.524, 0.276) |
| Independent branch CV | 0.128 (−0.944, 1.056) | 0.011 (−0.390, 0.607) |
| Common-trunk share | 0.024 (−0.431, 0.974) | −0.041 (−0.343, 0.575) |

All intervals span zero. The native branch-dispersion sign is positive, unlike
the isolated pulse-damping expectation. Matching the footprint therefore does
not produce a reliable monotonic field relationship in the available monthly
cohort. The adjusted design has condition number 2.63 and maximum leverage
0.629 at 05355250. Of 5,000 bootstrap draws, 4,985 retain full design rank;
15 excluded draws are recorded. Leave-system signs vary for several terms.

Whole-network balance correlates 0.278 with matched branch balance. Whole-network
path CV correlates 0.305 with total matched-path CV and 0.315 with independent
branch CV (receiver-level Spearman correlations). The measurements describe
different structural objects. Median receiver-level monitored drainage coverage
is only 33.8%; unmonitored lateral inputs remain part of the observed outlet.

## 4. Fixed model: where does timing information enter?

We reuse all saved held-system mean-delay calibrators without fitting anything.
Their training systems exclude the held system, and their lag fraction remains
one. Four source proxies are applied to the same coefficients and background
features. The reference is the complete saved mean-delay model, reproduced
within 1e-10 mg/L, not a weaker replacement baseline.

The existing current/previous interpolation separates into:

1. Same-month area-weighted source DOC.
2. A common-trunk shift of that mixture.
3. An equal independent-branch shift.
4. A differential arrival term proportional to branch length difference and
   to the difference between the two source concentration changes.

Receiver-equal mean absolute input terms are 0.1152 mg/L for shared delay,
0.0861 for equal-branch delay and 0.01475 for differential arrival. These are
separate input sensitivities, not additive percentages of explained variance.

| Source input under identical saved coefficients | MAE (mg/L) | Change relative to complete mean-delay reference |
|---|---:|---:|
| Same-month mixture | 1.65805 | 1.024% worse |
| Shared-trunk shift only | 1.64609 | 0.296% worse |
| Complete mean delay, equal branches | 1.64123 | Reference |
| Separate actual branch delays | 1.64105 | 0.011% better |

The paired same-month penalty is 0.01681 mg/L (95% interval 0.00245–0.02559),
and the shared-only penalty is 0.00485 (0.00065–0.00686). The branch-specific
gain is 0.000182 (−0.000306–0.001149), with relative interval −0.019% to 0.089%.
Average timing contains a small native-scale signal under this fixed model;
the extra differential-arrival term does not deliver an appreciable overall
prediction gain. Log-space intervals are also reported and do not establish
these timing improvements as scale-robust.

Q90 MAE falls from 4.95072 to 4.94652 mg/L after adding differential arrival.
That 0.00419 mg/L paired gain has interval 0.00040–0.01288, but is only **0.085%**.
The tail comparison covers 11 receivers and six systems and is an exploratory
diagnostic alongside all other metrics, not a substantial high-DOC advance.
Repeated predictions of the same receiver-month are not new ecological samples.

This experiment uses known upstream DOC; it is not the project's unmonitored
K0 task. Hidden receiver labels affect scores only. Perturbing those labels or
future source records leaves the corresponding earlier predictions unchanged.

## Research decision and next direction

**River organization regulates the timing and overlap of DOC signals. Independent
branch dispersion supplies peak damping; a shared trunk supplies common delay
under simple routing. Real monthly prediction currently captures mainly the
common timing scale, with little additional value from resolving two branch delays.**

The mechanism study supports a path-aware temporal filter, but not a claim that
this study has improved the current neural DOC model. It also does not establish
a universal ordering of the three river-outline classes.

The next work should test the time scale and actual transmission kernel:

1. Audit dated DOC samples and coincident discharge along these same real
   corridors to locate event sequences capable of resolving tributary arrival.
   Check within-month timing before adding a larger learned lag module.
2. Separate common-trunk translation from widening associated with mapped
   storage or varying flow, keeping the independent branches explicit. Compare
   timing and width with the same source pulse and area/discharge information.
3. Use the supported operators to construct a directed temporal filter with
   nonnegative weights and unit steady-signal gain. Compare equal-delay,
   path-dependent and no-message versions, with development confined to source
   roles and a separate station/region evaluation for performance claims.

This keeps the research centered on river structure and signal organization.
The next priority is resolving transmission at the right time scale, rather
than another outline clustering or an expansion of neural-model complexity.

## Reproducibility and study status

Analysis, figures and verification have dedicated scripts and independent
versioned products. Both English and Chinese maps/figures are supplied.
The new study is exploratory after earlier results were seen, within ST357.
It neither retrains the saved calibrators nor constitutes external validation.
Historical morphology, prediction and endpoint products remain intact.
