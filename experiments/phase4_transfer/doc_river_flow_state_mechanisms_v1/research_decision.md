# How water state changes the DOC response of a fixed river arrangement

Date: 2026-10-08.

## Main result

**The river-arrangement story needs both branch coordination and the amount of
water each branch contributes. These can push DOC mixing in opposite directions
as discharge changes. A larger outlet/input fluctuation ratio does not, by itself,
identify a weaker channel buffer.**

The strongest new observational pattern is the increased mixing contribution of
a more balanced branch water supply at C7. That contribution is positive on both
the common calendar and the complete observation calendar. At C16, the complete
calendar instead supports reduced fixed-weight mixing potential associated with
more coordinated branch DOC fluctuations. The four real configurations therefore
do not share one uniform high-flow mechanism.

## Same observations, more specific question

We retain the preceding four mapped configurations, 561 complete eligible
configuration-dates and the 47-date intersection in 2014–2017. The latter supplies
47 comparisons at each configuration, not 188 independent events. C7 is also an
upstream source of C9. These are nested locations within one Krycklan catchment.

The former laboratory-DOC matching, fixed-UTC+1 date rule, discharge records,
positive-flow terciles and annual-harmonic plus linear-time projection are
unchanged. State SDs use the same population-wide projection, rather than
separately fitted seasonal patterns. Five thousand complete-year bootstrap draws
refit the projection and preserve simultaneous configurations. Intervals quantify
temporal repeatability at these locations. This follow-up explains already-seen
observations; it is not independent confirmation of the preceding contrasts.

## 1. An increased ratio can result from a smaller incoming fluctuation

Write `R = SD(outlet)/SD(partial branch mixture)`. Its high/low change has the
exact decomposition

```text
log(R_high/R_low)
 = log(SD_outlet_high/SD_outlet_low)
 − log(SD_mixture_high/SD_mixture_low).
```

Joint-calendar point estimates, in mg C/L:

| Configuration | Mixture SD, low → high | Outlet SD, low → high | Log-ratio change [95% year interval] |
|---|---:|---:|---:|
| C12 | 4.010 → 4.955 | 3.508 → 4.083 | −0.060 [−0.217, +0.190] |
| C16 | 4.096 → 3.099 | 2.587 → 3.193 | +0.489 [+0.192, +1.282] |
| C7 | 5.141 → 3.149 | 4.113 → 4.713 | +0.626 [+0.174, +1.127] |
| C9 | 2.706 → 4.039 | 2.197 → 3.282 | +0.001 [−0.123, +0.536] |

At C7, the mixture SD decreases by **38.8%** while outlet SD increases by
**14.6%**. The corresponding log contributions are +0.490 and +0.136. In these
point estimates, most of the rising ratio arises from the changing denominator.
The individual percentage-change intervals are wide and cross zero
(mixture −49.5% to +65.6%; outlet −30.1% to +182.4%), even though the paired
ratio-change interval remains positive. We therefore do not assign a confidently
estimated causal fraction to either component.

The complete C7 calendar has the same point-estimate pattern: mixture SD −25.3%,
outlet SD +8.1%, and log-ratio change +0.370 [+0.183, +0.535]. At C16, the complete
calendar differs from the intersection: both SDs increase, with outlet SD +62.5%
[+20.4%, +346.9%] and mixture SD +13.9% [−9.5%, +146.7%]. Its log-ratio increase
is +0.355 [+0.114, +0.810]. Both calendars are retained rather than selecting the
one giving a simpler narrative.

**Refinement to the preceding interpretation:** C16 and C7 have a repeatable
higher outlet/mixture ratio at high flow. Calling that a directly measured loss of
channel smoothing, or attributing it to reduced residence time, goes beyond what
this ratio establishes.

## 2. Branch synchrony and water balance can oppose each other

For the two adjusted branch series A and B, let `w` be the state's mean
`Q_A/(Q_A+Q_B)`. The constructed fixed-weight mixture variance is

```text
V_mix = w² V_A + (1−w)² V_B + 2 w(1−w) Cov(A,B).
```

We describe its reduction relative to `w V_A+(1−w)V_B` as fixed-weight mixing
potential, in percent. This is a variance summary, not a carbon-removal fraction.
All eight low/high substitutions give the all-order Shapley decomposition into
branch correlation, relative branch amplitudes, and mean water fraction.

At C7 on the joint calendar:

- Mean branch-A water fraction changes from **0.109 to 0.334**. A previously
  minor branch supplies substantially more of the measured two-branch water.
- Adjusted branch correlation changes from **0.092 to 0.524** in the point
  estimates: greater coordination opposes cancellation when the other summaries
  are held fixed.
- The water-balance contribution to mixing potential is **+17.31 percentage
  points [+11.67, +23.34]**. The correlation contribution is −14.10 pp
  [−43.09, +24.59], and the amplitude contribution is +0.85 pp
  [−17.64, +4.86]. The observed total is +4.06 pp [−45.92, +38.60].

Thus the positive balance contribution is repeatable, while the total change and
the correlation component on four joint years are uncertain. The complete C7
calendar supports the same positive water-balance contribution, **+15.31 pp
[+11.36, +18.74]**, and a positive total change **+11.17 pp [+1.95, +20.02]**.
Increased synchrony does not automatically make the mixture less smooth because
the stronger second-branch contribution can offset it.

C16 supplies the contrasting full-calendar case. Fixed-weight mixing potential
falls from **16.92% to 3.28%**, a change of −13.64 pp [−47.87, −5.14]. Its
correlation contribution is −13.08 pp [−26.62, −5.62], despite a +2.41 pp
[+0.36, +3.88] balance contribution. On the common calendar the analogous total
is −15.41 pp, with an interval crossing zero [−48.48, +0.70].

These substitutions explain the analytic summary, not hypothetical measured
floods. The previous population-wide fixed weight is retained as a sensitivity.
Changing from population-wide to state-mean weights can change the apparent
direction at C7; this directly motivates reporting the changing water fractions.
The actual dynamic mixture also includes daily-weight and seasonal-projection
interactions, so its SD is reported separately from the constructed mixture.

## 3. The outlet departure is not an isolated channel-processing signal

For `D=outlet−partial mixture`, the exact variance identity is

```text
V_outlet = V_mixture + V_D + 2 Cov(mixture,D).
```

At joint-calendar C7, the normalized covariance term changes from −0.641 at low
flow to +0.333 at high flow, while the normalized departure-variance term changes
from 0.282 to 0.908. The sum reproduces the squared outlet/mixture ratios. The
departure combines unmonitored input, noncontemporaneous daily flow information
and any processing; these terms do not isolate a physical channel coefficient.

The measured branches supply median fractions of outlet water of 0.630, 0.229,
0.625 and 0.610 at C12, C16, C7 and C9. For a usable daily water fraction
`0<f≤0.95`, a no-net-processing water-mixture equation requires

```text
C_unmonitored = (C_outlet − f C_partial_mixture)/(1−f).
```

| Configuration | Usable joint dates | Nonnegative implied solutions | Median required unmonitored DOC, mg C/L |
|---|---:|---:|---:|
| C12 | 44 | 41 | 14.58 |
| C16 | 47 | 47 | 9.72 |
| C7 | 47 | 45 | 17.97 |
| C9 | 46 | 46 | 12.47 |

Many observed outlet concentrations therefore admit a nonnegative unmonitored-
water explanation without adding net processing. Those concentrations were not
measured, and nonnegativity alone is not an ecological plausibility test or a
proof that processing is absent. The five negative joint solutions and the four
excluded budgets remain visible. Daily discharge is not a complete instantaneous
carbon load. This diagnostic identifies what new measurements must resolve.

## What this adds to the morphology question

The same mapped arrangement can recruit different water contributions as flow
changes. Geometry describes where separate paths meet and how much corridor they
share; coordination describes whether their DOC fluctuations coincide; water
balance describes whether both paths materially participate in the mixture.
**These are three complementary parts of a flow-dependent river response.**

The C7 common corridor is only about 14 m in the available stream map, whereas
C16 has about 3.13 km. Both display a higher outlet/mixture ratio at high flow,
but their incoming mixing summaries differ. This is useful evidence against
assigning the same ratio-change mechanism solely from corridor length. It does
not estimate a class-wide effect of elongated versus broad/branched networks.
The existing whole-network shape classification remains unchanged.

This keeps the research centered on river arrangement: a branch must carry
enough water to contribute to structural averaging, and correlated branch
signals can counteract that averaging. No new land-source classification is
introduced. In a future prediction experiment, geometry, branch water support
and synchrony should be distinguishable inputs rather than one shape label.
The existing best DOC predictor is retained; this observed-process result is not
claimed as a new prediction-performance improvement.

## Research decision

The current observational explanation is complete. Integrate the three-part
arrangement–coordination–water-balance representation and the revised interpretation
of high-flow ratios into the paper's mechanism discussion. Use the paired-calendar
observations as the measured evidence and the earlier geometry experiments as
controlled demonstrations of possible timing/spreading, with their different
evidence roles stated plainly.

The next decisive process measurement is coordinated DOC and instantaneous flow
on both branches and the common corridor, including unmonitored lateral water,
through the same events. Replicate that design in independently elongated and
broad/branched networks. It can test whether mapped convergence geometry changes
arrival overlap and spreading after the water and carbon budgets are observed.
Repeated analysis of these same four configurations cannot supply that independent
shape contrast or an unmeasured transit time.

## Reproducibility

All 24 flow-state rows, eight population-specific contrasts and 749 ledger rows
replay from the saved observations. The preceding ratio point estimates are
unchanged. Bootstrap valid-draw counts accompany every interval; the complete
calendar C9 high/low comparison has 4,656 valid draws, while all joint comparisons
have 5,000. Seven mathematical/interpretation tests pass. The full suite reports
1,351 passed and two skipped, Ruff passes, and the historical artifact audit exits
zero with its earlier provenance limitations preserved. English and Chinese
figures were rendered, viewed and corrected for legend overlap and font coverage.

Sources: [SITES laboratory DOC](https://meta.fieldsites.se/collections/368_zFIGZFTzg1Ynt5tNEGPv),
[SITES daily discharge](https://meta.fieldsites.se/collections/2M74g6oPabfeXXoOkaiGbFs2),
and the [SLU-linked Krycklan map](https://ttiwarir.github.io/krycklan-map/).
This study has been made possible by data provided by the Swedish Infrastructure
for Ecosystem Science (SITES).
