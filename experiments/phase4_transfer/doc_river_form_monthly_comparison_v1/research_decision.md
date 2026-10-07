# Research decision: river form, DOC level and variability on shared months

## Scientific focus

This step addresses the user's direct question: **do long/narrow and broad,
tributary-rich river forms differ in DOC level, variability and high-value
frequency under comparable size, environment and hydrology?** River form is
the explanatory focus. Environment is comparison context rather than the new
research outcome.

The earlier morphology analysis compared station-level summaries over each
station's own available dates. This extension keeps its actual geometry-based
classes and fixed, non-nested matched rivers, then puts the two members of each
pair on the **same calendar months**. It also estimates background deviations
using receiving-region-excluded environment/hydro fits.

Earlier morphology results were already known. This is an exploratory
source-role extension, not independent confirmation, external validation or
a performance upgrade of the released DOC reconstruction model.

## What was completed

- The 297-station source cohort has 18,688 permitted observed months across
  62 HUC4s and 295 distinct receiving COMIDs. Two duplicated receiving geometries
  stay in the same HUC4 fold; they cannot cross from training to evaluation.
- All 29 preceding matched pairs stay in the availability ledger. Common-month
  and month-of-year criteria retain 26 pairs across the three contrasts.
- The main elongated/broad comparison has **20 pairs, 40 stations, 11 HUC4s and
  669 shared pair-months**. Each member uses the same dates within its pair.
- Joint observed discharge and temperature retain 17 elongated/broad pairs
  in ten HUC4s. A >=24-common-month sensitivity retains 13 pairs in seven HUC4s.
- Six fixed information models and two background exceedance fits use five
  HUC4-held folds, source-only imputation/scaling, and station-equal fitting.
  Receiving DOC labels are excluded from the fit for their receiving region.
- All DOC responses and information comparisons are retained, with 5,000
  HUC4-block bootstrap draws and omitted-region response summaries.

## Main observed result: broad minus elongated

These are mean differences between the two members of each fixed pair, followed
by equal averaging over pairs. A positive difference means the broad member
has the larger value. Intervals are paired HUC4 bootstrap 95%.

| Property | Difference | Interval | Pair-direction information |
|---|---:|---|---|
| Mean DOC | +0.462 mg/L | [-1.121, +1.913] | 10/20 pairs positive |
| Median DOC | +0.429 mg/L | [-0.390, +1.813] | 9/20 positive |
| DOC standard deviation | +0.444 mg/L | [-1.586, +1.625] | 13/20 positive |
| DOC coefficient of variation | -0.0405 | [-0.2000, +0.0881] | 13/20 positive |
| log1p DOC standard deviation | +0.0152 | [-0.0487, +0.1036] | 13/20 positive |
| DOC >=10 mg/L frequency | +4.01 percentage points | [-2.26, +14.19] | 6/20 positive |

The class-level averages overlap substantially. The current same-month
observations **do not establish a stable concentration, variability or high-
frequency ordering of broad versus elongated networks**. This is not an
equivalence result or proof of an absent morphology effect.

Native and log-space variability both remain in the outputs. SD uses ddof1;
CV divides sample SD by the mean. Concentration SD, relative CV and log-space
SD describe different properties and should not be substituted for one another.
A majority of positive paired CV differences does not imply a positive
class-average difference: some large negative pairs shift the mean below zero.
The full pair scatter/evidence table retains that heterogeneity.

The high-value threshold is the existing fixed **10 mg/L**, and counts and
denominators remain in the pair evidence. These are high-value **monthly
observations**, not an estimate of individual event peaks or the fraction of
all unobserved days exceeding the threshold.

## Measured hydro and longer records

| Population | Mean DOC difference (mg/L) | log1p SD difference | High-value frequency difference (pp) |
|---|---:|---:|---:|
| Common DOC months: 20 pairs / 11 regions | +0.462 [-1.121, +1.913] | +0.0152 [-0.0487, +0.1036] | +4.01 [-2.26, +14.19] |
| Joint measured hydro: 17 / 10 | +0.497 [-1.356, +2.100] | +0.0121 [-0.0786, +0.1081] | +4.61 [-2.85, +16.24] |
| >=24 shared months: 13 / 7 | +0.529 [-0.432, +1.512] | -0.0044 [-0.0622, +0.0761] | -0.03 [-2.36, +2.36] |

Restricting to measured hydro does not produce a clear form ordering. In the
longer-record population the high-value difference is close to zero and the
variation point estimate changes sign. The populations change together with
record length, so this cannot be attributed exclusively to sampling error or
record length. It shows that the initial +4 pp average is not a stable general
property across these predefined populations.

Hydro is measured rather than assumed identical. On joint hydro months, broad
members average +0.0512 in log1p discharge and +0.648 degrees C relative to
elongated members. Each pair's differences and coverage are recorded in
`hydro_balance.csv`; matching improved environmental comparability without
making the two rivers physically identical.

## Background-adjusted differences

The adjustment backgrounds have no class or morphological inputs. They use
calendar, environmental attributes and basin area; the hydro version adds
observed discharge and temperature, including explicit missingness handling.
They are fitted in other HUC4s and applied to each receiving region.

For the 20 main pairs, broad-minus-elongated hydro-background deviations are:

- Mean log1p deviation: **+0.0584**, interval [-0.0667, +0.1691].
- SD of log1p deviations: **+0.0150**, interval [-0.0494, +0.1069].
- Observed minus background-expected high frequency: **+3.50 pp**, interval
  [-3.03, +14.08].

These do not establish a residual form ranking either. They describe
conditional associations under a fixed empirical background, rather than a
physical or causal isolation of the shape effect. Missing discharge is not
converted to a measured value. The measured-hydro population remains alongside
this larger adjustment-based comparison.

Omitting individual HUC4s changes the native mean difference from -0.028 to
+0.840 mg/L and the log-SD difference from -0.0065 to +0.0374. The main high-
frequency point difference stays positive across omissions, but its bootstrap
interval includes zero and the longer-record comparison is near zero. Retain
both aspects of this sensitivity instead of choosing one as the narrative.

## What about the mainstem-sparse form?

The same-month comparisons retain three elongated/sparse pairs and three
sparse/broad pairs, all in HUC4 1013. Sparse members are descriptively higher:
mean difference sparse-minus-elongated +8.178 mg/L; broad-minus-sparse
-8.846 mg/L. The corresponding high-frequency differences are +15.35 pp and
-14.45 pp. These are retained in the full tables.

One region provides no geographic confidence interval. These large local
contrasts do not establish that mainstem-sparse networks generally have the
highest DOC. Additional independent sparse-form rivers are needed for that
class question.

## Additional monthly information in measured structure

The fixed linear diagnostics use exactly the same 297 stations, 18,688 observed
months and geographic folds. Fit and score station-equal means; native MAE is
2.9400 mg/L for the environment+calendar+hydro background.

| Addition to that background | Native MAE reduction | Paired HUC4 interval | Positive folds |
|---|---:|---|---:|
| Three form labels | +0.39% | [-0.41%, +1.54%] | 4/5 |
| Branch organization | +0.94% | [-0.69%, +2.14%] | 4/5 |
| Path organization | +1.16% | [-0.99%, +2.70%] | 4/5 |
| All seven morphology variables | +1.69% | [-0.60%, +3.31%] | 4/5 |

Path organization improves log-space error in five of five folds, with a
+1.61% aggregate point gain whose interval still includes zero. Full morphology
has +1.80% log-space point gain, also with an interval including zero. Report
the positive directions and their limited statistical strength together.

Adding measured hydro to the environmental/calendar background alone makes
almost no aggregate difference in this fixed linear diagnostic (+0.03% native
MAE). That does not establish unimportant hydrology: nonlinear or state-specific
responses are not represented by these additive fits.

These monthly results are substantially smaller than the earlier +9.24%
station-median information result. The targets, calendars and averaging rules
differ: a station median and every individual monthly DOC observation are
different tasks. The earlier finding remains a station-median result; it
cannot be carried over as a monthly reconstruction gain. Both studies should
appear with their actual target definitions.

## Research judgment and next question

Keep the measured three-form classification as an intelligible description
of real rivers. **Do not write a universal DOC ranking from those labels.**
The current data show heterogeneous DOC behavior inside each form, and do not
yet establish a consistent monthly shape effect after the matched comparisons.

The next morphology question should become more specific:

> Under different observed hydrologic states, do tributary organization and
> path dispersion change the DOC response of otherwise comparable river forms?

This keeps river form central while allowing an effect to depend on flow rather
than assuming a constant class offset. A later version can predefine wet/dry
or low/high-flow contrasts using flow observations alone, compare within-river
responses, and test whether those responses differ between fixed matched forms.
It should inspect data support before fitting an interaction, retain all
states, and preserve the current comparisons. It should not alter these
background models after seeing their held-region errors.

Higher-frequency tributary/receiver measurements remain the complementary
way to test path-specific arrival. That event mechanism addresses a different
resolution from monthly class-level DOC summaries. Controlled routing, observed
monthly forms, and event arrival should remain distinguishable parts of the
same morphology-centered explanation.

## Completed products

Shared-month pair records and all property differences; background states and
held-region predictions; information scores and gains; complete availability
and hydro balance; 5,000-draw paired intervals and omitted-HUC4 sensitivity;
English/Chinese figures; replay and label-isolation checks. Repository
validation is recorded in `validation.md`. No old model or paper endpoint was
overwritten and no neural model was retrained.
