# Real DOC signals: same-month, common delay and branch-specific arrival

## Question

Do the river timing mechanisms identified on measured geometry improve the
description of observed downstream DOC? Keep river organization central:
compare immediate mixing, one common delay, and different arrivals along the
two real monitored tributary paths.

## Observations and resolution

Reuse the permitted source-training DOC union of station roles 142/143/144 and
the independently screened tributary inventory. Restrict receivers to the fixed
297-station morphology cohort. This is an exploratory source-role study.
Existing geographic/external test predictions are not inputs to this analysis.

A data-availability audit, before operator fitting, found 59 independent branch
pairs at 22 receivers in 11 shared monitoring systems with at least 24 usable
months when both sources are observed in the current AND immediately previous
calendar month and the receiver is observed in the current month. These 3,026
pair-months define one fixed evaluation population for every operator. Missing
months are not filled or treated as adjacent observations. Longer windows lose
receivers rapidly; the primary experiment uses only current/previous months.

Monthly bins can examine history-weighted information, not identify a measured
travel time of days or establish grab-sample temporal order within a month.
Upstream DOC is a known input in this task. It is not an unmonitored-station K0
experiment: a receiver can also be an upstream input in another connection.

## Five shared-background comparisons

1. Calendar/hydro/area background without upstream DOC.
2. Same-month area-proxy mixture.
3. Uniform recent-history mixture, independent of channel distances.
4. A common delay applied to both sources, set by their area-weighted mean path.
5. Branch-specific delays, set by the two measured source-to-receiver paths.

Every comparison uses the same downstream calendar, measured hydro and basin
area inputs. The upstream concentration proxy is mixed in native concentration
space, then log1p transformed. A common ridge empirical calibrator (alpha 1)
maps the background and proxy to downstream log1p DOC; inverse output is
nonnegative. These are small diagnostic regressions, not a new deep backbone or
an estimate of net DOC processing.

Total branch weights sum to one. Source weights are fixed unique incremental
area shares. The one-parameter uniform-history operator has candidate previous-
month weights {0, .10, .25, .50, 1}. Geometry operators use the same candidate
maximum lag fractions at the longest path in the eligible geometric inventory.
For a path L, lag = fraction * L / inventory_max_path. A fractional monthly lag
splits input weight between observed current/previous bins; it does not fill
missing DOC or infer a within-month velocity. The common-delay operator retains
the area-weighted mean lag; the branch-specific operator additionally retains
the path imbalance. All operators include zero delay.

The whole upstream-basin travel kernel is NOT applied a second time to an
already integrated upstream gauge signal. Here only the two monitored
source-to-receiver paths define the compared arrivals.

## Fit and evaluate

Outer leave-one-monitoring-system-out holds all connections in a shared-source/
receiver component together. Select delay fraction only inside each outer
training population, with three-fold component validation and receiver-equal
log-space MAE. Fit imputation/scaling and the empirical calibrator only on that
training population. Held-out receiving DOC is used for scoring only; measured
upstream source channels remain legitimate inputs. Retain all inner scores,
chosen fractions and fitted coefficients. Ties favour the smaller fraction.

Each receiver contributes total training weight one: its connections have equal
weight, and dates have equal weight inside each connection. For evaluation,
average errors over dates, then connections, then receivers. No per-receiver
intercepts or fitting on its held-out DOC are allowed.

Report native MAE, log1p MAE, RMSE, signed bias, Q90 MAE (threshold from unique
outer-training receiver/month observations), and signal correlation. Compare
all operators on the exact same held-out records. Use 5,000 paired connected-
system bootstrap draws, seed42, with HUC4 sensitivity and leave-one-system-out
effect summaries. Report class summaries and continuous path imbalance/branch
balance plots without selecting a class by its results. A singleton class has
no estimable class interval.

Native prediction errors can include missing lateral sources. This diagnostic
does not estimate concentration removal from two incomplete tributary budgets.
Do not interpret a fractional-month weight as a measured hydraulic travel time.

## Scientific decisions

- Branch-specific arrival beating common delay supports useful information in
  how the paths differ, beyond their mean length.
- Geometry arrival beating uniform history supports distance organization beyond
  generic smoothing.
- Uniform history alone improving suggests useful memory, with geometry still
  unresolved at this observation cadence.
- All selected fractions collapsing to zero means the available monthly records
  do not establish added delay information; retain the real-network scenario
  result and identify finer observation needs.

No coefficients are tuned on outer held-out receiving DOC. New process losses or
flow-varying delays belong to a later version after this comparison is complete.
