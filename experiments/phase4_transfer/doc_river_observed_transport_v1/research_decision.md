# Research decision: measured river paths and observed DOC arrival

## What this step answers

The preceding real-geometry routing experiment showed how path dispersion and
branch timing can buffer an identical DOC pulse. This study asks whether those
ideas improve the description of **observed downstream DOC**. It compares
same-month tributary mixing, generic recent history, a common path-based delay,
and separate arrival weights along the two real tributary paths.

This continues the river-form question. Terrestrial source composition is not
the new explanatory target. The historical DOC reconstruction neural models
were not retrained, and their geographic/external test predictions were not
used here.

## Population and analysis

- 59 independently screened tributary pairs, 22 receivers, 11 connected
  monitoring systems, and 15 receiver HUC4 regions.
- 3,026 connection-month records represent **1,092 unique receiver-month
  observations**, not 3,026 independent downstream truths.
- Both sources must have observed DOC in the current and immediately previous
  calendar month. Each pair has at least 24 usable receiving months. Every
  operator uses exactly the same rows, with no gap filling.
- The source DOC permissions are the existing source-training union of station
  roles 142/143/144. This is an exploratory source-role analysis of monitored
  tributaries. It is not unmonitored K0, external validation, or a new geographic
  confirmation of the full reconstruction model.
- Each connected monitoring system is held out in turn. Lag fractions are
  selected in three inner component folds; the held receiver labels do not
  choose the lag, imputation, normalization, coefficients or tail threshold.
- A shared ridge calibrator in log1p space uses calendar, receiver hydro and
  basin area, plus the corresponding native-concentration upstream proxy.
  This is a deliberately small diagnostic background, not the current full
  environmental tree/neural baseline.
- Dates average within connections; connections average within receivers;
  receivers have equal weight. Intervals use 5,000 paired connected-system
  bootstrap draws, with HUC4 sensitivity and omitted-system effect summaries.

## Observed results

| Operator | Receiver-equal MAE (mg/L) | Log1p MAE | Q90 MAE (mg/L) |
|---|---:|---:|---:|
| Calendar / hydro / area background | 2.1243 | 0.3665 | 5.7568 |
| Same-month upstream mix | 1.6669 | 0.2768 | 5.0140 |
| Uniform recent-history weight | 1.6354 | 0.2716 | 5.0424 |
| Common mean-path delay | 1.6412 | 0.2723 | 4.9507 |
| Separate branch arrivals | 1.6416 | 0.2724 | 4.9465 |

The table uses native MAE for concentration performance and log1p MAE for the
predefined inner selection objective. The results are descriptive of this
fixed monitored population; comparisons use the paired intervals below.

### 1. Known upstream concentration is useful

Adding the same-month upstream mixture lowers native MAE by **21.53%** relative
to the small shared background: 0.4573 mg/L, paired 95% CI [0.2561, 0.8426].
Eighteen of 22 receivers improve. Log-space improvement is also supported.
The effect stays positive when any one monitoring system is omitted.

This establishes information in the allowed upstream concentration inputs. It
does not isolate river geometry from a matched, unconnected-station reference,
and it is not a 21.53% gain over the full current DOC model.

### 2. Recent history adds a small improvement

Relative to same-month mixing:

| Candidate | Native MAE reduction | Paired 95% CI for reduction | Receivers improving |
|---|---:|---:|---:|
| Uniform history | 1.89% | [0.08%, 4.27%] | 13/22 |
| Mean-path delay | 1.54% | [0.30%, 3.39%] | 15/22 |
| Separate branch arrival | 1.52% | [0.27%, 3.38%] | 15/22 |

Uniform history has the lowest aggregate native and log error. Its paired
log-space difference against same-month mixing includes zero. Mean-path and
branch-arrival log-space differences against same-month mixing are positive.
Native gains for history/arrival remain positive in every omitted-system
summary. Improvements are small compared with the gain from knowing upstream
DOC at all.

### 3. Separating the two path delays has not improved overall error

Branch arrival versus mean-path delay:

- Native error reduction **-0.00037 mg/L**, 95% CI [-0.00094, 0.00087].
- Relative reduction **-0.023%**, 95% CI [-0.058%, 0.067%].
- Log-space reduction -0.000091, 95% CI [-0.000205, 0.000131].

Branch arrival versus uniform history has a -0.00616 mg/L point difference,
95% CI [-0.02186, 0.00958]. Neither comparison establishes extra overall
information in path-specific arrival. This is not an equivalence finding and
does not invalidate the real-network pulse-routing mechanism.

Every geometry-based outer fold selected fraction **1**, the top of the fixed
candidate grid. Uniform-history choices were 0.25 or 0.50. Geometry fractions
are therefore boundary-selected monthly input weights, not estimated river
velocities or identified physical travel times. The grid has not been expanded
after inspecting held-out receiving outcomes.

### 4. Retain the small tail signal

Branch arrival has 0.00417 mg/L lower Q90 error than common mean-path delay,
paired 95% CI [0.00027, 0.01384]. The branch-versus-same-month tail reduction
is 0.06743 mg/L, CI [0.00679, 0.11330]. These are retained rather than hidden
because the overall comparison was inconclusive.

The tail population has only 122 unique receiver-months, 11 receivers and six
monitoring systems; just three receivers have at least 20 high-DOC months.
Branch arrival has not established a tail advantage over uniform history
(CI includes zero). The very small incremental branch-versus-mean result is a
follow-up clue, not a reason to change the primary question to tail selection.

## What can be said about the three river forms?

| Fixed form | Receivers | Connected systems | Same-month MAE | Branch-arrival MAE |
|---|---:|---:|---:|---:|
| Elongated / tributary-rich | 6 | 1 | 0.9046 | 0.9112 |
| Mainstem dominated / sparse | 1 | 1 | 3.3336 | 3.5061 |
| Broad / tributary-rich | 15 | 11 | 1.8608 | 1.8095 |

Most aggregate history improvement is observed in the broad-form receivers.
However, all six elongated receivers belong to one shared observation system;
their point count does not provide six independent tests of this form. The
sparse form has one receiver, where upstream input itself worsens the small
background comparison. All forms and failures remain in the tables.

Primary connected-system class intervals are unestimable for the elongated and
sparse forms. HUC4 sensitivity cannot create independent monitoring systems.
The current observed comparison cannot resolve an elongated-versus-broad form
mechanism merely from these class averages.

## A precise mechanism to test next

For source current values A and B, previous values A' and B', area share w,
and fractional path delays dA and dB, the native upstream proxy difference is:

```text
branch_arrival - mean_delay
  = w(1-w) * (dA-dB) * [(A'-A) - (B'-B)]
```

Separate branch arrival needs **both unequal travel paths and different
changes in the two branch signals**. Equal paths or parallel source changes
produce exactly the same proxy. Balanced branches give the path difference
more opportunity to matter. These identities are tested against hand-computed
examples; they do not estimate a DOC processing rate.

This makes the next morphology study concrete: compare how elongated and
broad networks organize **differing branch pulses**, rather than asking only
whether their long-term average DOC differs.

## Next research actions

1. Audit finer-date raw DOC coverage in independent elongated and broad
   monitoring systems. Preserve measured sample dates, look for upstream and
   downstream coverage through the same events, and catalogue actual intervals.
   Selection should use geometry and observation coverage, not the best
   downstream fit. Monthly bins alone do not locate submonthly pulse arrival.
2. Include systems where branches differ in path length and in their observed
   pulse phase, plus balanced and dominant-branch examples. Keep the three
   measured form definitions fixed and record coverage gaps explicitly.
3. With usable records, compare shared delay and path-specific arrival against
   generic history on the same receiving samples. Test attenuation and arrival
   timing separately; bring measured flow into source weights when available.
4. Add a matched, unconnected-station information reference in a later version
   to separate knowledge of upstream concentration from the extra value of a
   real routing relationship. Keep river-form/path contrasts the main question.
5. Transfer a geometry term into the reconstruction model once its additional
   observed information is demonstrated. This version does not justify adding
   a more complex graph branch just from the scenario results.

The current evidence chain is: **measured form changes pulse organization in
controlled routing; observed upstream DOC carries information; monthly history
helps modestly; observed shape-specific arrival still needs independent,
better-resolved tests.** This provides a focused experiment about river form,
not a switch back to terrestrial source differences.

## Deliverables

- Frozen study plan and configuration; all fitted states and inner trials.
- Observed connection predictions with a complete input/source snapshot.
- Receiver-balanced metrics, tail counts, class summaries, 5,000-draw paired
  intervals and omitted-system sensitivity.
- English/Chinese figures for operator comparisons and real-form diagnostics.
- Full replay verification, label-perturbation checks and meaningful algebraic
  tests. Repository validation is recorded separately in `validation.md`.
