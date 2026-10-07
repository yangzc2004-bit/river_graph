# Research decision: river organization as a DOC signal filter

Date: 2026-10-07.

## What this step adds

The project now links the actual outline of a river network to two operations:
how tributaries meet and how far their signals travel. These operations give a
more concrete explanation of DOC buffering than the outline label alone.

All 322 previously classified complete networks enter the geometry analysis.
They include primary and secondary NHDPlus connections, with a single geometric
vote per receiving COMID. Junction balance is the median non-dominant parent
area share at mapped confluences. Path dispersion is the unique incremental
catchment-area weighted SD/mean of channel midpoint distances to the outlet.
All positive-area sources have a reachable path in this cohort. The two axes
are split at their geometry-only medians, 0.1512 and 0.4398 respectively.

## 1. Similar outlines can contain different internal structures

| Internal profile | Distinct networks | Median junction balance | Median path CV |
|---|---:|---:|---:|
| Less balanced junctions, concentrated paths | 67 | 0.133 | 0.408 |
| Less balanced junctions, dispersed paths | 92 | 0.126 | 0.491 |
| More balanced junctions, concentrated paths | 93 | 0.189 | 0.402 |
| More balanced junctions, dispersed paths | 68 | 0.177 | 0.484 |
| No junctions | 2 | 0 | — |

These are four relative structural combinations, with the two junction-free
networks listed separately. "More balanced" means above the cohort median; it
does not imply equal 50:50 tributary discharge.

| Original outline | Less balanced / concentrated | Less balanced / dispersed | More balanced / concentrated | More balanced / dispersed | No junctions |
|---|---:|---:|---:|---:|---:|
| Elongated, tributary-rich (111) | 23 | 55 | 10 | 23 | 0 |
| Mainstem-dominated, sparse (33) | 1 | 1 | 23 | 6 | 2 |
| Broad, tributary-rich (178) | 43 | 36 | 60 | 39 | 0 |

Dispersed paths occur in 70.3% of elongated networks and 42.1% of broad networks.
Nevertheless, every outline contains all four internal combinations. A broad
outline is therefore not a sufficient description of the integration mechanism.
Mainstem length dominance and balance at individual confluences measure different
features; a long mainstem can contain locally balanced junctions.

The four displayed examples are chosen from geometry alone. Three happen to
have elongated outlines but belong to different internal profiles. Their maps
use actual channel geometries and basin boundaries, not illustrative drawings.

## 2. Relative path dispersion supplies a clear peak-damping mechanism

Every incremental catchment receives the same Gaussian concentration anomaly
(SD 0.15), with constant flow proportional to catchment area. For each network,
paths are normalized to weighted mean one. This removes differences in average
arrival time from the comparison. We then retain actual spread, halve that
spread, or make every path equal while keeping the same mean and inputs.

| Internal profile | Actual-path mean outlet peak / input peak | Half-spread peak | Equal-path peak |
|---|---:|---:|---:|
| Less balanced / concentrated | 0.355 | 0.581 | 1.000 |
| Less balanced / dispersed | 0.283 | 0.481 | 1.000 |
| More balanced / concentrated | 0.381 | 0.593 | 1.000 |
| More balanced / dispersed | 0.305 | 0.488 | 1.000 |

The dispersed-path groups have mean peaks **20.3%** and **20.0%** below their
respective concentrated-path groups. These group comparisons are descriptive
geometry contrasts, not a fitted independent effect of junction balance.
More decisively, all 322 individual networks have a lower peak with actual
spread than with half spread, and a lower half-spread peak than with equal paths
under this forcing. Integrated anomaly mass remains one, steady concentration
remains five, and mean relative delay remains one in all 966 scenarios.

The structural operation is temporal redistribution: signals arriving by
different paths overlap less at their peak and spread over a longer interval.
Peak damping is not DOC removal. The delay coordinate is relative scenario time,
not inferred velocity or days. Pulse width follows the exact mixture identity
`output SD² = input SD² + path-delay variance`; peak shape also depends on the
full path distribution. Nested networks are not independent field replicates.

## 3. Actual DOC does not give the same universal profile ordering

All 22 receivers, 59 observed tributary connections and 11 connected monitoring
systems from the previous signal study are retained. Profiles contain 5, 10, 3
and 4 receiving stations respectively. Receiver-equal native-concentration
outlet/mixture geometric SD ratios are:

| Profile | Native SD ratio | 95% system-bootstrap interval | Receivers / systems |
|---|---:|---:|---:|
| Less balanced / concentrated | 1.507 | 1.060–5.384 | 5 / 3 |
| Less balanced / dispersed | 0.598 | 0.425–0.816 | 10 / 7 |
| More balanced / concentrated | 0.361 | 0.217–0.568 | 3 / 3 |
| More balanced / dispersed | 1.089 | 0.969–1.225 | 4 / 4 |

A ratio below one means the measured outlet is less variable than the
contemporaneous mixture of its two gauged tributaries. This comparison involves
partially monitored networks and monthly records, rather than the uniform
whole-network forcing in the controlled experiment. Native calculated
two-source mixing reductions are 10.3%, 19.9%, 35.2% and 26.6%; those values
describe mixing potential and are not measured along-channel losses.

All pairwise contrasts and the log1p concentration sensitivity remain in the
tables. The first profile's log1p SD ratio is 1.020 with an interval crossing one,
so its native amplification is not a scale-robust result. The smallest apparent
ratio occurs in a group with only three receivers. Median basin areas range
from 439 to 4,561 km² among the four groups. A ranking of universal "best
buffering river types" would conceal these differences.

## 4. The measurement scale explains where to work next

Whole-network junction balance correlates only weakly with the smaller-area
share of actually gauged tributary pairs (Spearman 0.280). Whole-network path CV
also aligns weakly with the two gauged paths (0.305). Both are receiver-level
descriptive correlations, with 22 receivers and 11 systems. They establish
that these structural measurements represent different objects; they do not
identify the cause of the DOC profile differences.

After seeing the first profile tables, four exploratory continuous associations
were added, adjusting log basin area, source drainage coverage and the other
structural axis. The native log-SD-ratio coefficients per receiver SD are
−0.336 for junction balance (95% CI −2.947 to 1.251) and −0.275 for path CV
(−1.369 to 0.525). Both remain negative when each monitoring system is omitted,
but the 5,000-draw system intervals span zero. The log1p outcome gives
−0.176 (−1.769 to 0.741) and −0.034 (−0.603 to 0.448).

The largest leverage is 0.807 at receiver 05357245. Of 5,000 bootstrap draws,
4,999 retain full design rank; the excluded draw is counted. These checks
support a structural hypothesis to test more directly, not an established
adjusted DOC coefficient. All four tests were retained; no profile, outcome or
threshold was selected from their results.

The fixed-reference cuts (balance 0.25, CV 0.5) yield profile sizes 223/57/35/5,
plus two junction-free networks. The observed sample then concentrates at
19/2/1/0 receivers. This reinforces use of the continuous descriptors and
shows that the four relative combinations are a research organization tool,
not evidence for four discrete natural river categories.

Existing 205-station flow-response fits were mapped as descriptive context;
they were not refitted. Their group means mix geographic and hydrologic
differences and are not substituted for the structural mechanism result.

## Research conclusion

**River morphology is useful because it organizes how DOC signals combine and
arrive. The same outline can have different internal integration structures,
and relative path dispersion can damp peaks while conserving the total signal.**

The controlled geometry result is now concrete. The observed association needs
to be tested at the river segments actually connected by the available monitors.
That is the next scientific priority, ahead of another broad outline ranking
or a return to comparing terrestrial source regions.

## Next study

1. Build the monitored structural footprint of each tributary pair: unique
   source-to-receiver reaches, shared downstream trunk, independent branch
   segments, area shares and length dispersion. Keep whole-network descriptors
   beside these footprint descriptors.
2. Use the same observed dates and calendar anomalies to compare event
   alignment, mixture buffering and outlet variability at this matched scale.
   Treat nested connections and shared receiving sites as dependent units.
3. Compare equal-path and actual-path signal operators under the same forcing
   and saved coefficients. Examine junction balance and path dispersion as
   continuous variables before interpreting class means.
4. Only after that comparison, turn the supported operation into a model
   component: a directed, path-aware temporal filter with unit steady-signal
   gain, tested against no-message and equal-delay versions. This is a proposed
   follow-up, not a neural-model improvement measured in the current study.

This version is a geometry and DOC-mechanism analysis. It does not alter the
existing outline classes, train a new DOC reconstruction model, or constitute
independent external validation.
