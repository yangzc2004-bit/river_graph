# River morphology as an organizer of DOC transmission

## Research decision

Keep river organization at the centre of the next study. The strongest working
explanation is that **river form organizes the distribution of arrival times,
while branch balance and the shared channel environment determine how those
signals combine and are processed**. A class label is a useful description of
that organization; a fixed high/low DOC ranking is not the main result.

This analysis uses 297 receiving-station upstream-network instances and 121 independent monitored
tributary pairs feeding 38 receivers in 17 connected monitoring systems. It is a
controlled routing experiment on real geometry, paired with the preceding
observational signal analysis. It does not add a new fitted DOC model or an
independent basin validation. No landscape or vegetation source weights enter
the routing calculation.

There are 295 distinct receiving COMIDs: two station pairs share a receiving
reach and therefore the same whole-network geometry. Primary summaries retain
the fixed station cohort and use HUC4 blocks, which keep those repeats together.
They should not be interpreted as 297 independent geometric replicates.

## What changed scientifically

The previous morphology comparison showed that network descriptors contain DOC
information, and the observed connection analysis linked path organization to
signal transmission. This step now asks what changes when **the input
concentration and total flow are fixed**. It separates four quantities that are
often bundled together as “river structure”:

1. Spread of source-to-outlet channel paths.
2. Relative contributions of two tributaries.
3. Their arrival-time alignment.
4. Length and processing environment of the shared downstream segment.

This moves the explanation from differences in terrestrial DOC supply to how
the river itself redistributes and transforms a given signal.

## 1. Real paths produce different pulse shapes

Every positive-area reachable incremental catchment receives the same Gaussian
concentration anomaly (peak 1, SD 0.15). Its constant flow is proportional to its
unique incremental area; total outlet flow is one. Distances are divided by the
square root of measured basin area. The resulting common normalized velocity
and pulse width compare path organization after removing a characteristic basin
length scale. Time units are not measured days or months.

| Real-network class | Networks | Mean outlet/input peak | HUC4 bootstrap 95% CI | Mean pulse SD |
|---|---:|---:|---:|---:|
| Elongated, tributary-rich | 102 | 0.1896 | [0.1792, 0.1998] | 0.9087 |
| Mainstem-dominated, sparse | 30 | 0.4549 | [0.4155, 0.4876] | 0.4852 |
| Broad, tributary-rich | 165 | 0.2458 | [0.2367, 0.2553] | 0.6697 |

For the existing 22 environmentally matched, nonnested elongated–broad pairs in
12 HUC4s, broad minus elongated is:

| Response | Paired difference | HUC4 bootstrap 95% CI |
|---|---:|---:|
| Outlet/input pulse peak | +0.0496 | [+0.0331, +0.0794] |
| Mean arrival time | −0.2772 | [−0.3940, −0.1843] |
| Pulse SD | −0.1927 | [−0.2782, −0.1362] |
| Path-delay SD | −0.1966 | [−0.2837, −0.1392] |
| Period-4 amplitude gain | +0.1942 | [+0.1285, +0.2939] |
| Period-12 amplitude gain | +0.0419 | [+0.0288, +0.0621] |

The pulse-peak difference is about **five percentage points of the common input
peak**, not a five-percent reduction in field DOC concentration. Period-1
differences span zero: the network response depends on the timescale of input
variation, not only on a single spread statistic.

Contracting each network's paths around its own weighted mean preserves mean
arrival time, flow and integrated anomaly load. Halving path spread increases
mean peaks by 0.1334, 0.1258 and 0.1751 in the three classes respectively. Zero
spread recovers the common input pulse to numerical resolution. Thus the
within-network experiment directly isolates the signal-buffering contribution
of path dispersion under the stated routing model.

## 2. Sparse-network peaks require attention to spatial resolution

The primary calculation places each incremental catchment's contribution at its
reach midpoint. Sparse networks contain a median of only 12 positive-area
reachable reaches, compared with 561.5 and 1,226 in the two tributary-rich
classes. After the first routing readout, a sensitivity calculation distributed
each reach contribution over five equally weighted points along that reach.
This retains each reach's weight and mean delay but introduces within-reach
spread. It is an assumed distribution, not a measured source map.

Mean peaks become 0.1853, 0.3342 and 0.2413. The sparse-class peak falls by
0.1207 (about 26.5% of its midpoint estimate), so its magnitude is substantially
resolution-dependent. The main elongated–broad contrast is much less affected:
**+0.0469 [0.0328, 0.0706]**, versus +0.0496 in the primary calculation.
Its pulse-width contrast also persists: −0.1916 [−0.2768, −0.1350].

Use the elongated–broad comparison as the principal class-level mechanism
evidence. Retain the sparse class in all tables and figures, with the resolution
comparison visible. Its physical travel-time behavior has not been established
by the reach-midpoint calculation.

## 3. Branch balance changes buffering without removing DOC

The two monitored sources retain their measured source-to-receiver paths.
Replacing area-proxy branch flow shares with a 50:50 split lowers outlet/input
peak by **0.1886 [0.1404, 0.2529]** in synchronous, uniform-speed conservative
routing. Pair cases are averaged within receiver before connected-system
bootstrap. Total flow and integrated input anomaly remain identical.

When one branch dominates, its individual peak can dominate the downstream
signal. Balanced branches with different travel delays distribute the same
input over separate arrivals. Changing the input timing changes this mixing:
branch B leading by 0.30 scenario units lowers average peak by 0.0449
[0.0166, 0.0693] with area-proxy flow, whereas the corresponding lagging contrast
is not resolved (−0.0132 [−0.0431, +0.0116]). Neither is a universal phase rule;
the actual path differences decide whether a time shift aligns or separates
arrivals.

The arrival-aligned diagnostic cancels the path-delay difference. Every pair
then recovers peak 1 within 0.000263, with integrated anomaly load still one.
**Branching cannot provide unconditional buffering against a forcing whose
arrival phases compensate for that branching.** This explains why counting
tributaries alone is an incomplete DOC mechanism.

The two monitored branches are not a complete basin budget: their combined
drainage coverage has median 0.2927. Their controlled experiment describes the
combination of those two signals, not the entire receiving basin's field DOC.

## 4. Junction position matters through timing or processing

Virtual short/long common segments reallocate the branch and shared parts while
preserving both total source-to-outlet distances. They are not real channel
alterations. Four process cases distinguish the explanation:

| Scenario | Long minus short shared segment, receiver-average effect |
|---|---|
| Uniform speed, conservative routing | No change in outlet pulse, mean or load (numerical tolerance 1e−12). |
| Uniform speed, the same first-order rate everywhere | No change: total exposure is unchanged. |
| Flow-sensitive branch speed | Arrival centroid −0.0714 [−0.0973, −0.0448]; peak +0.0036 [−0.0044, +0.0099], not resolved. |
| Higher processing rate in the common segment | Peak −0.1319 [−0.1581, −0.1108]; retained-load fraction −0.1526 [−0.1814, −0.1273]. |

The speed case assumes branch velocity = flow share^(1/3), with a 1/2 exponent
sensitivity. The process case imposes rates 0.10 in branches and 0.80 in the
common segment. These rates and velocities were not estimated from DOC.
They demonstrate **which additional process would make a junction-position
effect possible**; the loss values are not field DOC removal estimates.

The inference is specific: holding total paths fixed, junction position alone
is not an independent conservative mixing mechanism. It can matter when it
changes travel-time allocation or exposure to a different processing environment.
Moving a real junction can also change total paths and other hydraulic properties,
which this controlled partition experiment deliberately holds fixed.

## 5. Connection to the observed DOC evidence

The preceding `doc_river_form_process_v1` analysis found a negative association
between path dispersion and downstream/source anomaly SD
(−0.0842 [−0.1637, −0.0093] per morphology SD), and longer normalized paths were
associated with weaker upstream/downstream signal correlation
(−0.0653 [−0.1270, −0.0108]). This routing experiment supplies a compatible
mechanism: heterogeneous arrival times can spread peaks, and path timing can
alter signal correspondence.

These studies reuse overlapping source-role networks, so they are complementary
analyses rather than independent replications. Monthly DOC does not yet fix a
physical travel-time scale. A weaker or wider downstream fluctuation is also
different from lower mean concentration or net DOC loss. In every conservative
case here, a constant common input concentration of 5 remains 5 at the outlet.

## Contribution and next experiment

The emerging contribution is **a morphology-to-transmission framework**:

> River form affects DOC reconstruction by organizing when tributary signals
> arrive, how their contributions balance, and what shared channel environment
> they traverse. Path timing and branch combination provide information that an
> undifferentiated same-month graph message does not represent.

Next, test a small geometry-guided river transmission operator against the
observed source-role signals. Keep one shared background model and compare:

1. Same-month upstream mixing.
2. Mean path delay without dispersion.
3. A distributed-delay kernel derived from real channel paths.
4. Distributed delay plus a separately estimated common-segment process term.

Use source-role validation to determine a common physical time scale, informed
by available flow/reach information, before applying it to held-out connections.
If monthly observations cannot distinguish mean delay from dispersion, identify
the higher-frequency monitored connections needed for that comparison. Keep
branch balance and shared-segment exposure as explicit terms. Terrestrial source
covariates remain fixed background, rather than replacing the river-form question.

This is a targeted scientific extension of the current model: first establish
which river-transmission operator captures additional observed information,
then attach that operator to the existing local residual model. No new deep
backbone or model training was needed for this routing study.

## Deliverables

- `study_plan.md`: inputs, manipulations and interpretation.
- `analysis/`: all 891 primary whole-network cases, 297 spatial-resolution cases,
  10,890 tributary cases, 121 arrival-alignment checks and 5,000-draw summaries.
- `figures/`: real forms and pulse responses; junction/process experiment;
  branch balance and arrival phase, each in English and Chinese, PNG/PDF.
- `verification.json`: source replay, physical invariants, numerical resolution
  and bootstrap reproduction.

Class comparisons have pointwise intervals, reused fixed pair selection, and
limited overlap for sparse-network matching. The whole-network routing is a
shortest directed single-path approximation on measured reach geometry, including
secondary links; it does not estimate hydraulic flow splits in divergent channels.
