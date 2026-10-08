# Tributary layout, pathway compensation and DOC signal overlap

## Research result

We have moved from naming whole-network forms to identifying how their inputs
are arranged. In the original elongated–broad matches, broad networks concentrate
lateral contributing area at fewer effective mainstem entries. Their tributary
lengths and remaining mainstem distances also compensate more strongly: longer
tributary pathways tend to enter nearer the outlet, while shorter pathways enter
farther upstream. Both findings survive two explicit routing definitions.

This provides a concrete structural mechanism to investigate for DOC pulses:
the outlet combines signals according to their complete arrival pathways.
Confluence position and tributary length can reinforce or offset one another.
The result is about whole-network arrangement, rather than a characteristic
local junction angle or a change in source land-cover composition.

## Actual network population

Retain the original 297 receiving station-network instances from 62 HUC4 groups,
with 102 elongated, 30 sparse/mainstem and 165 broad networks. They represent
295 distinct receiving COMIDs: two physical networks each have two station
aliases. The 22 original elongated–broad matches and their 12 HUC4 blocks remain
unchanged. No DOC outcomes select a representative or define a new class.

Every routed reach has one downstream successor. Unique incremental catchment
area is assigned once to a lateral subtree or a mainstem-local input. Each
lateral subtree attaches at its first entry onto the analysis mainstem. Two
networks have no positive-area lateral subtree; their outlet responses remain
in the analysis, with entry-position statistics unavailable. Catchment weights
are drainage-area proxies, not measured discharge or carbon-load contributions.

## Why two route definitions are necessary

The original mapped mainstem and the saved shortest-route mainstem are different
objects in bifurcating networks. Across 170 networks, 3,366 original trunk links
are bypassed in the saved routing tree. Their mainstem COMID Jaccard overlap has
mean 0.520 and median 0.429. A short alternate channel may change which subtree
is called a lateral tributary even when the total source-to-outlet length changes
little.

After seeing this preliminary result, we added a separate sensitivity that keeps
the original mapped mainstem. Only its successors are changed, every link is
checked against a real primary/minor downstream connection, and off-mainstem
successors remain fixed. Saved shortest-route products remain intact. In the
conditioned case, a mean 41.1% of incremental area has a changed route, but the
median area-weighted mean-path increase is only 0.092%; the maximum increase is
12.1%. Neither definition supplies measured flow splits at bifurcations.

The saved-route partition contains 9,177 positive-area lateral units and 9,121
entry instances (7,520 physical entries). Conditioning on the original trunk
gives 47,040 units and 46,535 entry instances (22,603 physical entries). These
counts expose the definition sensitivity; they are not independent rivers or
additional field observations.

## Findings that agree across route definitions

The following contrasts are broad minus elongated, using the original 22 pairs,
equal pair weight and 5,000 resamples of complete HUC4 pair groups.

| Metric | Saved shortest routes: difference [95% interval] | Original-mainstem routing: difference [95% interval] |
|---|---:|---:|
|Effective area-weighted entries|−2.730 [−4.562, −0.834]|−4.731 [−7.136, −2.537]|
|Relative total-path SD (CV)|−0.0432 [−0.0851, −0.0164]|−0.0437 [−0.0865, −0.0167]|
|Tributary-mean path variance / total variance|+0.1238 [+0.0151, +0.2229]|+0.1940 [+0.0689, +0.2998]|
|Signed entry–tributary covariance contribution / total variance|−0.2551 [−0.5200, −0.0746]|−0.4416 [−0.6708, −0.2339]|
|Entry-distance / tributary-mean-distance correlation|−0.1712 [−0.4325, −0.0051]|−0.1941 [−0.3315, −0.0485]|

Effective entries equal the inverse sum of squared lateral entry-area shares.
For the matched elongated and broad networks, their means are 7.87 versus 5.14
on saved routes and 11.14 versus 6.41 on the original trunk. These are equivalent
numbers of equally contributing entries, not literal counts of junctions.

The covariance term explains an apparent tension: broad networks have more
variation in mean within-tributary distance, but less relative variation in
complete outlet paths. The stronger negative covariance offsets that tributary
distance variation. Entry spacing alone misses this relationship.

Mean entry position is nearer the outlet in the broad matches in both cases
(differences −0.0499 and −0.0517), but its interval spans zero on saved routes.
Entry-position SD is not separated in either case. Therefore the more consistent
findings are concentration of contributing area and pathway compensation, rather
than a simple class ordering of junction positions.

## Where the outlet spreading originates

For source path D and tributary/local input unit G, the exact partition is:

`Var(D) = E[Var(D | G)] + Var(E[D | G])`.

A unit mean path then has remaining trunk distance E and mean tributary distance
B, giving `Var(E + B) = Var(E) + Var(B) + 2 Cov(E, B)`. Signed covariance is part of
the budget; its magnitude is not a fraction of DOC removed.

Matched broad networks have a larger within-unit variance share: +6.45 percentage
points on saved routes (interval −4.31 to +14.79) and +9.94 points on the original
trunk (+3.66 to +16.28). The attribution direction agrees, while its numerical
strength depends on how the mainstem is defined. Between-unit variance includes
mainstem-local inputs, so it cannot be called a purely tributary-only statistic.

## Controlled same-input experiment

All incremental catchments receive the same Gaussian concentration anomaly,
with fixed area-proportional flows and an imposed common speed proxy. We compare
actual paths, paths equalized within each unit, aligned unit-mean arrivals, and a
common translation of the original response. Unit means are aligned only by
adding nonnegative waits; the intervention isolates timing, not a proposed
physical river modification. Curves are centered separately for display.

Making unit-mean arrivals coincide increases peaks throughout this scenario
population. That increase is smaller in the broad matches by 52.90 percentage
points on saved routes (interval −91.88 to −16.25), and by 65.26 points on the
original trunk (−102.72 to −39.80). Equalizing paths within each unit has a larger
peak effect in broad networks on the original trunk (+22.33 points, +5.74 to
+37.27), but that contrast is not separated on saved routes (+0.29 points,
−41.02 to +30.23). Both cases are reported rather than selecting one result.

Within-unit equalization decreases arrival variance by identity, but its peak
decreases in 11 saved-route networks and 12 original-trunk networks. Narrower
variance does not alone establish a higher maximum for a multimodal response;
the complete pulse curves remain part of the interpretation.

Every scenario preserves integrated anomaly, unit flow and steady concentration.
A common translation leaves the peak, duration and complete centered curve
unchanged. Therefore this experiment resolves signal timing and spreading,
not DOC uptake. The time axis is mean-path units, not measured months or event
travel times. Resolution sensitivity (dt 0.005 versus 0.0025) changes any peak
by at most 0.000275 and central-80% duration by less than 0.000093.

## Observed DOC context and next research step

The existing matched monthly DOC medians, CV, seasonal amplitudes and Q90
frequencies are retained alongside the geometry. Their four paired intervals
still span zero. They are copied observational context, not new confirmation of
the controlled arrival mechanism. The earlier one-stream paired tracer study
supports the separate timing-versus-carbon-processing distinction; it does not
assign carbon loss rates to these whole-network forms.

Next, use receiving networks with simultaneous upstream and outlet DOC records
to test **pathway compensation → source-signal overlap → outlet high-DOC
response**. Keep the original whole-form comparison, mark observed source
locations on the complete network, and quantify which major entry units the
gauges actually cover. Compare aligned and separated source fluctuations on the
same observation population before attributing measured buffering to whole form.
This keeps morphology central and uses source timing as a mediator, rather than
moving the research question to differences in source land cover.

## Products and checks

The figures show real complete channel geometry, actual entry positions,
area-weighted entry profiles and same-input arrival interventions. English and
Chinese PNG/PDF versions are available for both route definitions. The original-
mainstem example is especially useful for explaining the user's long-versus-
broad whole-network distinction.

Reproduction instructions are in `README.md`; exact full-network replays are
recorded in each route definition's `verification.md`. The implementation tests
area partition, first attachment, both variance identities, row-order invariance,
translation, no-tributary networks, real-link trunk conditioning and preservation
of the original saved paths. No model training or historical results are changed.

Final checks: 297/297 complete raw-topology and pulse replays passed for each
route definition (594 instance replays); both HUC4 bootstrap tables reproduced;
all eight final English/Chinese PNGs were actually viewed after layout repairs,
with their PDF companions and figure receipts verified. The representative
mapped trunk endpoint gaps are zero in both cases. The full suite reports
1,277 passed and two existing skips; all ten new analytic-tree tests passed;
Ruff passed. Historical `audit_artifacts.py --verify` returned zero, retaining
83 parquet-only verifications, one zero-coverage artifact and the known excluded
G0 conflict. Historical predictions remain without original sidecars, and those
checks are not upgraded to full dataset/mask validation.
