# Whole-network tributary entry layout and DOC signal routing

## Question

Where do lateral tributaries enter a river's mainstem, and does outlet pulse
spreading arise within individual tributaries or between their mean arrivals?
This follows the local-junction study: whole-network outline did not distinguish
250 m incoming angles in the original 22 elongated–broad matches. It does not
repeat the earlier all-source path-dispersion experiment or classify source land
cover. Previous results have been seen.

## Fixed observational population

Keep the original 297 source-role DOC receiving networks, their three outline
classes, and the original 22 class-1 versus class-3 pairs (12 HUC4 blocks).
Keep the original geometry-selected representative of each outline class.
DOC medians, monthly variability, seasonal amplitude and Q90 frequency are
context copied from the existing panel, not new event measurements.

## Routing mainstem and non-overlapping tributary units

Use the saved directed midpoint-to-outlet routes. A preliminary topology-only
audit found 3,366 consecutive links of the old geometric mainstems do not follow
the saved single-successor routes, although all their reaches are present. The
old mainstem includes alternate channels. For this analysis, select a contiguous
mainstem on the saved route tree, choosing the upstream parent with greatest
NHDPlus cumulative channel length (`arbolatesu`; COMID breaks ties). Save its
overlap with the original geometric mainstem. Original outline classes and
historical results are unchanged; this is an explicit routing definition.

Every off-mainstem reach belongs to its first lateral subtree attaching to this
mainstem. Each mainstem incremental catchment is a separate local unit. Sum
unique incremental `areasqkm`, never cumulative drainage areas. Units partition
the full reachable catchment exactly, including zero-area reaches. Lateral
entry position is remaining trunk length divided by total routing-mainstem
length: 0 is the outlet and 1 is the upstream end.

Describe lateral area share, area-weighted entry mean/spread, lower/middle/upper
third shares, and entry-area concentration. These are geometry and area proxies,
not measured discharge or DOC load.

## New mechanism decomposition

For unique source midpoint paths D and tributary/local unit G, decompose:

`Var(D) = E[Var(D | G)] + Var(E[D | G])`.

Then decompose unit mean paths into remaining mainstem distance E and mean
within-tributary distance B:

`Var(E[D | G]) = Var(E) + Var(B) + 2 Cov(E, B)`.

The signed covariance matters: widely separated entry positions need not imply
widely separated outlet arrivals when tributary length compensates. An outlet
response under conservative linear routing depends on the complete path-delay
distribution; changing entry labels alone cannot change that response.

## Controlled same-input comparisons

Use the same Gaussian concentration anomaly (SD 0.15 in mean-path units),
constant area-proportional source flows, speed proxy equal across reaches, and
unit total flow. No reactions, DOC loss, storage or fitted event speeds.

1. Actual paths.
2. Within-unit paths replaced by their unit mean: preserves global mean, removes
   only within-tributary dispersion.
3. Unit mean arrivals aligned by adding nonnegative waits: retains within-unit
   spread and removes between-unit mean dispersion. This is a diagnostic timing
   intervention, not a proposed physical river modification.
4. Actual paths plus one common translation to the same centroid as (3): shows
   that delaying the whole signal alone does not change its peak or width.

Compute exact moments and numerical peak / central-80% duration at dt=0.005;
repeat at dt=0.0025 as a numerical sensitivity. Arrival clocks are dimensionless,
not months, field travel times or physical flow velocities. Report both negative
and positive peak changes; variance reduction does not ensure a monotonic peak.

## Comparisons and outputs

Compare the original three classes and original matched pairs, using 5,000
whole-HUC4 resamples and equal network/pair weight. Report differences with
intervals, not a new outcome-selected classification. Separate geometry,
controlled responses and copied monthly DOC context in every table.

Deliver replayable partition, descriptor, variance-budget, scenario and matched
tables; three real-network maps with entry markers and arrival diagnostics;
English research decision. Test exact area partition, branch attachment,
row-order invariance, variance identities, no-junction case, translation and
same-path invariance. Run existing tests, Ruff and historical audit. No neural
training, endpoint changes, new DOC prediction or external validation.

## Post-preliminary routing sensitivity, added before its execution

The saved shortest-route analysis has now been seen. Its derived trunk has mean
COMID overlap Jaccard 0.520 with the original geometric trunk (median 0.429;
170/297 networks contain original links bypassed by the shortest route).
Consequently, add a separately reported original-mainstem-conditioned routing
scenario. Force only original mapped trunk links, verifying each against real
primary/minor NHDPlus connections; retain off-trunk saved successors. Recompute
causal distances, repeat the same partition and diagnostics, and report how much
catchment area has a changed route. This is a routing sensitivity, not selection
of a more favourable result. Preserve all saved-shortest-route results. Whole-
form conclusions must discuss agreement and disagreement between both route
definitions; neither is a measured discharge split at a bifurcation.
