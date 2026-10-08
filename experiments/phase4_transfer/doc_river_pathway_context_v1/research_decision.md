# Research decision: river pathways and receiving DOC variability

Date: 2026-10-08

## Scientific advance

The study follows the user's whole-network question: long, elongated networks
versus broad, tributary-rich networks. It now distinguishes **variation among
tributary inputs** from **the receiving signal after they traverse the river**.
Actual gauge-to-receiver paths, their common downstream corridor, mainstem entry
positions and mapped waterbodies provide the structural context.

Keep 32 fixed monitored frontiers, seven complete catchment-overlap systems and
the original classes. Their original-mainstem paths contain 10,298 source/reach
records; a common reach occurs in each contributing source path, rather than
being counted as several independent physical reaches. All DOC is restricted to
the previously permitted source-training cells. No model is trained, and no
geographical or external test prediction is changed.

## 1. A positive whole-form lead emerges with adequate monitoring

Twelve networks represent at least 80% of their mapped area: four elongated and
eight broad networks, across five overlapping-catchment systems. After removing
the common calendar/year design, the geometric mean receiving/mixture SD ratio
is **1.735 for elongated** and **0.700 for broad** networks. Their ratio is
**0.403**, paired complete-system bootstrap interval **0.194 to 0.940**.

In this particular high-coverage sample, the broad receiving signal is smaller
relative to its observed tributary mixture. The estimate is a ratio of
geometric means of SD ratios, not 60% lower DOC concentration, a measured peak
reduction or DOC removal.

Report the entire metadata-defined coverage ladder:

| Minimum represented area | Elongated / broad networks | Broad-minus-elongated log-SD ratio | 95% system interval |
|---|---:|---:|---:|
| 0% | 9 / 21 | -0.127 | -0.795 to +0.258 |
| 50% | 4 / 16 | -0.628 | -1.406 to -0.063 |
| 70% | 4 / 11 | -0.848 | -1.617 to -0.099 |
| 80% | 4 / 8 | -0.908 | -1.638 to -0.062 |
| 90% | 3 / 4 | -0.740 | -1.531 to +0.143 |

The direction persists as coverage improves, but the 90% cohort is smaller and
its interval spans zero. This is an exploratory follow-up after prior results
were inspected; the 80% working cut and sensitivity ladder were written before
the new calculations. Different cuts contain different networks and are not a
within-network treatment curve. The original all-receiver null form contrast is
preserved alongside this new lead.

## 2. The exact obstacle to attributing that pattern to shape

Among all 295 physical receiving networks, there are 473 elongated/broad
candidate pairs in the same HUC4. Of these, **105 have an area ratio <=2**.
**None of those 105 has adequate simultaneous source/outlet DOC at both
receivers under the current ST357 source-role availability**, even before
requiring >=80% coverage. These are candidate
combinations with reused stations, not 105 independent field pairs.

Thus the high-coverage class comparison contrasts regional contexts as well as
river form. Same HUC4 and comparable area would themselves still be only coarse
controls, but the current sampling does not reach even that overlap. The
sampling gap is more specific than simply needing more months at the existing
outlets: both whole forms need observed tributaries and a receiver in the same
region and size range.

The scientific decision is to keep the high-coverage contrast as a **field
hypothesis worth following**, while using the earlier identical-input routing
experiments as structural mechanism evidence. A controlled pulse experiment
and a field receiving/tributary SD ratio are different outcomes; their form
directions should not be merged into one universal attenuation claim.

## 3. Tributary mixing leaves substantial receiving variation unexplained

On common dates, receiving anomaly Y equals upstream-mixture anomaly M plus
the discrepancy D = Y-M. Its exact variance identity is:

**Var(Y) = Var(M) + Var(D) + 2 Cov(M,D).**

This identity is checked directly from every saved series. The discrepancy can
contain unmonitored inputs, changing shares, sampling mismatch and route
processes. It is not an identified local DOC source or a concentration mass
balance. Negative covariance often offsets large mixture and discrepancy
variances, making simple variance shares inappropriate as physical source
percentages.

A complementary linear diagnostic allows the mixture's amplitude to adjust.
The receiver-equal mean fraction not explained by its contemporaneous mixture
is **77.1% [68.5%, 84.1%]** across all 32 networks. It remains **76.1%
[65.6%, 85.9%]** in the high-coverage 12. This is 1 minus squared anomaly
correlation on the same sample, not out-of-sample predictive performance.
High mapped area coverage does not by itself make the receiving signal a simple
weighted average of measured tributaries.

## 4. The new structural measurements

The original-route inventory identifies lake/reservoir tags on 19 of the 32
monitored corridors. Tags establish mapped presence, while a missing tag does
not establish a river-only route. Source-to-receiver length includes partial
gauge and receiver reaches using their NHD measures. The common corridor is the
contiguous downstream suffix shared by every source, counted once in its length
summary. Mainstem-entry and independent-branch length variances plus their
covariance reproduce the total source-path variance exactly.

For the high-coverage 12, the mapped storage-length fraction has an exploratory
correlation of **-0.359 [-0.929, -0.238]** with the mixture-unexplained fraction.
Its leave-one-system-out correlations retain the negative direction. However,
only 4,616/5,000 draws retain variation in that geometric variable, and the
largest storage fractions occur in very few receiving networks. This was one
of several exploratory associations; neither the all-receiver estimate nor the
storage/receiving-amplitude estimate separates zero. The lead concerns **how
well the outlet preserves tributary fluctuations**, not demonstrated smoothing
by lakes. Common-path length and path dispersion do not yet give stable direct
associations with receiving amplitude.

## 5. Flow shares do not automatically settle the question

Eighteen receivers have enough common months with positive monthly discharge at
every source and receiver, across six systems. Compare area shares and varying
flow shares on precisely these same observations. The flow-minus-area change
in log receiving/mixture SD is **+0.013 [-0.077, +0.136]**. The change in mixture
unexplained fraction is **+0.014 [-0.005, +0.031]**. Neither identifies an overall
improvement.

Seven receivers also pass the 0.8--1.2 source-sum/receiver flow check with enough
common months. Their paired intervals again span zero. Positive measured
monthly flows and closure are useful context; monthly means do not reconstruct
synchronous parcel mixing, discharge-weighted monthly DOC loads or transit time.

Shortest routing retains the 12 high-coverage receivers and their form contrast
exactly. It identifies 17 tagged monitored corridors and 19 flow-eligible
receivers because some source selections and common months differ. Its paired
flow conclusions remain unchanged in direction and uncertainty. Do not treat
different routing populations as one perfectly matched field cohort.

## 6. Loch Vale: route context is now directly mapped

The new corridor audit resolves a detail left open by the previous study. The
gauges and receiving reach themselves have no lake tag, but **their intervening
common corridor contains COMID 13676, waterbody 11922, with 0.413 km of tagged
lake reach**. The common downstream corridor is 1.033 km and the area-weighted
gauge-to-receiver mean is 1.191 km. Tagged lake length accounts for 34.7% of
that mean path, and both source routes cross it. Length fraction is not lake
volume or residence time.

This matches the documented Loch Vale setting. Historical field research
describes within-lake and terrestrial organic carbon inputs that vary by season.
[Baron et al., 1991, USGS](https://www.usgs.gov/publications/sources-dissolved-and-particulate-organic-material-loch-vale-watershed-rocky-mountain).
The DOC-gaining tributaries and surrounding-forest inputs are also described in
[McKnight et al., 1997, USGS](https://pubs.usgs.gov/publication/70020284).
These studies establish site context, not the cause of the present variance result.

On the **same 80 flow-complete monthly observations**, receiving/mixture SD is
1.38 with area shares and 1.36 with flow shares. Thus changing monthly weights
alone leaves the receiving amplification present. The earlier within-month
262-day diagnostic, with 1.60 SD ratio, is retained as a different, finer
sampling population rather than combined with these monthly calculations.

## Research direction

Continue **whole form -> entry/path arrangement -> overlap -> receiving
variability**, with two concrete targets:

1. Build paired field opportunities from the 105 same-HUC4, comparable-area
   candidates. Prioritize common sampling of each network's distinct tributaries
   and receiver; more nested outlets alone will not provide form replication.
2. Within real mapped networks, separate the arrangement of branch lengths and
   entry positions from changes in the shared river/lake corridor. Retain the
   high-coverage form lead and the storage/signal-preservation lead as hypotheses
   to test with comparable input timing and better-resolved discharge/DOC.

The main question stays structural. The present result supplies an observed
shape-related lead and identifies precisely what information is needed to
distinguish a river-form mechanism from regional or receiving-process context.
