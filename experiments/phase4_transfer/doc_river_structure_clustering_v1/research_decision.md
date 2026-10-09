# River organization and DOC: research decision

## What this study answers

The physical river neighbourhood is associated with more than a station's average
DOC: it also organizes seasonal amplitude and concentration–flow response. This
study builds an empirical river typology and separates descriptive DOC patterns
from environmental adjustment. It does not evaluate a newly trained GNN.

All 357 station reaches enter geometry-only classification. DOC analysis uses
21,459 distinct source-training station-months from the union of roles 142/143/144;
340 stations have source DOC, and 333 meet the >=12 observations / >=6 calendar
months criterion. There are 233 qualifying station C–Q slopes. No geographical
confirmation or external DOC prediction table was read.

## River types

The three-class cut has the highest eligible silhouette (0.2059): 25 small steep
tributaries, 239 intermediate branching networks and 93 low-gradient integrated
networks. Classes are mutually exclusive station-centred neighbourhoods. They
are not whole-basin planforms, and small tributaries are not automatically physical
headwaters. Stream order and drainage area are strongly correlated; the declared
pruning rule retains order, while area is retained for interpretation.

The same Ward hierarchy has a useful six-class subdivision:

| Physical subtype | All reaches | DOC-eligible stations | Median river order | Median area (km²) | Major confluence within 5 km | Median upstream storage fraction (20 km) |
|---|---:|---:|---:|---:|---:|---:|
| Small steep tributaries | 25 | 21 | 1 | 6.7 | 0% | 0% |
| Confluent small tributaries | 22 | 21 | 2 | 38.6 | 81.8% | 0% |
| Confluent intermediate rivers | 80 | 73 | 5 | 779.7 | 100% | 0% |
| Few-confluence intermediate rivers | 137 | 130 | 5 | 1,431.6 | 8.8% | 0% |
| Low-gradient integrated rivers | 71 | 66 | 6 | 11,526.5 | 63.4% | 0.30% |
| Storage-influenced large rivers | 22 | 22 | 8 | 85,382.5 | 27.3% | 13.92% |

Names describe structural medians; not every member has each defining attribute.
Zero median storage does not mean all members have zero exposure. These are
network-path storage fractions, not catchment lake-area fractions.

Classification is a working typology rather than discrete natural boundaries.
Across 100 independent 80% station refits, median adjusted Rand index is 0.329
at three classes and 0.707 at six (six-class 10th–90th percentiles 0.553–0.819).
The primary cut was selected by the recorded geometry-only silhouette rule;
six-class results are retained as a nested structural sensitivity. Scale, branch
geometry and storage influence classification more than removal of the mixing
block. The first raw-storage run is archived, including its code: seven extreme
storage reaches dominated that classification. The subsequent logarithmic
storage transform was selected from geometry before inspecting DOC contrasts.

## DOC response: concentration is only part of the story

The class median of station median DOC is 3.40, 4.20 and 4.01 mg/L at the three
coarse classes. Concentration distributions overlap. A clearer descriptive
gradient occurs in annual harmonic amplitude: 0.388, 0.314 and 0.190 in
peak-to-trough log1p DOC units. This is a 51.0% lower median amplitude in the
integrated class than the small-tributary class, not a 51% reduction in native
DOC concentration or a causal estimate.

The six physical subtypes have median station DOC 3.40 / 3.51 / 3.35 / 4.60 /
4.08 / 3.88 mg/L and median harmonic amplitude 0.388 / 0.299 / 0.272 / 0.329 /
0.207 / 0.155. Large storage-influenced rivers show the weakest seasonal signal
in this descriptive comparison. The result motivates a buffering hypothesis;
lake-path exposure alone does not establish retention or DOC removal.

Coarse-class season/year-adjusted C–Q slopes have medians 0.1125 / 0.0322 /
0.0564. Small steep tributaries have the strongest concentration response to
flow in this summary. Flow uses the stored NWIS cubic-feet-per-second scale;
log1p slopes are not physical concentration–discharge elasticities.

Q90 is 10.0 mg/L, derived once from permitted source cells. Q90 frequencies
describe observed samples, not the fraction of every calendar month in a river.
Calendar curves average station monthly medians; their contributing station
population can differ between calendar months.

## What environmental adjustment changes

The models include cover, climate, local hydro, geography, HUC2 and sampling
covariates. All stations have equal regression weight; monthly observations
do not become independent ecological replicates. Intervals are conditional on
the fixed classification. Station and HUC4-block bootstrap intervals are both
reported with 2,000 draws.

- With local hydrology included, coarse classes 2 and 3 have positive log1p
  median-DOC contrasts against class 1: 0.350 and 0.353; HUC4-block intervals
  [0.115, 0.522] and [0.109, 0.548]. These are not native DOC percentage effects.
- Adjusted seasonal amplitude is lower in class 3 under station resampling
  (-0.096, interval [-0.175, -0.009]), but the HUC4-block interval crosses zero
  [-0.184, 0.032]. The descriptive seasonal gradient is clearer than its
  independent regional association.
- The intermediate class has a lower C–Q slope than class 1 (-0.220), including
  under HUC4-block resampling [-0.497, -0.005]. The integrated class contrast
  is similar in size, but its HUC4 interval [-0.488, 0.008] crosses zero.
- Overall CV contrasts are not established by either interval scheme.
- Continuous structure gives additional hypotheses: the transformed 50-km
  storage fraction has a positive concentration coefficient per station SD
  (0.078, HUC4 interval [0.003, 0.163]), while 20-km storage does not. Junction
  density at 20 km has a positive station-bootstrap signal, but its regional
  interval crosses zero. Retain every coefficient rather than selecting these
  from the table as confirmed mechanisms. Multiscale structure variables remain
  correlated; 50-km storage is not an isolated physical intervention.

Without local hydro controls, some concentration contrasts weaken. Thus both
hydrology-conditional and no-local-hydro associations are kept. Static 2019 land
cover and 1991–2020 climate approximate a long historical DOC record; local flow
is unavailable at source DOC times for 66 of the 340 source stations. Availability
and missingness indicators are included, but do not eliminate measurement or
sampling differences. Exact duplicate missingness columns are removed and
reported; effective linear designs have condition numbers about 25–40.

## Does structure add information beyond environment?

Five HUC4-blocked folds predict *station median DOC*, using fixed ridge alpha=10,
with fit preprocessing restricted to each fold's training stations. The physical
taxonomy uses all available geometry, without DOC. This is an explanatory
prediction diagnostic, not the previous new-station reconstruction experiment.

| Inputs | Equal-fold log1p MAE | Equal-fold native MAE (mg/L) |
|---|---:|---:|
| Environment and sampling | 0.3280 | 2.1744 |
| Environment + coarse river classes | 0.3133 | 2.0719 |
| Environment + continuous structure | 0.3158 | 2.0755 |

Classes improve log1p MAE by 4.48% (native 4.72%); continuous structure by 3.71%
(native 4.55%). Both improve all five folds. This supports river structure as a
useful information source for this diagnostic. It is not a measured improvement
of the released DOC model, nor proof that a message-passing GNN realizes it.

## Upstream DOC information

The fixed physical-edge subset has 141 edges. Receiving-class 2 and 3 same-month
seasonal-anomaly median correlations are 0.352 (61 edges) and 0.327 (79 edges).
At 1 / 3 / 6 / 12 months they are 0.129 / 0.055 / 0.033 / -0.121 and 0.169 /
0.135 / 0.109 / -0.030. The small-tributary group has only one eligible edge
and cannot support an upstream-class comparison. Shared endpoints, common forcing
and monitoring distance matter; a monthly lag is not an observed travel time.

## Research decision

Continue the river-structure science. The strongest storyline is not that one
river type always has higher DOC. River organization is associated with **the
shape of DOC dynamics**, while also providing a modest information increment
beyond environment in a station-median diagnostic.

Next investigate three mechanisms with actual connected stations:
1. **Confluence mixing:** compare tributary and receiving seasonal/flow responses,
   branch-size balance and upstream observation completeness.
2. **Longitudinal integration:** test whether downstream seasonal signals are
   smoother after accounting for local forcing and observation coverage.
3. **Storage buffering:** contrast storage-exposed paths with comparable paths
   of similar order, distance, ecology and hydrology.

Use continuous structure alongside coarse and fine types. Only then implement
structure-conditioned GNN mixing, conveyance and memory operators. The earlier
river-residual pilot already suggests that supported nearby upstream information
is useful; these results identify what information a next operator should model.
No neural fitting or release change was performed in this study.

## Reproduce

Using the project's uv environment:

```bash
uv run python scripts/analyze_doc_river_structure_clustering_v1.py
uv run python scripts/plot_doc_river_structure_clustering_v1.py
uv run python scripts/verify_doc_river_structure_clustering_v1.py
```

Five figure families are available as PNG/PDF/SVG and were visually inspected.
The full suite passed: 1,045 tests, 2 skips; Ruff passed; historical artifact
verification exited 0. Older study files, predictions and manuscript remain intact.

Methods: [Ward hierarchy](https://scikit-learn.org/stable/modules/clustering.html#hierarchical-clustering),
[adjusted Rand index](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_rand_score.html).
