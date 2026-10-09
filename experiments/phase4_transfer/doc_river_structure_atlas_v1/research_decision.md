# River structure and DOC reconstruction

## Research conclusion

Real river structure gives us a more specific model question: how should
upstream information be transformed by intervening tributaries and storage
before correcting a local DOC prediction? The station graph collapses long,
heterogeneous river paths into a single edge. The next model should encode
those paths and their observation support explicitly.

This atlas is complete. It uses the full NHDPlus cache for geometry, permitted
source-training DOC for behaviour, and existing validation predictions for
errors. It fits no model and leaves completed geographical and external
evaluation products unchanged.

## Five overlapping river settings

All 357 station reaches match the physical network. The categories describe
network position, nearby mixing and upstream storage; a station can have more
than one label.

| River setting | Stations | Physical interpretation to investigate |
|---|---:|---|
| Low-order tributary | 92 | Land-source input and local hydrologic response |
| Chain conveyance | 184 | Longitudinal propagation without a nearby major junction |
| Major confluence vicinity | 161 | Mixing between substantial tributaries |
| Integrated mainstem | 69 | Integration of multiple upstream sources |
| Lake or reservoir path | 184 | Upstream storage and possible release or transformation |

A major junction has at least 20% non-dominant tributary drainage-area share
and lies within 5 km upstream. Chain conveyance describes the same local
window. Storage denotes positive mapped LakePond/Reservoir path length within
20 km, not measured residence time. Two station positions require midpoint
approximation; their flags remain in the table. Wetland cover and 5/20/50 km
continuous metrics are available alongside the profiles.

The monitoring graph has 206 stations without a sampled upstream neighbour,
but only 12 station reaches carry the VAA headwater flag. Missing monitors
would therefore severely distort a topology classification based on station
degree alone. StreamRiver waterbody polygons are excluded from the lake and
reservoir definition.

## What a sampled river edge contains

All 324 monitored edges are reconstructed along the VAA main downstream path.
Their median station separation is **105.104 km**, with a range from **0.523
to 1,389.066 km**. A median edge traverses **49 mapped junctions**, including
**3 major junctions**; 32.72% of edges traverse some mapped lake or reservoir
path. The longest paths can therefore include considerable lateral input and
storage between the observed source and receiver.

These geometry results motivate path-conditioned messages. They do not by
themselves prove that compression caused the previous GNN's small gain.

## Source DOC behaviour

Training-role summaries cover 340 distinct stations across the three source
partitions. Stations appear once in the distribution figures, after averaging
their available partition summaries. The profiles overlap.

| Setting | Station median DOC, group median (mg/L) | DOC CV, group median | C-Q slope, group median |
|---|---:|---:|---:|
| Low-order tributary | 4.000 | 0.428 | 0.052 |
| Chain conveyance | 4.500 | 0.426 | 0.034 |
| Major confluence vicinity | 3.700 | 0.464 | 0.051 |
| Integrated mainstem | 4.017 | 0.358 | 0.077 |
| Lake or reservoir path | 4.500 | 0.427 | 0.048 |

Concentration distributions overlap substantially. Mainstems have lower
median relative variability in this panel, while confluence sites have higher
variability. The season-adjusted log1p concentration-discharge slopes do not
show a simple downstream decline. They require at least 24 paired months;
eligible counts and distribution quartiles are reported in the tables. Land
cover, climate, sampling and discharge availability can also contribute to
these differences.

For eligible upstream station pairs, seasonal-anomaly Spearman correlations
have medians **0.327, 0.149, 0.109, 0.086 and -0.054** at lags **0, 1, 3, 6
and 12 months**, respectively. Counts are 181, 160, 163, 160 and 153 physical
edges. In the fixed partition-edge subset, 141 physical edges have medians
**0.332, 0.155, 0.106, 0.085 and -0.055** across the same lags. Thus changing
edge membership alone does not account for this profile. The fixed subset is
exported and shown separately. This supports investigating available upstream
information at short monthly lags. Common forcing and uneven observation
times prevent interpretation as physical travel times or causal transport.

## Where the current model gains occur

The source-validation panel contains 140 distinct receiving stations, three
partitions and three training seeds. The complete current model has MAE
**1.723002 mg/L**, versus **1.866362** for strong matched trees and **1.769053**
for the retained complete model. Its overall reduction against trees is
**7.68% [4.96%, 10.78%]**; Q90 error reduction is **5.32% [2.77%, 8.86%]**.
These are selected development-role results, not new geographical confirmation.

| Setting | Receiving stations | MAE reduction against strong trees | 95% paired interval |
|---|---:|---:|---:|
| Low-order tributary | 32 | 10.26% | 4.26% to 15.79% |
| Chain conveyance | 72 | 10.14% | 5.84% to 14.59% |
| Major confluence vicinity | 64 | 5.06% | 2.19% to 8.39% |
| Integrated mainstem | 28 | 3.10% | -3.70% to 9.68% |
| Lake or reservoir path | 73 | 6.20% | 2.91% to 10.03% |

The weaker central mainstem gain makes mixing and path information attractive
next questions. The intervals above do not test a difference between classes;
they estimate each class's model contrast. Current attention retrieves
ecologically/hydrologically similar source observations. Its 1.09% improvement
over a seasonal-source control is not a physical river-message contribution.

## Genuine river messages in the historical validation panel

Upstream versus no-message residuals change overall MAE by **-0.029%** in the
older temporal validation task and **+0.009%** in the older spatial-task
validation. The mainstem subgroup of the latter changes by **+0.032%**. Even
where its station interval excludes zero, the absolute benefit is very small.
The spatial-task validation population includes source stations; it is not an
unmonitored-station test. Both-direction comparisons and Q90 results remain
in the exported tables.

The atlas does not establish useful transport prediction from the old operator.
It identifies the information and structural heterogeneity that a new operator
must learn to use on top of the complete current model.

## Next model experiment

Build an observation-supported river residual branch with path features for
distance, tributary mixing and lake/reservoir exposure. Use three related
operator components: longitudinal conveyance, confluence mixing and storage
memory. Let continuous physical attributes control their fusion; the five
descriptive profiles should not become five independently fitted small models.

Compare the fixed current complete model, a simple directed river correction,
a structure-conditioned correction, and a matched graph-rewiring control on
source roles first. Return to actual source availability and station-blocked
training when forming messages. Receiving DOC remains hidden in the K0 task.
Detailed design is in `next_experiment.md`.

## Reproduction and figures

```bash
uv run python scripts/build_doc_river_structure_atlas_v1.py --bootstrap-draws 5000
uv run python scripts/plot_doc_river_structure_atlas_v1.py
uv run python scripts/verify_doc_river_structure_atlas_v1.py
```

The first execution completed all tables but its final console summary used
the pandas `tail` method name instead of the `tail` column. That display error
was corrected; the successful build recomputed all outputs. The initial log
is retained. Figure margins were corrected after visual inspection. Neither
repair changes fitted models or historical predictions.

- `figures/river_structure_overview`: physical settings and compressed paths.
- `figures/doc_structure_behaviour`: source DOC distributions and C-Q slopes.
- `figures/model_gain_by_river_structure`: current and historical contrasts in separate panels.
- `figures/upstream_doc_associations`: all lags, fixed-edge control and distance.

The independent verifier checks source content, joins, physical inlet counts,
training-role DOC summaries and the main model comparison arithmetic. New
unit tests cover physical paths, waterbody semantics, hidden labels and seed
averaging. Full pytest reports **1,029 passed and 2 skipped**. Ruff passes and
the historical provenance audit exits zero under its existing known-conflict
policy. Their logs are saved locally.
Git submission remains unavailable under the current read-only Git permission;
this atlas is included in the scoped pending file list.

## Scientific context

[Creed et al.](https://www.usgs.gov/publications/river-a-chemostat-fresh-perspectives-dissolved-organic-matter-flowing-down-river)
motivate downstream integration and C-Q questions.
[McGuire et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC4024884/) motivate
multiscale connected-network analysis.
[Lynch et al.](https://www.nature.com/articles/s41467-019-08406-8) connect channel
configuration and hydrology to DOM composition; their result does not prescribe
the sign of a bulk DOC concentration response here.
