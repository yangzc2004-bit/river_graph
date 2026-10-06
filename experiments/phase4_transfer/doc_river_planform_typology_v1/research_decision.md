# Research decision: real whole-network planform

## Result and next direction

Proceed to DOC analysis using three broad empirical river-form groups alongside
the continuous shape measurements. The atlas now represents actual upstream
channels and basin boundaries, including secondary connections. It describes
whole upstream forms rather than the monitoring-station adjacency graph.

All 357 ST357 stations were attempted. Complete mapped geometry and basin
measurements are available for **353 stations (98.9%)**, representing **346 distinct
receiving COMIDs**. After listing 24 networks with fewer than five mapped reaches
separately, **322 networks enter classification**. Duplicate station COMIDs have
one geometric vote. Four large networks remain incomplete after whole-navigation
requests returned HTTP 502 and bounded fragments were explored:
07373420, 06818000, 06934500 and 07022000. Their partial channels remain cached;
they receive no morphology class in this release. Classification is completed for
the available complete-geometry cohort.

## What the real forms show

| Group | Networks | Description | Median basin aspect | Median mainstem share | Median basin area, km² |
|---|---:|---|---:|---:|---:|
| 1 | 111 | Elongated, tributary-rich | 2.32 | 12.0% | 1,656 |
| 2 | 33 | Mainstem-dominated, sparse | 1.73 | 49.4% | 65 |
| 3 | 178 | Broad, tributary-rich | 1.45 | 7.1% | 3,154 |

Group 1 combines an elongated footprint with substantial tributaries. Group 2
has a large fraction of mapped length in its mainstem, with relatively few mapped
branches. Group 3 has a broader footprint and more total tributary development.
Group 1's median drainage density exceeds Group 3's, so “broad” should not be
equated with the highest branch density per unit area. Absolute branch development,
area-normalized density and footprint width are distinct measurements.

Representatives nearest the geometric class centroids are stations **03030852**,
**06893564** and **07099200**. Two further actual examples accompany each type.
Chinese and English representative plates, class profiles and subset-stability
figures are saved in `figures/` as PNG/PDF/SVG. The maps preserve native projected
orientation and equal x/y scaling; each panel has its own physical scale bar.

## How the grouping was obtained

No DOC values enter acquisition, inclusion, features, cluster selection or example
selection. Geometry is projected to EPSG:5070. Shape, branching and organization
feature blocks have equal aggregate weight after standardization. Hierarchy order
is removed by the declared within-block Spearman redundancy rule. Ward cuts from
3–8 groups are retained; the maximum-silhouette solution meeting the planned
minimum group size of 10 has **three groups**, silhouette **0.159**.

Across 100 random 80% network subsets with preprocessing refitted, the three-group
adjusted Rand index has median **0.476** and 2.5–97.5 percentiles **0.159–0.782**.
These are broad groups in a continuous morphological space, with appreciable
boundary movement. Retain continuous variables in DOC models and report class
contrasts as a complementary summary. Nested basins share geometry; network
counts are not independent ecosystem replicates.

## Data discoveries and corrections

1. **Secondary connections matter.** Primary-only VAA traversal missed real
   in-basin channels. Adding `dnminorhyd` alongside `dnhydroseq` exactly recovered
   the NLDI UT membership in the inspected examples (1,610 → 4,124 and 226 → 764
   reaches). Primary-route preview tables are preserved separately.
2. **Use unique catchment area to validate boundaries.** Projected basin areas
   agree with sums of unique contributing VAA `areasqkm` within **0.286%** for all
   measured stations. Three Kansas stations (06887000, 06889000, 06892350) have
   discrepant cumulative `totdasqkm` values while their polygons agree with unique
   catchments. Actual polygon area is used for density and shape and should be
   used for future DOC area controls. Parent-area balance remains a VAA-based proxy.
3. **Acquisition is resumable.** Shared geometry from previous downstream
   navigation, full upstream responses and bounded fragments is reused. Station
   measurement requires every independently enumerated upstream COMID. Raw large
   caches remain in `data/raw/river_planform_v1/`.
4. **Computational changes preserve measurements.** Bulk CRS projection matches
   the scalar operation bitwise in the regression test. Cached parent-area access
   avoids repeated dataframe construction. These changes accelerate large-network
   calculations without changing the geometric formulas.

The spatial anchor is the receiving reach outlet; in-reach station position is
approximated by that outlet. Mainstem follows maximum upstream cumulative channel
length. Side imbalance uses a straight outlet-to-headwater axis; confluence
position uses receiving-reach midpoints. Definitions are in
`measurement_dictionary.md`.

## Next scientific analysis

1. Link these shapes to source-role DOC station summaries: concentration level,
   variability, seasonal amplitude and high-value frequency. Preserve existing
   hidden evaluation roles.
2. Estimate continuous effects of elongation, mainstem share and branch density
   jointly with basin area, wetlands, forest, land use, hydro conditions and record
   coverage. Area is especially important given the group-size differences.
3. Compare full versus no-message DOC errors within morphology and upstream
   observation-support strata using existing matched predictions. This answers
   which structures actually supply useful river information for reconstruction.
4. Use the resulting patterns to design a targeted river residual or retrieval
   mechanism; keep development on source roles. Geometry classification itself
   does not establish a DOC response or a new prediction gain.

## Reproduction and checks

```bash
uv run python scripts/build_doc_river_planform_v1.py
uv run python scripts/analyze_doc_river_planform_v1.py --draws 100
uv run python scripts/plot_doc_river_planform_v1.py
uv run python scripts/plot_doc_river_planform_v1.py --chinese
uv run python scripts/verify_doc_river_planform_v1.py
```

Recorded errors retry only with `--retry-errors`. The optional fragment acquisition
script retains each actual response and its request. Repeated analysis uses the
saved measured cohort and source geometry. Old Phase 0–3 results, existing model
fits, local-structure atlas and illustrative six-type artwork are preserved.

Checks: **1,055 pytest passed, 2 skipped**; Ruff passed; historical artifact audit
passed; full-network membership, mainstem endpoint continuity and unique-catchment
area checks passed for every classified network. This stage performed geometry
acquisition and analysis, with no new DOC model training.
