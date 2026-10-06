# Whole-network planform measurements

The observational unit is the connected upstream NHDPlus network ending
at the downstream outlet of the station's receiving COMID. An in-reach monitoring
station is approximated by that reach outlet. Reaches are independently enumerated
from VAA `hydroseq`, `dnhydroseq` and `dnminorhyd`, then matched to actual NLDI
channel geometry. Secondary connections matter in anabranching systems and are
included, not silently removed to manufacture a tree.
The polygon includes the full receiving catchment (`splitCatchment=false`).
All geometry is projected from WGS84 to CONUS Albers EPSG:5070 before measuring.

| Measurement | Definition | Interpretation |
|---|---|---|
| Basin aspect | Long/short edge of minimum rotated bounding rectangle | Overall narrow versus broad basin footprint, independent of north orientation |
| Network axis ratio | Square root of major/minor eigenvalues of channel-length-weighted segment-midpoint covariance | Spatial spread of actual channels, distinct from the basin boundary |
| Basin compactness | `4*pi*area/perimeter²` | Compactness of the full-resolution basin boundary; sensitive to boundary detail |
| Basin elongation proxy | Equal-area circle diameter / long rectangle edge | Descriptive geometric proxy; not a river-length-based Schumm elongation index |
| Drainage density | All measured channel length / basin area, km/km² | Amount of mapped river per unit basin area |
| Junction frequency | Receiving reaches with at least two upstream parents / basin area, km⁻² | Branching density; junction location represented at reach level |
| Hierarchy order | Maximum VAA `streamorde` | Branching hierarchy; still related to network scale |
| Mainstem share | Mainstem measured length / all channel measured length | Mainstem dominance versus tributary-rich organization |
| Side imbalance | Absolute left/right tributary-length difference / sum | Asymmetry around the straight outlet-to-headwater axis, not bank-local asymmetry on bends |
| Tributary axial alignment | Length-weighted `abs(mean(exp(2i*segment_angle)))` | Parallel versus varied tributary orientations; direction reversal leaves it unchanged |
| Confluence position | Mean normalized mainstem distance of junction-bearing receiving-reach midpoints | Approximate location of merging along the mainstem, measured from outlet |
| Tributary balance | Median `1-max(parent_area)/sum(parent_area)` at junctions | Balanced versus strongly dominant branches |
| Mainstem sinuosity | Mainstem channel length / outlet-to-headwater chord | Curvature of the mainstem, kept separate from overall footprint shape |

Mainstem follows the parent with maximum cumulative upstream channel length
(`arbolatesu`). Flowline coordinate sequences are stitched by endpoint proximity,
not assumed to have a uniform stored orientation. Maximum endpoint gap is reported.
No branch is drawn by joining monitoring locations with straight lines.
Parent drainage areas can overlap where a network splits and rejoins, so tributary
balance is an organization proxy rather than an additive drainage-area budget.

For branchless networks, side imbalance and tributary alignment are structural
zeros. When no mainstem junction exists, confluence position is a structural zero.
Networks with fewer than five mapped reaches are listed but not assigned a shape
class. Duplicate station COMIDs receive one geometric vote in clustering.

Classes are descriptive combinations of footprint and organization, rather than
the six pre-drawn illustrative prototypes. Representatives are observed networks
nearest each class centroid; maps retain native projected orientation and equal
x/y scales. Different panels use labelled independent physical extents.

Nested upstream basins share geometry. Class counts therefore count geometric
units, not independent ecosystem replicates. Later DOC comparisons should account
for shared basin/geographical context and ecosystem covariates. This classification
alone does not measure a DOC effect.

## Source interfaces

- [USGS NLDI upstream navigation](https://api.water.usgs.gov/docs/nldi/navigation/)
- [USGS NLDI full basin boundaries](https://api.water.usgs.gov/docs/nldi/basin/)
- NHDPlus VAA: existing `cache/nldplus_vaa.parquet`;
  original downstream flowline caches: `data/raw/nldi/flowlines/`.
- New upstream responses, polygons and shared channel geometry:
  `data/raw/river_planform_v1/` (local, re-fetchable cache).

## Commands

```bash
uv run python scripts/build_doc_river_planform_v1.py
uv run python scripts/analyze_doc_river_planform_v1.py --draws 100
uv run python scripts/plot_doc_river_planform_v1.py
uv run python scripts/verify_doc_river_planform_v1.py
```

Partial-cohort figures are explicitly labelled provisional and use separate
directories. Final class count is selected after acquisition, not from DOC scores.

Large navigation requests can be completed by combining actual bounded UT
fragments at missing upstream frontiers (`fill_doc_river_planform_gaps_v1.py`).
Fragments never substitute for complete networks: full VAA membership must be
present in the shared store before station measurements are accepted. Acquisition
errors remain listed; classification can be completed for the available complete
geometry cohort after every station has been attempted.

For area validation, sum **unique** VAA `areasqkm` over the independently enumerated
network and compare with the projected basin polygon. Three Kansas stations have
large discrepancies in the cumulative `totdasqkm` field even though the polygon
agrees with the unique-catchment sum. Use measured polygon area for shape/density
and future DOC controls. The cumulative attribute is a diagnostic, not a basin
completeness criterion. Parent-area balance remains a descriptive VAA proxy.
