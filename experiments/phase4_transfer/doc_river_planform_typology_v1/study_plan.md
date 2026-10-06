# Data-derived whole upstream river planform

## Aim and cohort
Classify real whole-network spatial forms for ST357, rather than imposing the six
illustrative prototypes. Read geometry, VAA topology and station locations only;
DOC does not determine features, inclusion, cluster count or representative cases.

## Spatial unit
Use the full connected upstream network of the station's NHDPlus COMID, anchored
at the receiving reach outlet, and its matching upstream basin polygon. This is
a reach-outlet approximation for an in-reach monitoring station, explicitly
recorded. Do not mix whole-reach networks with station-split basin boundaries.
Follow both VAA primary (`dnhydroseq`) and secondary (`dnminorhyd`) downstream
connectivity to identify expected upstream reaches;
compare NLDI UT geometry coverage against this independently enumerated membership.
Keep diversion/secondary-channel attributes as diagnostics, not inferred trees.
Upstream mainstem follows the parent with maximum upstream cumulative channel
length (`arbolatesu`); do not infer mainstem from monitoring edges.

## Acquisition
Retain the existing downstream geometry caches. Use a shared SQLite geometry
store and fetch NLDI UT only when expected upstream geometry is missing. Basin
requests use `simplified=false`, `splitCatchment=false`, and COMID features.
Use explicit 9999-km UT navigation, then compare returned membership with the
VAA full-network membership; partial responses never stand in for complete networks.
Requests are resumable, finite retries and two workers at most. Store errors and
coverage; geometry acquisition is not model training. Keep large caches in data/raw.

## Shape measurements
Project basin and channels to EPSG:5070, metre units; orientation-independent
measurements use a basin minimum rotated rectangle and length-weighted network
principal axes. Compute basin aspect ratio, basin area, compactness, network axis
ratio, lateral spread relative to main axis, drainage density, branch frequency,
hierarchy depth, mainstem share, side imbalance, tributary axial alignment and
relative confluence position. Mainstem sinuosity is reported separately from
footprint elongation. Document definitions and channel-length/geometric coverage.

## Classification
Use complete real geometry with at least five reaches and finite shape measurements.
Single-reach networks remain a separately reported sparse/undelineated shape group.
Log-transform skewed ratios/counts; standardize features and balance shape,
branching and organization blocks. Prune redundant features within each block
(absolute Spearman >0.90, first declared feature retained).
Ward hierarchy, candidate 3–8 classes, minimum preferred class size 10. Select
maximum silhouette among eligible cuts; retain every cut, resampling stability,
class sizes and the raw continuous feature table. Do not force six classes.
If fewer than 30 complete eligible networks exist, report acquisition/feature
progress and representative real cases, rather than a definitive taxonomy.
Treat nested basin dependence in later DOC analysis; no DOC association is claimed
by a geometry-only classification.

### Acquisition-driven topology correction (2026-10-06)
The primary-route preview omitted actual in-basin channels in anabranching systems.
For COMID 9277778, primary-only membership gave 1,610 reaches versus 4,124 real UT
reaches; for COMID 17963095 it gave 226 versus 764. Adding VAA `dnminorhyd` yields
an exact independent match to UT in both examples. The extra channels are inside
the basin. Preserve preview tables separately and regenerate full-network
measurements from cached geometry using both connection fields before definitive
classification. Feature blocks and class-selection rules remain the same;
no DOC was inspected.

Feature blocks (declared before clustering): shape = log basin aspect, log network
axis ratio, basin compactness; branching = log drainage density, log1p junction
frequency, mainstem share, maximum stream order; organization = tributary side
imbalance, tributary axial alignment, mainstem confluence position, tributary area
balance. Each standardized block has equal aggregate weight. Sinuosity is a
separate descriptive measurement. No channel-count or drainage-area feature is
used directly in clustering. Networks sharing a COMID are one geometric unit.
When no tributaries or mainstem junctions exist, corresponding organization
measures use structural zero; they are not treated as missing observations.
Side imbalance is the length imbalance around the outlet-to-headwater straight
axis; confluence position uses the midpoint of the receiving reach as a proxy.
Neither is an exact bend-local bank-side or confluence-point measurement.

## Real examples
For each class select the nearest observed network to its weighted class centroid
(a real representative, not a synthetic average), and retain two alternatives.
Draw actual full flowlines and basin outlines with identical line and symbol
conventions; preserve shape with equal axes and annotate physical scale and source
station. Rotation or normalization for shape comparison must be labelled.
Use imagegen only as a conceptual companion; data maps come from actual geometry.

## References
- [USGS upstream tributary navigation](https://api.water.usgs.gov/docs/nldi/navigation/)
- [USGS basin boundaries](https://api.water.usgs.gov/docs/nldi/basin/)
- [USGS basin morphometry](https://www.usgs.gov/publications/a-geographic-information-system-procedure-quantify-drainage-basin-characteristics)
