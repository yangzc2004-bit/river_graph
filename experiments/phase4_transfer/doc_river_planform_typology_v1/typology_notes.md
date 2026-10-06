# Whole-network river planform: working typology

## Unit and meaning
The user's intended structure is the spatial organization of a complete upstream
river network: narrow, elongated networks versus broad networks with many
tributaries. It is distinct from the previous station-neighbourhood classification
based on order, local confluences and storage. Network breadth concerns geographic
spread, not channel width. Mainstem sinuosity is an independent attribute.

Use each station's physically connected upstream network and drainage basin as
the primary unit. The same upstream reach may occur in nested station networks;
later DOC analysis must account for this dependence. A hydrologic unit can serve
as a sensitivity unit, but HUC boundaries are not interchangeable with exact
upstream contributing basins.

## Six illustrative prototypes

These are proposed project descriptors, not six universal mutually exclusive
geomorphological classes or the measured classification of ST357. A real network
can combine several attributes. Future clustering will use continuous geometry,
without imposing these six prototypes or using DOC to choose the classes.

| Panel | Chinese label | English descriptor | Visual definition | Candidate discriminating measurements |
|---|---|---|---|---|
| A | 细长少支流型 | Sparse elongated network | Narrow geographic footprint, long dominant trunk, few lateral tributaries | High length/width ratio; high mainstem length fraction; low branch frequency |
| B | 细长多支流型 | Branched elongated network | Narrow footprint with abundant tributaries entering along the trunk | High length/width ratio; higher branch frequency and hierarchy depth |
| C | 宽展树枝型 | Broad dendritic network | Broad footprint with nested, broadly balanced tree-like tributaries | Low length/width ratio; high lateral spread and branching; comparatively balanced subtree size |
| D | 扇形汇聚型 | Fan-shaped convergent network | Several major tributary arms converge near a common downstream outlet | Broad upstream spread; tributary bearing distribution; downstream concentration of major confluences |
| E | 单侧偏枝型 | Asymmetric tributary network | Most tributary extent and contribution lies on one side of the dominant trunk | Left/right upstream-area or tributary-length imbalance, evaluated relative to the directed trunk |
| F | 梳状近并行型 | Comb-like, near-parallel tributaries | Tributary axes are approximately parallel and enter a common longitudinal trunk | Tributary bearing alignment; junction angles; longitudinal junction spacing |

Sinuosity is measured separately, such as along-channel mainstem length divided
by end-to-end distance. A narrow network can have a straight or highly sinuous
mainstem. Longest-channel length alone does not define footprint elongation.
Bearing alignment is axial (directions separated by 180 degrees can represent
parallel valleys). Shape measurements require projected metre coordinates,
consistent upstream extent and a recorded resolution of included channels.

## Measurements for an actual classification

1. Upstream-basin length/width and area-based elongation; network principal-axis
   spread as a complementary channel-footprint descriptor.
2. Total channel length divided by contributing basin area, branch counts per
   area and hierarchy depth. Channel inclusion/resolution must be consistent.
3. Mainstem share of total channel length; tributary subtree size balance.
4. Relative positions of major confluences along the normalized mainstem.
5. Tributary left/right balance, junction angles and bearing alignment.
6. Mainstem sinuosity, retained separately from shape and branch density.

Extract geometry from actual flowlines and basin boundaries, not the map of 357
monitoring points or the 324 compressed monitoring edges. The cached VAA topology
is useful for tracing connectivity but does not by itself encode planform curves
or exact basin width. Inspect cached geometry availability before building these
measurements. Existing 5/20/50-km topology summaries remain complementary.

## Scientific use
First identify structural groups from geometry. Then compare station DOC level,
variability, seasonality, concentration–flow response and upstream associations.
Ask whether narrow long networks show different dynamics from broad, highly
branched networks, and whether confluence position or asymmetry adds information.
Do not assign DOC effects to an illustrative prototype before those analyses.

## Figure
The six-panel Chinese illustration is saved at
`docs/figures/river_planform_typology_v1/river_planform_six_types_cn_v1.png`.
It was generated with the built-in imagegen tool and visually inspected for
labels, distinct planforms, convergent flow and single outlets. It is a conceptual
figure, not geographical data. The exact prompt is retained beside the image.
No previous manuscript figure or experiment was replaced.

## Source context
USGS treats drainage-basin boundary, drainage network and basin length as separate
GIS inputs to quantify morphometric basin characteristics:
[Eash, basin characteristics procedure](https://www.usgs.gov/publications/a-geographic-information-system-procedure-quantify-drainage-basin-characteristics).
The six prototype labels above are the project's proposed operational descriptions.
