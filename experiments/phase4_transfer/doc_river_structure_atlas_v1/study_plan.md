# River structure and DOC reconstruction atlas

## Scientific question

Which physical river settings contain DOC information that local environment
and time models fail to reconstruct? The atlas connects NHDPlus reach topology,
source DOC behaviour and existing validation errors. It prepares a structure
aware river residual experiment on the current complete model.

## Structure definitions

Use the full cached NHDPlus VAA network, with upstream connections defined by
an incoming reach's `tonode` matching the receiving reach's `fromnode`. Compute
upstream neighbourhoods within 5, 20 and 50 km of the station. Distances follow
reach lengths, including the station's fractional reach position. Missing
positions use the reach midpoint and receive an explicit approximation flag.
These are distances along a network, not travel times or catchment radii.

Keep continuous stream order, drainage area, slope, upstream length, branching,
tributary area balance and waterbody-path metrics. Overlapping profiles are:

| Profile | Definition |
|---|---|
| Low-order tributary | Strahler order 1 to 3 |
| Chain conveyance | No major upstream confluence within 5 km, excluding a VAA headwater reach |
| Major confluence vicinity | A junction within 5 km whose non-dominant incoming drainage-area share is at least 0.20 |
| Integrated mainstem | Strahler order at least 7 |
| Lake or reservoir path | Positive LakePond or Reservoir path length within 20 km upstream |

Also retain physical headwater flags, divergence, and catchment wetland cover.
Headwater is not inferred from absence of a sampled upstream station. A
StreamRiver waterbody polygon is not a lake or reservoir. Incoming drainage
area is a branch-size proxy; it is not observed tributary discharge. A chain
profile describes the 5 km neighbourhood and does not imply an unbranched basin.
The bands and distances above are chosen before DOC stratification; continuous
metrics remain available for later mechanism experiments.

## Observation and model panels

Source partitions 142, 143 and 144 provide permitted training observations for
DOC median, variability, seasonality and concentration-discharge associations.
Use only their train cells. Station seasonal summaries need at least 12 readings
and six represented calendar months; seasonal-adjusted log1p DOC versus log1p
discharge slopes need at least 24 paired months and full-rank design matrices.
These are observational associations, not identified process effects.

The nine saved current-availability runs provide source-validation K0 errors.
Compare the current complete model with strong station-hidden trees, the
retained complete model and its seasonal-source control. Average seed losses
first, then give each source partition equal weight. Bootstrap whole station
IDs jointly across partitions, 5,000 paired draws. Report cell-weighted and
station-weighted errors and group sizes; overlapping profiles are not additive.
This panel was used in development and does not constitute new confirmation.
Current-source attention is similarity retrieval, not physical river messaging.

A separate historical panel uses only the saved KGML K1 validation products
for upstream, both-direction and no-message residual arms. It describes genuine
river-message differences on those older tasks. Keep its temporal and spatial
tasks separate from the current source roles. Do not load its test products or
mix its estimand with the current-model comparison.

Upstream station-pair DOC associations use only source-training observations,
lags 0, 1, 3, 6 and 12 months, and at least 12 co-observations. Report all lags
and counts without selecting a best lag. Same-month deseasonalized association
is also reported. Monthly alignment cannot resolve event-scale travel times.

## Deliverables

Station structure table; monitored-edge path table; source DOC summaries;
source-validation model effects by structure; historical message effects;
upstream DOC association table; reproducible scientific figures; an English
research decision identifying the next river operator experiment.

No model fitting is part of this atlas. Completed geographic and external
predictions, old endpoints and manuscript results are preserved.

## Scientific references

- [Creed et al. 2015, river continuum DOC and C-Q patterns](https://www.usgs.gov/publications/river-a-chemostat-fresh-perspectives-dissolved-organic-matter-flowing-down-river).
- [McGuire et al. 2014, multiscale stream chemistry network controls](https://pmc.ncbi.nlm.nih.gov/articles/PMC4024884/).
- [Lynch et al. 2019, channel connectivity and DOM composition](https://www.nature.com/articles/s41467-019-08406-8). Its DOM-composition findings do not determine a universal DOC concentration effect.
- [USGS NHDPlus VAA definitions](https://www.usgs.gov/national-hydrography/value-added-attributes-vaas).
