# How river landscapes modify the monthly DOC response to flow

## Main finding

The clearest dynamic lead is **wetland amount versus riparian organization**.
More wetland cover is associated with a stronger positive monthly DOC-flow
response; greater wetland enrichment along riparian corridors is associated
with a weaker response at the same source amount and environmental/shape
controls. Both directions persist after controlling observed temperature.
The tested source-distance modifiers and adjusted differences among the three
planform classes are not established. The geographic response-descriptor
diagnostic also does not show a reliable gain from adding placement or shape.

This moves the research question toward **when source landscapes connect to
the stream**, while keeping river routing and processing as separate candidate
explanations. It does not establish that high flow activates distant forest
sources, and it does not improve or replace the current DOC neural model.

## Observations and analysis

The monthly panel has 21,459 unique DOC cells from the source-training union of
splits 142/143/144. It uses the existing ST357 monthly data and the reconciled
source-placement panel. Landscape/mapping exclusions remain as in the previous
study; no geographic or external model target result is read.

| Population | Eligible stations | DOC-flow month pairs | HUC4 groups |
|---|---:|---:|---:|
| All permitted source months | 210 | 13,890 | 48 |
| Observed-temperature adjustment | 209 | 13,571 | 48 |
| Permitted source months since 2009 | 49 | 4,939 | 22 |

Stations require 24 usable months, six calendar months, two years and
sufficient log1p-flow variation. The monthly panel keeps all exclusions.
Discharge is the existing monthly aggregate of NWIS daily means in cfs;
temperature is a monthly aggregate of observations. Negative flows can denote
real reversal. Two permitted cells are excluded from this nonnegative C-Q
analysis, not deleted or labelled erroneous in the historical dataset.

At each station, log1p DOC and the linear/quadratic centered log1p-flow terms
are residualized against that station's intercept, annual sine/cosine and
linear year trend. The temperature sensitivity adds observed temperature.
Thus the response concerns changes **within a station**, rather than differences
in average DOC between stations. A pooled model gives each station equal total
weight and allows the linear flow response to vary with environment, source
amount, continuous shape and the six fixed position descriptors. It retains
one common quadratic flow term. Complete terms, identification diagnostics and
5,000 HUC4-bootstrap intervals are saved.

The local C-Q slope is d log(1+DOC) / d log(1+Q) at the station's median log
flow. It is not a transport velocity or a DOC load coefficient. The fitted
interquartile contrast compares flow Q25 and Q75 under common nuisance
conditions, in log1p DOC units. Monthly data do not resolve event hysteresis.

## Three planforms

| Fixed morphology class | Stations | Positive fitted Q25-to-Q75 response | Median log1p-DOC contrast, HUC4 95% interval |
|---|---:|---:|---|
| Elongated tributary-rich | 76 | 67.1% | 0.0619 [0.0142, 0.1111] |
| Mainstem dominated sparse | 20 | 70.0% | 0.0372 [0.0241, 0.1068] |
| Broad tributary-rich | 114 | 76.3% | 0.0821 [0.0562, 0.1312] |

Overall, 152 of 210 stations have a positive fitted interquartile flow
contrast. This broad positive direction is compatible with mobilization,
while negative-response stations remain visible. It does not separate the
three classes into distinct process regimes.

At a common mean environment and region composition, adjusted linear slopes
are 0.0349 [0.0094, 0.0681], 0.0704 [-0.0388, 0.1813] and
0.0373 [0.0135, 0.0677]. Direct paired class differences are:

- Mainstem sparse minus elongated: 0.0355 [-0.0843, 0.1401].
- Broad minus elongated: 0.0025 [-0.0296, 0.0343].
- Broad minus mainstem sparse: -0.0331 [-0.1501, 0.0879].

All direct intervals span zero. The sparse class has only 20 stations; the
recent-period version has eight. Do not use differences between point estimates
or marginal medians to claim a verified class effect.

## Source organization and flow

Coefficients below modify the local C-Q slope per population-specific station
SD of the landscape feature, with HUC4 95% intervals. They jointly control
source amount, environment and continuous shape. Intervals are exploratory,
without a multiple-comparison correction.

| Modifier | All source months | Temperature adjusted | Since 2009 |
|---|---|---|---|
| Wetland amount | +0.0686 [0.0211, 0.1089] | +0.0694 [0.0180, 0.1096] | +0.0332 [-0.3075, 0.3501] |
| Wetland riparian enrichment | -0.0614 [-0.0912, -0.0193] | -0.0535 [-0.0846, -0.0101] | -0.0167 [-0.4475, 0.5310] |
| Wetland source-distance ratio | +0.0145 [-0.0144, 0.0345] | +0.0147 [-0.0118, 0.0356] | +0.0224 [-0.1335, 0.2098] |
| Forest source-distance ratio | +0.0178 [-0.0156, 0.0542] | +0.0163 [-0.0147, 0.0500] | +0.0443 [-0.1935, 0.2663] |

Wetland amount and riparian organization have opposite signs in both full
and temperature-adjusted analyses. A useful hypothesis is that sources close
to an active riparian corridor can contribute under lower flow, whereas less
continuously connected landscape requires a larger hydrologic change. That
hypothesis still needs low-flow concentration and matched-site evidence;
the present slope analysis alone cannot identify continuous supply versus
dilution or processing.

Farther-upstream source positions have positive point estimates, but all their
intervals span zero. Consequently, the proposed distant-source activation
mechanism is not established by this version. Forest riparian enrichment is
also not an established flow-response modifier.

For continuous shape, mainstem sinuosity has a negative modifier in the
temperature-adjusted analysis (-0.0261 [-0.0485, -0.0072]); its all-period
interval narrowly spans zero, and the recent-period direction changes.
This is a processing/routing lead rather than a reproducible class-wide effect.

The recent population is much smaller, with only 49 stations and 22 HUC4
groups. Its full moderation model retains 30 identified columns and has poorer
conditioning. It is an imprecise sensitivity, not a successful independent
replication. Different sampling composition also prevents attributing changes
solely to temporal alignment with NLCD 2019.

## Geographic response-descriptor diagnostic

Five HUC4-blocked Ridge folds predict the measured station linear C-Q descriptor,
with preprocessing fit inside each fold. This is separate from the monthly
pooled explanatory model and from unmonitored-station DOC reconstruction.

For the main population, adding shape to environment changes descriptor MAE
from 0.1288 to 0.1327: -3.02% gain [-7.13, 0.01]. Adding placement gives
0.1341: -4.10% [-11.75, 1.55]. Adding placement to the shape arm gives
-3.33% [-10.78, 2.29]. Temperature-adjusted results likewise do not establish
placement gains. The recent shape-only improvement is +3.24% [-7.73, 10.49],
with an interval spanning zero. Complete comparisons are retained.

These results distinguish an adjusted association from transferable response
prediction. They do not justify adding the six static placement features to
the production GNN as a demonstrated performance upgrade.

## Next research direction

1. **Discriminate low-flow supply from high-flow activation.** Compare adjusted
   low- and high-flow DOC levels, controlling total wetland amount, region and
   season. Test whether riparian enrichment corresponds to elevated low-flow
   DOC rather than simply lower high-flow DOC. Keep negative-response examples.
2. **Use physically checked networks.** Find matched real catchments with
   similar area and wetland amount but different riparian organization and
   routing exposure. Check station locations and contemporaneous upstream/
   downstream observations before interpreting a transport mechanism.
3. **Separate lateral connection and channel processing.** Evaluate riparian
   source connection against sinuosity/storage/temperature alternatives.
   Daily hydrology can clarify antecedent wetness and hydrograph position;
   event claims require DOC observations at matching resolution.
4. **Translate repeatable effects into the existing graph branch.** The likely
   useful operator would distinguish local/riparian input from routed upstream
   input and condition their weights on hydro state. Evaluate against explicit
   feature controls and no-message before assigning graph-specific benefit.

The revised scientific emphasis is: a river network organizes both the location
and the changing connection of DOC sources. Shape classes are a useful map of
that organization; they do not yet identify distinct DOC response mechanisms.

## Relation to literature

[Wen et al. (2020)](https://hess.copernicus.org/articles/24/945/2020/)
show that production and export can respond differently to temperature and
hydrologic connectivity, with both flushing and dilution.
[Lupon et al. (2023)](https://hess.copernicus.org/articles/27/613/2023/)
show the importance of spatially organized inflow paths under changing
hydrologic conditions. [Prijac et al. (2023)](https://hess.copernicus.org/articles/27/3935/2023/)
demonstrate event-scale peatland-stream connections. These studies motivate
the alternatives above; this monthly multi-region analysis is not a replication
of their site-specific process measurements.
