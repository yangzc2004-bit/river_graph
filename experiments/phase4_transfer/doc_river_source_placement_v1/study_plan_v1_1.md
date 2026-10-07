# DOC source placement within real river networks

## Scientific purpose
Determine whether the locations of potential terrestrial DOC sources help
explain what whole-network shape alone leaves unresolved. The working pathway
is network morphology -> source connection / routing exposure -> DOC response.
The study separates landscape amount from its organization along real channels.

## Work in this version
1. Review the 15 gross station/drainage discrepancies identified in
   `doc_river_mechanisms_v1`, using independent USGS station references,
   hydrolocation, official coordinates and drainage areas. Save proposed
   alternatives and unresolved cases without changing the historical graph.
2. Acquire EPA StreamCat NLCD 2019 percentages for individual NHDPlusV2
   catchments and 100-m riparian catchments, with completeness and areas.
   Watershed aggregates alone cannot answer source-placement questions.
3. Calculate shortest upstream **channel** distance from each reach midpoint
   to the receiving reach outlet, allowing primary and secondary NHD links.
   Represent wetland/forest amount with catchment area times land-cover fraction;
   this is potential source landscape, not measured DOC production or loading.
4. Quantify source-weighted distance, proximity relative to general drainage,
   share within 50 km, and 100-m riparian enrichment. Summaries retain missing
   coverage explicitly and do not treat missing land cover as zero.
5. Compare environmental controls, environmental controls + continuous shape,
   controls + source placement, and controls + shape + placement in the same
   five HUC4-blocked Ridge diagnostic (alpha=10). Preprocessing fits each source
   training fold. The response is station median of permitted monthly DOC means,
   not monthly full-grid or model K0 performance.

## Population and analysis decisions
- The response population remains the deduplicated source-training union
  142/143/144; no geographic or external model outcomes are read.
- Primary cohort excludes the 15 gross mapped/reported-area discrepancies.
  Missing official drainage is retained and explicitly marked unverified.
- Require >=95% represented positive drainage area, >=95% land-cover completeness
  in retained catchments, and >=12 source DOC months spanning >=6 calendar months
  for the station diagnostic. Keep all excluded sites and reasons.
- Fixed source-placement model features: wetland/forest distance ratios,
  near-50-km share minus the near-50-km drainage share, and riparian enrichment
  (six variables). A zero source amount makes its placement undefined; fit-fold
  imputation handles this and source absence is reported separately.
- Controls, continuous shape and native/log1p error definitions follow the
  earlier planform diagnostic. Wetland/forest amount is recomputed on exactly
  the represented catchments used for placement. Every model is evaluated on
  the same stations.
- Report equal-fold paired gains; 5,000 paired station and HUC4 bootstrap draws,
  seed 42. Keep all comparisons. Also describe class differences and adjusted
  source-placement associations, including all coefficients.
- NLCD 2019 is a static landscape descriptor. Repeat the response diagnostic
  and adjusted associations using source months from 2009 onward as a temporal-alignment sensitivity;
  it is not annual historical land-cover reconstruction.
- Source distance is channel distance from a catchment-associated reach,
  not actual hillslope distance, travel time or a causal mediation estimate.
- Source placement and riparian contrast can justify a later upstream model
  operator; this version develops the evidence without retraining it.

## Research products
Official location review, catchment acquisition inventory, station source
placement, coverage exclusions, real-network source maps, blocked diagnostic
comparisons, exploratory associations and an English research decision.
Preserve all earlier experiments and any acquisition failures for inspection.

## Documented reconciliation after the first diagnostic
The first comparisons used the pre-existing whole-watershed wetland/forest
controls. An independent catchment-versus-watershed reconciliation found near
agreement at most stations but large differences at three previously noted
Kansas catchments (06887000/06889000/06892350). The wetland discrepancy at
06887000 is 1.72 percentage points and forest discrepancy 6.19 points.
After seeing the first diagnostic, all four arms were recomputed with amount
and placement from the same catchment population. Alpha, folds, source features,
outcomes and inclusion rules were unchanged. Original results, code and plan
remain in `initial_watershed_controls/`. This corrects the scientific amount-
versus-position comparison; neither set is a monthly neural-model test.
The original `study_plan.md` stays intact; this revision governs the reconciled
amount/position comparisons and complete recent-period sensitivity tables.
