# Independent laboratory-DOC validation of whole river-network form

Date: 2026-10-09. Written before fitting DOC–form associations in this version.

## Research question

Do actual upstream footprint and branching organization explain differences in
DOC level or variability across new rivers, after accounting for basin size,
hydroclimate, land cover and sampling calendar? The earlier ST357 results are
already known: branching has predictive information, while the direct broad
versus elongated contrast is unresolved. This is a new observational replication,
not an additional model-training round.

## Population and geometry

Start with all 27 stream/river sites in the NEON-authored Zenodo archive 20039155.
Use laboratory DOC represented in that absorbance archive, retaining its selection
and unavailable original quality flags. Average numeric-suffix field replicates
within an exact sample occasion; relocated RE samples remain excluded because
their actual coordinates are unavailable. Lakes are outside this question.

Use the public NEON location tree to identify the registered S2/buoy routine
sampling location, including historical sensor locations. A site-center point is
never a verified outlet. Resolve registered points to NHDPlusV2 COMIDs using the
USGS NLDI. If historical/current location alternatives reach different COMIDs,
retain only occasions with an unambiguous active-location period; unresolved
occasions are inventoried. Registered routine locations are not individually
verified sample coordinates. The routine sample may be just downstream of S2;
the analytical unit is explicitly the receiving reach outlet.

Extract complete upstream channels and the unsimplified basin, including primary
and secondary downstream links. Reuse earlier geometry read-only and store new
downloads separately. Require full expected reach coverage, at least five mapped
reaches and a basin/unique-catchment-area ratio within 0.8–1.2. Document missing
NHD coverage, inaccessible geometry and small headwater networks without selecting
on DOC. Keep the existing continuous descriptors and transfer the original
geometry-only standardization/centroids for descriptive classes; do not recluster
using new DOC. Assignment to old centroids is an approximation to the old Ward
partition, not a fresh independent discovery of three universal classes.

Record pairwise overlapping/nested upstream sets and overlap with ST357. Keep
whole connected/overlapping sets together for validation and resampling. Report
the all-NEON cohort and a sensitivity excluding upstream overlap with ST357.

## DOC and calendar

Primary window: 2018-01 through 2025-12, fixed before outcome fitting. Aggregate
occasions into site-month medians. Include sites with at least 36 observed months,
all 12 calendar months represented and at least six distinct years. Compute one
median per calendar month across years, then take their median for season-balanced
DOC level. Primary variability is IQR/median of the observed monthly medians;
CV and normalized seasonal amplitude are diagnostics. A 2020–2024 common-window
sensitivity uses at least 24 months, all seasons and four years. Do not count
replicates or months as additional independent river forms.

## Small, fixed comparisons

Controls: log basin area, watershed wetland/forest/agriculture/urban cover,
climate precipitation/temperature, latitude/longitude and observed-month count.
EPA StreamCat metrics have explicit full-watershed coverage checks; missing
coverage is not zero vegetation. Hydroclimate normals are context controls;
coincident discharge is unavailable, so no event-routing attribution is possible.

Compare fixed Ridge(alpha=10) models, with preprocessing fit inside each holdout:
context alone; context + footprint (log basin aspect and network axis ratio);
context + branching (log drainage density and mainstem share); context + path
organization (mainstem sinuosity, scaled mean path length, distance CV); and all
three blocks. Predict log1p DOC level and log1p variability separately. Leave
whole independent upstream-overlap groups out. Report site-weighted held-out MAE,
paired gains, 5,000 group-bootstrap intervals, group directions and leave-one-group
influence. A domain-blocked sensitivity tests broader spatial separation when
there are at least five domains. All fits and failures remain in the report.

Complement the prediction comparison with broad/elongated site pairs selected
without DOC: basin-area ratio <=2, precipitation ratio <=1.25, temperature difference
<=3 C, wetland difference <=5 percentage points, forest difference <=20 points,
and context RMS distance <=1. Pairs may not share upstream channels. Report empty
or tiny matched sets directly rather than relax the matching to obtain a result.

## Completion

Finish with acquisition/eligibility ledgers, actual network maps, DOC endpoints,
all declared comparisons, sensitivity results and a scientific decision. If the
new sites cannot identify a shape contrast, close that question for these data
and state why. A repeatable continuous branching contribution is a useful result
even without a universal ordering of long versus broad rivers. No new neural
training, simulation, endpoint replacement or historical-file overwrite is included.
