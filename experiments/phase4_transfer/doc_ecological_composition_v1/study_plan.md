# Detailed ecological composition for water-quality-free DOC reconstruction

## Question

Does retaining distinct forest, wetland, farming and urban land-cover classes
provide useful new-site DOC information beyond the current broad totals?
The retained dataset collapses eleven cached StreamCat categories into four
land-cover aggregates. The upstream catchment percentages already exist locally;
no target chemistry or new data acquisition is needed.

Use all predefined categories: coniferous/deciduous/mixed forest, crops/hay,
high/medium/low/open urban development, woody/herbaceous wetlands. Normalize
percentages by100 without fitting a scaler, and retain eleven validity flags.
Real zero remains valid; missing/invalid percentages become zero with validity
zero. Align by graph-node site identity and COMID, not node position or HUC.
Check the four reconstructed totals against frozen dataset regime values before
fitting. These are2019 static ecological proxies, just as in the retained model;
they do not reconstruct historical land cover.

## First source-only comparison

Use partitions142/143/144 × seeds42/43/44. All receiving-site DOC, pH,
conductance and derived history/availability remain hidden. Retain the existing
station-fold-hidden tree features and selected300-tree hyperparameters.
Append22 physical composition/validity channels to both training and prediction.

Compare:

1. Retained strong47-feature environmental tree and complete neural procedure.
2. Matched69-feature tree: each added class position receives its broad family
   total, with the same eleven validity flags.
3. Detailed69-feature tree: each added class position receives its own fraction.

The matched expanded control separates detailed ecological information from
repeating more columns or reweighting broad environmental predictors. Select no
land-cover category using DOC results. Fit only source labels; validation scores
describe development. Preserve all parent predictions and the deployed release.

Report source MAE/RMSE, signed bias, Q90 and partition/seed directions with
5,000 paired station draws. If detailed inputs add useful information, rebuild
the retained native ecological/GRU residual on a consistently cross-fitted new
environmental reference, then compare a base-only change with a detailed
neural-readout extension. Do not add the unconfirmed nonlinear head or failed
regional/calibration mechanisms. Further geographical or external scoring uses
a subsequently fixed version; old queries do not select ecological categories.
