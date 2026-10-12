# Aqua-OC reproduction: acquisition status and comparison requirements

The requested next experiment is a complete DOC retrieval framework comparison,
not another single-regressor adapter. Data acquisition and source audits have
started. No Aqua-OC model has been trained, and no new comparative accuracy
result is available. The original frozen results and primary endpoints remain
unchanged.

## Authoritative sources and implementation scope

The [2025 Aqua-OC paper](https://doi.org/10.1016/j.isprsjprs.2025.01.028)
and its supplement specify Landsat optical inputs, ERA5-Land environmental
predictors, optical water classification and class-specific regression models.
The cached main paper and supplement are in
`cache/doc_domain_work_v1/sources/`. A complete framework implementation must
retain all of those components.

The [authors' 2026 follow-up repository](https://github.com/qqq196988/Global-river-POC/)
was checked out at `bbbad668465fca267b4df63b28791c0ba8a74581`.
It provides downstream POC analysis, rather than a complete DOC training
implementation. Its README describes a Landsat processing notebook, but that
notebook is absent from the inspected checkout. No original DOC weights,
numeric optical centroids, original sample assignments or exact optimized
hyperparameter configurations have been established from these inspected
sources. This statement concerns the inspected sources; it does not establish
that no other implementation exists.

The [original AquaSat repository](https://github.com/GlobalHydrologyLab/AquaSat)
was checked out at `f2ce021f9a6898cbd5509f44132433975955071a`. Its reflectance
matchup products need rechecking against Aqua-OC's later sensor and quality
requirements; an AquaSat record is not automatically an accepted Aqua-OC record.

## Inputs acquired and audited

| Resource | Verified status | Consequence |
| --- | --- | --- |
| Local DOC samples | 32,259 individual records at all 357 ST357 stations; 30,760 distinct station dates; dates 1972-04-19 through 2026-09-01. | Original daily sampling dates are available for satellite matching. |
| Local paper-era samples | 24,579 records at 291 stations during 1984–2020. | This is a candidate pool before satellite quality filtering, not a final matchup count. |
| Monthly reconciliation | All 22,571 observed cells in the canonical dataset have corresponding raw samples. Largest absolute concentration difference is 0.000003052 mg/L; none exceeds 0.0001 mg/L. | Current raw data reproduce the frozen monthly targets within float32 rounding. Multiple same-date observations retain their original sample weighting. |
| GRQA DOC file | Downloaded 175,018,838 bytes; published MD5 verified. The retrieved file contains 413,328 records at 14,799 station identifiers, all labelled mg/l. | Global source observations are available; these are not the original authors' filtered training matchups. |
| GRQA quality flags | 8,576 records have detection-limit flags; 28,197 have IQR-outlier flags. No duplicate observation IDs, negative values, missing coordinates or invalid dates were found in the selected fields. | Censoring and outlier policy must be specified before training. Neither the dataset count nor its flags justify silently removing records. |
| Landsat access | Public Collection-2 Level-2 catalog queried; a 3×3 blue-band window was retrieved for a Landsat-7 scene at station 07022000 on 2000-05-03, matching a sampling date. | Public pixel retrieval works without GEE. This one-scene access test does not establish water-mask, cloud, CV or full-cohort coverage. |
| AquaSat final product | The [public release](https://figshare.com/articles/dataset/AquaSat/8139383) is identified. Its metadata, page and direct download endpoints returned HTTP 403 in this environment. | The file has not been acquired. A working download route or supplied original file is needed. |
| ERA5-Land | CDS ARCO and Earth Data Hub data endpoints returned HTTP 401; no Earth Engine, application-default Google or CDS credential file is present. | Authorized data access is required to acquire the seven daily environmental predictors. |

GRQA file SHA256:
`4f9fa86bdd62817ffd4a899403ef033a674fe466bf1eb46410fcd8be549c35bf`.
Its [release metadata](https://zenodo.org/records/5101057) and source URLs are
preserved with the download receipts. The retrieved record count differs from
the count cited for GRQA in Aqua-OC. The original authors' release selection and
filtering are not resolved, so the native training cohort cannot be declared
reproduced merely because a public GRQA file was downloaded.

## Direct comparison must control overlap, coverage and target resolution

Exact station identifiers match 281 of our 357 stations in GRQA, involving
15,580 GRQA records. Among the five existing target tasks, 74 of 104 target
stations occur in GRQA, involving 4,044 records. Training on the unrestricted
global file would expose target DOC measurements. Exclusions must cover entire
target and validation basins, including coordinate aliases and other stations
in those basins. Exact-ID removal alone does not establish geographical
separation.

| Target HUC4 | Existing target station-month cells | Cells during 1984–2020 | Target station IDs found in GRQA |
| --- | ---: | ---: | ---: |
| 1013 | 1,204 | 88 | 27/27 |
| 1019 | 2,166 | 1,885 | 14/21 |
| 0708 | 1,663 | 1,452 | 20/20 |
| 1030 | 1,001 | 953 | 3/19 |
| 1101 | 1,228 | 1,145 | 10/17 |
| Total | 7,262 | 5,523 | 74/104 |

The 5,523 cells are a temporal overlap ceiling for a comparison restricted to
the paper's training period, before optical quality and environmental coverage
are assessed. They are not a measured Aqua-OC prediction coverage. A deliberate
extension beyond that period would be a separate version and would need sensor
availability and extrapolation assumptions recorded.

The project predicts monthly means of observed DOC samples. Aqua-OC's training
matchups are daily observations near satellite overpasses. These targets must
not be treated as interchangeable. A common monthly evaluation must retain the
same monthly truth and weighting for both methods, and document how the
satellite retrieval is converted to that estimand. A separate daily diagnostic
can assess satellite matchups, with the monthly fusion prediction explicitly
identified as a monthly estimate rather than a native daily prediction.

The proposed experiment should therefore have two separately labelled tracks:

1. **Published-method reproduction.** Rebuild the native optical matchups and
   environmental inputs, water typing and regression-family selection. Record
   all choices the publication leaves unspecified. Original random-split
   metrics are reproduction diagnostics, not the direct comparison score.
2. **Geographical comparison.** Use the existing whole-basin holdouts with all
   target and validation DOC excluded from fitting, scaling, clustering and
   model selection. Score both methods on identical observed target cells with
   valid Aqua-OC inputs. Report paired-subset MAE, log1p MAE, RMSE and predictive
   R² alongside coverage. Report the fusion model's full-cohort performance
   separately. If training information or sample counts differ, state that
   difference and add a controlled-training comparison before attributing a
   gain to architecture.

The previous aggregate scores must not be reused as the fusion score on the
new optical subset. A global pretrained Aqua-OC model also cannot establish
unseen-basin performance if its original training-data overlap is unknown.

## Unresolved implementation details

The supplement specifies a DNN architecture and tuning ranges, but does not
establish the exact fitted configurations inspected here. The publication
also describes cosine-distance clustering while naming scikit-learn KMeans,
whose native fitting objective is Euclidean. Spectral preprocessing, cluster
centroids and cluster-number mapping need clarification or a transparently
declared reimplementation choice. Reordered newly fitted cluster IDs must not
be assigned the paper's expert labels by index alone.

Original sample split seeds, optical feature coefficients and detailed quality
filter choices also need to be verified. Published-method reimplementation can
be useful with declared choices; exact numerical reproduction requires the
original cohort, assignments and fitted details or original source code.

ERA5-Land's [official daily-aggregate documentation](https://developers.google.com/earth-engine/datasets/catalog/ECMWF_ERA5_LAND_DAILY_AGGR)
documents swapped evaporation-component values and differences between daily
flow sums and hourly averages. Preserve variable identities, units and
aggregation conventions, and resolve the open-water evaporation mapping before
fitting. Substituting ordinary ERA5, annual StreamCat values or a different
evaporation variable would not complete the published input set.

## Reproducible preparation and pending dependencies

```bash
uv run python scripts/prepare_aqua_oc_data_v1.py metadata
uv run python scripts/prepare_aqua_oc_data_v1.py download
uv run python scripts/prepare_aqua_oc_data_v1.py local
uv run python scripts/probe_aqua_oc_sources_v1.py
uv run python scripts/probe_aqua_oc_archives_v1.py
uv run python scripts/audit_aqua_oc_data_v1.py
```

Raw downloads and source checkouts live under `cache/aqua_oc_reproduction_v1/`.
Receipts and audit tables live under
`experiments/phase4_transfer/aqua_oc_reproduction_v1/`. The scripts do not start
training or overwrite frozen prediction artifacts. The author checkouts are
read-only evidence; no author code has been executed.

The immediate missing access is a working Earth Engine project or CDS account
configured locally, plus access to the AquaSat product. Credentials should not
be pasted into chat. Separately, the original DOC cohort, centroids and fitted
details would resolve exact-reproduction uncertainty. Requesting those from
authors would require explicit authorization to contact them; no message has
been sent. Once the actual input coverage is known, create and freeze a new
comparison protocol and isolated training runtime before scoring targets.
