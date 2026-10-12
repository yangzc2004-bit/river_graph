# Unified DOC model: spatial contribution and external replication

Completed 12 October 2026. Share with caveats: source-spatial evidence is supported, but this one external replication fails to establish a universal transfer model.

Source-validation-selected architecture: **environment MLP + GRU + upstream/downstream/global graph Transformer**. All arms belong to the same jointly trained model lineage; no local-anchor/forest correction branch was substituted.

## Matched results

Arithmetic mean of nine fit-level metrics (three existing whole-station splits x three optimization seeds); predictions are not ensembled. Native MAE/RMSE units are mg/L. Log-MAE uses log1p DOC. Bold marks the numerically best metric within each domain. All query predictions have 100% coverage.

### source_hidden_station

| Spatial setting | MAE | log-MAE | RMSE | R2 | Station-macro MAE |
|---|---:|---:|---:|---:|---:|
| Real upstream/downstream/global heads | **1.9313** | **0.2760** | **4.0633** | **0.4587** | **2.5164** |
| Generic dense spatial attention | 1.9646 | 0.2823 | 4.1257 | 0.4425 | 2.5189 |
| Self-only spatial path | 2.0819 | 0.3022 | 4.2521 | 0.4079 | 2.6817 |
| Permuted river topology | 2.0593 | 0.2984 | 4.2237 | 0.4168 | 2.6218 |

### external_river_K0

| Spatial setting | MAE | log-MAE | RMSE | R2 | Station-macro MAE |
|---|---:|---:|---:|---:|---:|
| Real upstream/downstream/global heads | 1.3505 | 0.4055 | 1.7888 | -1.0426 | 1.5211 |
| Generic dense spatial attention | 1.2162 | 0.3815 | 1.5596 | -0.5191 | 1.3849 |
| Self-only spatial path | **0.8956** | **0.3041** | **1.4084** | **-0.1820** | **0.7687** |
| Permuted river topology | 1.5083 | 0.4422 | 1.8544 | -1.4888 | 1.6249 |

## Architecture choice and spatial interpretation

Choice was sealed before either source-test evaluation or receiving-field prediction. Both primary validation ratios must improve and the average paired joint ratio must improve by at least1%; the lowest eligible real/generic spatial score wins, otherwise self-only remains. The permutation control is never eligible. This rule is new to this explicitly versioned spatial-inclusion experiment; prior paper endpoints were not changed.

Validation ratios versus self-only:

| Setting | Native ratio | Log ratio | Joint ratio |
|---|---:|---:|---:|
| Real upstream/downstream/global heads | 0.930533 | 0.911055 | 0.920794 |
| Generic dense spatial attention | 0.934582 | 0.925530 | 0.930056 |
| Permuted river topology | 0.947109 | 0.929949 | 0.938529 |
| Self-only spatial path | 1.000000 | 1.000000 | 1.000000 |

The self-only control keeps static river/environment covariates and every projection/feed-forward/residual self path. Its comparison isolates interstation messaging, not every conceivable spatial input. The generic arm has all-node attention and self/other biases; river-direction information is not encoded in attention masks. The permutation retains exact category counts and degree distributions but reassigns them across identities, with some true connected pairs possibly remaining; it is not a strict non-connected control.

Schema inspection confirms zero source/receiving COMID overlap. The source headwater flag is inherited from its original station-graph construction: 1 of357 flags differs from zero in-degree in the later repaired source graph. Source values were retained unchanged for model lineage and matched comparisons. Receiving flags use the same original construction formula on their own cohort. Water-temperature units are degC and discharge units are ft3/s.

The directional recipe wins the frozen source-validation ranking and has the lowest source-test means, but its paired native/log error intervals versus generic attention include zero. It is a source-domain candidate, not proven uniquely optimal among spatial architectures.

## Receiving-river interpretation

External HUC02040104 ([USGS basin identification](https://www.usgs.gov/data/delaware-river-basin-catchments-based-strahler-stream-order-1-national-hydrography-dataset-nhd)) contains 130 stations, 123 directed station edges and 6514 observed monthly DOC queries. Station overlap with source is zero. Temperature/discharge availability at receiving DOC queries is 96.52%/34.13%.

Across the entire receiving station-month grid, temperature/discharge availability is only 10.43%/8.44%. Query-cell covariates are therefore substantially more available than naturally unobserved grid-cell covariates. Output coverage does not establish accuracy in those information-poor cells; the evaluation does not establish reconstruction from universally available meteorological forcing alone.

The source-selected architecture changes receiving native/log MAE by -50.79%/-33.32% improvement versus self-only, with R2=-1.0426. This is descriptive evidence; the architecture is not reselected using receiving DOC. The receiving results do not provide joint positive evidence for the selected spatial arm; it must not be advertised as an established universal transfer model.

The receiving graph contains only receiving nodes; all preprocessing statistics and neural weights are source fitted, and no receiving DOC support, calibration or fine-tuning is used. All36 complete receiving fields were sealed before its DOC archive was opened. The same receiving case was previously inspected by other architecture releases. One network does not establish global river generalization, and nine source fits do not constitute nine independent external rivers.

## Prechosen single checkpoint

Primary checkpoint `split142_direction_heads_s42` was prechosen as split142 seed42 of the source-selected architecture. Its values differ from the nine-fit means:

| Domain | MAE | log-MAE | RMSE | R2 |
|---|---:|---:|---:|---:|
| source_hidden_station | 1.9399 | 0.2963 | 4.2887 | 0.4746 |
| external_river_K0 | 1.0909 | 0.3423 | 1.5064 | -0.3514 |

Primary receiving reconstruction is a complete 130 x 520 station-month grid saved in `release/primary_external_field_v2.npz` with a provenance sidecar. Scoring parquets contain observed queries only. Unobserved cells have no ground-truth validation. Monthly station reconstruction is not validated continuous-time or arbitrary-location reconstruction.

```bash
uv run python scripts/reconstruct_doc_unified_v2.py \
  --inputs experiments/phase4_transfer/doc_unified_transfer_v2/external/inputs.npz \
  --output path/to/new_reconstruction.npz
```

## Verification and uncertainty

Independent audit verified 36 checkpoints, 108 fit/field/test receipts and 72 source query roles. Maximum all-field/query replay error=0; maximum independently recomputed metric difference=4.44e-16. The primary public inference entry point also reproduces its audited field exactly.

Paired station-block bootstrap intervals (3000 draws) are exploratory within these fixed datasets; station correlation, overlapping source partitions and unadjusted multiple comparisons limit inference. Delta is selected model error minus comparator error:

| Domain | Comparator | Endpoint | Delta | 95% interval |
|---|---|---|---:|---|
| source | Generic dense spatial attention | mae | -0.03338 | [-0.11634, +0.04080] |
| source | Generic dense spatial attention | log1p_mae | -0.00630 | [-0.01609, +0.00389] |
| source | Self-only spatial path | mae | -0.15064 | [-0.27199, -0.06692] |
| source | Self-only spatial path | log1p_mae | -0.02625 | [-0.03987, -0.01494] |
| source | Permuted river topology | mae | -0.12807 | [-0.20507, -0.05469] |
| source | Permuted river topology | log1p_mae | -0.02245 | [-0.03275, -0.01219] |
| external | Generic dense spatial attention | mae | +0.13429 | [+0.08337, +0.20159] |
| external | Generic dense spatial attention | log1p_mae | +0.02396 | [+0.01224, +0.03514] |
| external | Self-only spatial path | mae | +0.45490 | [+0.29792, +0.65911] |
| external | Self-only spatial path | log1p_mae | +0.10133 | [+0.05609, +0.16125] |
| external | Permuted river topology | mae | -0.15782 | [-0.22844, -0.07942] |
| external | Permuted river topology | log1p_mae | -0.03678 | [-0.04895, -0.02313] |

Version1 failed at its first shared-scoring integration because its prediction table omitted high_doc; no completed validation selection or target score existed. Version2 fixes that schema only, retrains all36 fits, and preserves the failed frozen attempt. No tail-DOC endpoint is claimed.

Machine-readable evidence: protocol.json, manifest.json, selected_model.json, research_decision.json, analysis/summary.csv, analysis/by_fit.csv, analysis/independent_audit.json, analysis/paired_intervals.json, analysis/selected_model_intervals.json, and per-fit prediction/provenance receipts.
