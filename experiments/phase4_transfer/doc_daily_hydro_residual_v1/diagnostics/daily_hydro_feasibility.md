# Daily discharge is available for a focused DOC reconstruction experiment

The cached daily discharge supports adding within-month hydrologic shape to the existing model. Of 10,520 unique fixed spatial-query station-months across partitions 142–144, **8,141 (77.39%)** satisfy the frozen 80% day and consecutive-pair coverage requirements. This count uses query identities only; no query DOC values or comparative predictions were accessed. The new features preserve the existing monthly discharge footprint, allowing their values to be compared against an availability-only control.

## Coverage on the existing partitions

“All descriptors valid” requires at least `ceil(0.8 × D)` valid unique days for distribution width and `ceil(0.8 × (D−1))` actual consecutive within-month pairs for flashiness and rising fraction. Here `D` is the calendar month's length. Fixed queries exclude the same five reserved support cells at every K. Repeated partitions are shown separately and are not additional independent observations.

| Partition | Source observed cells: valid / total | Source-validation fixed queries: valid / total | Spatial fixed queries: valid / total |
| --- | ---: | ---: | ---: |
| 142 | 12,302 / 15,958 (77.09%) | 1,754 / 2,013 (87.13%) | 3,146 / 3,975 (79.14%) |
| 143 | 11,966 / 15,241 (78.51%) | 2,387 / 2,798 (85.31%) | 2,853 / 3,907 (73.02%) |
| 144 | 10,879 / 13,727 (79.25%) | 2,133 / 3,236 (65.91%) | 4,176 / 4,983 (83.80%) |

The unique spatial-query union contains 172 stations. Of its 8,141 fully usable months, 8,123 have complete daily calendars. The full station-month grid has lower coverage than the observed DOC query sample: the daily block is present in 120,541 of 233,478 cells. It contributes no new data where monthly discharge is absent.

## Frozen feature product

`../daily_features.npz` contains `full[357,654,8]` in the exact dataset station/month order. Its eight float32 channels are:

1. **Width:** `(Q90−Q10)/(mean(abs(q)) + Q90−Q10)`, using linear quantiles.
2. **Flashiness:** `sum(abs(q_current−q_previous))/sum(abs(q_current)+abs(q_previous))`, over true within-month consecutive-day pairs.
3. **Rising fraction:** the fraction of those pairs with `q_current > q_previous`.
4. Valid-day fraction, `n_days/D`.
5. Consecutive-pair fraction, `n_pairs/(D−1)`.
6. Width validity.
7. Flashiness validity.
8. Rising-fraction validity.

Missing descriptors are zero with validity zero. Availability fractions remain informative when the 80% threshold is not reached. Sufficiently observed all-zero discharge has ratios zero and validity one. Signed discharge is preserved. Every channel is zero wherever frozen `x_mask[:,:,1]` is zero. No fitted scale or DOC value is used. Positive flow-unit rescaling leaves the descriptors unchanged when both inputs satisfy the raw-unit QC bound.

The product contains **120,017 width-valid** and **119,989 flashiness/rising-valid** months. The availability-only control zeros channels 0–2 and retains channels 3–7. These are current-month reconstruction covariates: a full current-month summary is not an input available before that month ends.

## Raw-data quality and reconciliation

- **Source:** 78 NWIS RDB cache files, 221,453,996 bytes, containing parameter 00060/statistic 00003 daily-mean discharge in cubic feet per second. Raw calendars span 1900-01-01 to 2026-09-20. The final September 2026 calendar is partial.
- **Finite-value QC:** retain finite signed/zero discharge with absolute value at most 3,000,000 cfs, matching the frozen monthly pipeline. Of 6,890,854 first-series rows, 13,675 are nonnumeric/nonfinite; no finite row exceeds the bound, and no date is invalid.
- **Station-day grain:** equal station-day duplicates collapse before feature calculation; accepted conflicts would exclude the entire day. The full cache has 1,250,907 duplicate excess rows; the ST357 cohort has 1,246,482. **Neither has conflicting accepted first-series values.** Rows are never counted twice in quantiles or pair statistics.
- **Study-grid coverage:** 3,659,912 unique valid daily observations occur within the 654-month grid at 301 stations. Across all cached years, 304 cohort stations have valid daily discharge. The new daily-availability mask matches all 120,541 frozen monthly discharge cells exactly.
- **Monthly consistency:** reaggregating deduplicated daily data reproduces every visible frozen monthly mean within `rtol=1e-6, atol=0.001`. The largest absolute difference, 0.03024 cfs at large flow, is within this float32 tolerance. No monthly observation or DOC label was changed.
- **Signed flow:** the study grid contains 123 negative-flow days at two stations and 54 consecutive-day sign reversals. They remain valid physical measurements under the existing policy. No fixed spatial-query month contains a negative daily value.
- **Qualifiers:** the grid contains 353,608 unique estimated days and 35,625 provisional days. These are retained and counted, consistent with the frozen finite-value QC; qualifier tokens use colon delimiters such as `A:e`. The raw-file inventory preserves qualifier counts. Estimated records occur in 1,538 unique fixed spatial-query months and provisional records in 58.

## Multiple-series policy

The loader explicitly selects the **first** header column ending `_00060_00003`, as the existing monthly dataset does. Three headers contain a second discharge series:

- Station **03216600** appears in two cache files. Its first series ends on 2014-10-30; the second continues to 2026-09-19 and supplies 4,308 additional days. The 16,831 overlapping finite days agree exactly. The continuation is documented but is not added in this experiment, so daily descriptors do not silently expand the old hydro footprint.
- Station **06208500** has a second series explicitly adjusted for White Horse Canal. It differs on 1,701 of 32,188 common finite days, by up to 100 cfs. It is a distinct measurement definition, not a duplicate to average into the first series.

The product metadata records each alternative header and ignored series name. Alternative-series differences and date coverage are preserved in `multiple_series_inventory.csv` and `multiple_series_differences.csv`.

## Interpretation and reproducibility

This audit establishes available hydrologic information, not an association with held-out DOC errors. The feature comparison can test whether within-month discharge variability and rise/fall structure add useful information beyond the existing monthly hydro inputs and their availability. DOC remains a sparse station-month target, often based on isolated grab samples; daily flow coverage does not create event-resolved DOC labels.

Rebuild with `uv run python scripts/build_doc_daily_flow_features_v1.py`. Identical inputs reuse the verified product; changed inputs or code require a new output version. The metadata binds the dataset, all raw files, builder/module, feature definitions and product contents. `raw_cache_inventory.csv`, `role_coverage.csv`, `station_role_coverage.csv`, `station_inventory.csv`, `availability_summary.json` and `sources.json` retain the coverage audit. The builder and 12 focused tests cover calendar thresholds, gaps, leap months, duplicates, signed/zero discharge, first-series selection, target independence, future-month isolation, ordering and the availability ablation.
