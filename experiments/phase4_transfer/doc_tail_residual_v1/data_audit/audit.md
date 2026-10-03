# Source audit of large DOC reconstruction errors

The audit uses only v4 **source-validation** queries for the K=5 updated-GRU adapter. The top 20 unique station-months are ranked by mean absolute error, averaging seeds within partition and available partitions within each cell. These are previously reused selection labels. No outer-test prediction table or outer-query label is inspected; no label is altered.

## Findings

- 20/20 months have verified source DOC records; 0 have ambiguous provenance/QC; 0 have a directly established label defect.
- All 14 examined raw files match their SHA-256 values in the cached370 build-input manifest. The ST357 cohort is its documented stream-station subset.
- Monthly arithmetic means reproduce the frozen float32 labels for 20/20 months. 19/20 are based on one retained point sample.
- These 20 cells contribute 15.4% of absolute error and 89.8% of squared error in the unique validation-cell population.

`verified_source_doc` means the high value is genuinely present as a DOC measurement in the unchanged provider file and is correctly reproduced by the pipeline. It does **not** establish the physical accuracy of a decades-old laboratory measurement. Historical status alone is not a rejection flag. No arbitrary DOC cutoff or decimal-point repair is applied.

## Trace of the 20 largest errors

| Station | Month | Dataset DOC | Raw dates | Raw values (mg/L) | n | Classification |
|---|---|---:|---|---|---:|---|
| 06438000 | 1980-07 | 460 | 1980-07-10 | 460.0 | 1 | verified_source_doc |
| 06438000 | 1980-10 | 450 | 1980-10-16 | 450.0 | 1 | verified_source_doc |
| 06438000 | 1978-08 | 212 | 1978-08-02 | 212.0 | 1 | verified_source_doc |
| 06438000 | 1979-07 | 160 | 1979-07-17 | 160.0 | 1 | verified_source_doc |
| 06339100 | 1980-06 | 110 | 1980-06-03 | 110.0 | 1 | verified_source_doc |
| 06364700 | 1980-08 | 76 | 1980-08-13 | 76.0 | 1 | verified_source_doc |
| 06438000 | 1980-01 | 64 | 1980-01-14 | 64.0 | 1 | verified_source_doc |
| 06805500 | 1993-06 | 51 | 1993-06-29 | 51.0 | 1 | verified_source_doc |
| 05451210 | 2020-12 | 45.3 | 2020-12-14 | 45.3 | 1 | verified_source_doc |
| 03289500 | 1991-02 | 44 | 1991-02-12 | 44.0 | 1 | verified_source_doc |
| 06753990 | 2001-05 | 51.1 | 2001-05-07 | 51.1 | 1 | verified_source_doc |
| 05464420 | 2012-09 | 41 | 2012-09-27 | 41.0 | 1 | verified_source_doc |
| 06438000 | 1980-04 | 53 | 1980-04-15 | 53.0 | 1 | verified_source_doc |
| 06880800 | 1987-12 | 37 | 1987-12-01 | 37.0 | 1 | verified_source_doc |
| 06306300 | 1981-02 | 36 | 1981-02-18 | 36.0 | 1 | verified_source_doc |
| 06330000 | 1975-09 | 37 | 1975-09-09 | 37.0 | 1 | verified_source_doc |
| 06295000 | 1981-01 | 31 | 1981-01-20 | 31.0 | 1 | verified_source_doc |
| 03460000 | 1974-06 | 28 | 1974-06-14 | 28.0 | 1 | verified_source_doc |
| 06339100 | 1980-08 | 43 | 1980-08-13 | 43.0 | 1 | verified_source_doc |
| 05531500 | 1999-12 | 32.25 | 1999-12-08; 1999-12-20 | 57.6; 6.9 | 2 | verified_source_doc |

## What was checked

The local WQP narrow-profile files identify the constituent as Organic carbon, USGS parameter 00681, with the filtered fraction and mg/L units. The DOC extractor admits 00681 or, when the parameter code is absent, an explicit dissolved/filtered fraction; it requires mg/L and no detection condition. `to_monthly` uses an arithmetic mean of retained point observations, not a time-weighted monthly concentration. No DOC value-range filter is applied. Suspended-carbon observations in the same months are recorded as excluded candidates rather than mixed into DOC.

Accepted records were checked for provider result/activity identifiers, sample date, fraction, units, detection/censoring condition, qualifiers, status, media, routine/QC activity, laboratory method and duplicate result identifiers. The extractor itself does not filter general qualifier/status fields; this audit inspects them explicitly.

Laboratory method names are missing for 13/21 retained raw results. The largest 06438000 values are Historical routine records with no qualifier/detection flag and no method name. Confirming their environmental authenticity would require provider/laboratory documentation beyond this local archive; their scale alone is not evidence of a pipeline mistake.

## Full validation high-value population

Counts below use distinct station-months. Unique-cell error shares average repeated seed/partition predictions before summing, so repeated appearances add no sample count. The CSV also retains shares under the experiment's equal-partition weighting.

| DOC threshold | Cells / all cells | Stations | Raw absolute-error share | Raw squared-error share | Log absolute-error share |
|---|---:|---:|---:|---:|---:|
| >50 mg/L | 10/7197 | 5 | 12.8% | 87.9% | 1.3% |
| >100 mg/L | 5/7197 | 2 | 10.8% | 85.7% | 0.9% |

Station 06438000 contributes 11.7% of all unique-cell absolute error and 85.1% of squared error. This concentration explains why an SSE-based tail diagnosis can overstate the breadth of the problem; MAE contributions remain the relevant companion.

## Implication

No data deletion is supported by this audit. Preserve the high labels and their raw identities. Treat learning to reconstruct high DOC as a model question, while separately tracking how concentrated the evaluation is in historical records. An unflagged archived measurement and a validated ecological event are different levels of evidence.

## Reproduction and evidence

`uv run python scripts/audit_doc_tail_source_v1.py`

`top20_trace.csv` contains month-level provenance and classification; `raw_doc_records.csv` contains retained raw values/identifiers and metadata; `raw_organic_carbon_candidates.csv` also includes the excluded same-month carbon observations. `validation_high_values.csv` and `validation_station_error_mass.csv` retain population counts/contributions. `sources.json` records raw/code/dataset hashes. Values come from WQP; local NWIS site/catalog files provide inventory, not an independent DOC concentration measurement.
