# Hydro context of remaining DOC tail errors

This is a descriptive diagnostic of reused source-validation queries for the updated-GRU
fusion adapter at K=5. These labels previously supported model/checkpoint selection.
No model is trained and no outer-test DOC label or prediction table is used.

The panel has 8,047 partition-cell occurrences, 7,197 distinct station-months
and 140 distinct stations. Predictions/errors are averaged over three
seeds per partition. Tables retain individual partitions; summary means weight partitions equally.

## Tail errors and hydro coverage

Residual = prediction minus observation; negative values indicate underprediction.

| Population | Mean partition MAE | Mean signed residual | Current flow coverage | Current temperature coverage |
|---|---:|---:|---:|---:|
| all | 1.642 | -0.610 | 79.6% | 98.3% |
| q90 | 9.109 | -8.149 | 85.5% | 95.9% |
| non_tail | 1.000 | 0.034 | 79.0% | 98.5% |

## Descriptive hydro shifts

Flow anomaly is (current flow − prior-12-month mean) / prior mean absolute flow,
requiring at least three earlier observed flow months. It preserves flow sign and is
unit invariant. Temperature anomaly is a difference in degrees Celsius. No DOC values
or future hydro months enter these features. Missing hydro remains missing.

| Feature | Non-tail mean of partition medians | Q90 mean of partition medians | Q90 coverage |
|---|---:|---:|---:|
| discharge_past12_anomaly | -0.256 | -0.384 | 84.0% |
| discharge_change1 | -0.035 | 0.031 | 85.3% |
| discharge_change3 | 0.001 | 0.130 | 83.7% |
| discharge_change12 | -0.028 | -0.114 | 77.1% |
| temperature_past12_anomaly | 0.137 | 0.318 | 92.3% |
| temperature_change1 | 0.475 | 0.333 | 78.6% |
| temperature_change12 | 0.000 | -0.014 | 72.5% |

Station-centered descriptive Spearman correlations with absolute error follow.
Each variable is centered within station on available paired validation cells; stations
need at least three pairs. This removes mean offsets, not all ecological confounding.

| Feature | Split 142 | Split 143 | Split 144 |
|---|---:|---:|---:|
| discharge_past12_anomaly | 0.153 | 0.093 | 0.091 |
| discharge_change1 | 0.112 | 0.071 | 0.117 |
| discharge_change3 | 0.094 | 0.056 | 0.087 |
| discharge_change12 | 0.050 | 0.039 | 0.002 |
| temperature_past12_anomaly | 0.018 | 0.018 | -0.032 |
| temperature_change1 | 0.030 | 0.001 | -0.017 |
| temperature_change12 | -0.020 | -0.039 | -0.003 |

## Largest-error station-month cases

`top_error_cases.csv` retains the top 20 cells per partition (58 unique cells).
Among those unique cases, 56 meet their listed source-training Q90;
7 lack current flow and 2 lack current temperature.
Only 50 have a usable relative-flow anomaly; 8 of these are positive.
`top_error_months.csv` separately ranks calendar months by their total absolute-error mass
within each partition, retaining query counts so an isolated extreme is not mistaken for
a spatially widespread event. These are review cases, not independent event validation.

Station 06438000 contributes 81.8% of equal-partition squared error and 8.9% of absolute error; its maximum validation DOC is 460.0 mg/L.
The largest-error case is 06438000, 1980-07-01: observed 460.0, predicted 15.53 mg/L.
The completed [source audit](../data_audit/audit.md) verifies 20/20 largest-error months as provider DOC records and reproduces the frozen aggregation in 20/20. The files identify filtered USGS 00681 in mg/L; no pipeline defect or data deletion is supported.
Laboratory method metadata are absent for 13/21 retained point results. Verified archived records do not independently establish the physical mechanism of these historical events. This report reuses that audit rather than rerunning it.
Station contributions under this report's equal-partition weighting are retained in
`station_error_mass.csv`; the audit also reports unique-cell-weighted contributions.

## What this diagnostic supports

The tail problem is strong underprediction despite mostly available monthly hydro inputs.
Short flow changes and relative flow anomalies show weak positive within-station associations
with error across all queries; tail-only associations vary across partitions. Most large-error
cases with a usable flow anomaly are not above their preceding flow baseline. A single
high-flow explanation therefore does not describe the observed tail failures. Temperature
changes and the annual-lag changes show weaker or inconsistent error associations.

## Minimal additions to the current residual model

The immediate native-MAE versus tail-weighted scalar-head experiment keeps inputs unchanged.
The following compact additions are candidates for a subsequent input comparison:

1. Keep the spatial encoder and GRU. Add a compact covariate block to the existing residual
   branch: relative flow anomaly and signed-log flow changes at 1/3 months, with their
   validity flags. The GRU already receives monthly raw hydro; these
   are explicit transformations and missingness cues rather than new measurements.
2. Supply prior-12 hydro counts or observation age alongside anomaly validity. M1's existing
   age and support counts refer to DOC, so they do not directly describe hydro freshness.

Temperature changes and same-month-last-year hydro differences are lower-priority additions
given this diagnostic. The latter would provide t−12 information outside the current window,
but the present associations do not make it the first change to implement.

The tables show which transformations have usable coverage and consistent descriptive
associations. They do not establish that any proposed input improves prediction. Preserve
missingness indicators and avoid converting absent covariates into apparent low-flow events.
Monthly flow/temperature and DOC may be aggregated from different observation dates; their
co-occurrence cannot establish storm timing, source flushing or a causal event mechanism.
A high monthly DOC value without a hydro shift can reflect unmeasured forcing, timing
mismatch or observation variability; this diagnostic cannot distinguish those explanations.

## Reproduction and definitions

Run `uv run python scripts/diagnose_doc_tail_hydro_v1.py`. All derived quantities are
covariate-only and causal in calendar time; the supplied DOC predictions themselves are
retrospective K-shot reconstructions. Covariate order/units come from the dataset builder.
The source grid contains 2 observed negative
flow cells; their signs are retained. Full bounded validation features are saved as parquet,
with partition/population distributions, missingness, correlations, source hashes and an
inventory of features already available to RF-context and M1. No significance tests are used.
