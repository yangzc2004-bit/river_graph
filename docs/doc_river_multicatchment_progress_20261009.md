# River form and DOC response: independent catchment comparison

## Current research state

The previous controlled routing experiments and Kervidy field measurements are
preserved. The next independent comparison is complete under
`experiments/phase4_transfer/doc_river_multicatchment_pulses_v1/`.

Three real outlets are now measured under a common hourly protocol: Kervidy,
Rappbode and Bouleau. They contribute 128 resolved positive DOC responses and
112 complete DOC/flow half-width pairs. River maps are actual provider vectors
or attributed published maps, not generated river illustrations.

| Case | Resolved DOC / widths | Median lag, h | Median width ratio |
|---|---:|---:|---:|
| Kervidy | 58 / 55 | 2 | 1.556 |
| Rappbode | 62 / 51 | 3 | 1.444 |
| Bouleau | 8 / 6 | 22 | 1.533 |

DOC spreading is observed in all three median responses. The measured widths
do not establish a simple ordering by visually described river form. Bouleau's
lag median changes from 34 h in 2018 (five responses) to 0 h in 2019 (three),
under the same mapped form. Catchment summaries therefore do not isolate
geometry from event forcing and DOC mobilisation.

## Data distinctions retained

- Rappbode DOC/Q are provider-processed with a 2.5 h moving average and short-gap
  interpolation. Duplicate conflicting DOC clocks are masked.
- Bouleau uses only observed/calibrated DOC; the separate RF-predicted DOC column
  is never used for pulse measurements. Its lag interval reaches zero.
- Kervidy's earlier native 59-response/55-width result is unchanged; the new
  area's specific-discharge threshold and hourly sampling use a separate table.
- Only Kervidy currently has a validated rooted vector network for measured
  path dispersion and common terminal lengths. The other two original maps
  support qualitative descriptions, not complete numerical network inventories.

## Next scientific work

Keep the target as **network geometry modifying an incoming DOC pulse**.
Obtain comparable rooted vectors for the other measured outlets; measure
headwater-path dispersion, confluence positions and common terminal distance.
Add independent events/catchments with similar flow duration and amplitude
before estimating a form effect. Prioritize simultaneous branch/receiver DOC
records to separate differing arrivals from spreading after confluence.

Do not treat outlet DOC–flow peak lag as channel travel time, call events
independent river-form replicates, classify peatland pools as tributaries,
substitute land-cover differences for river form, or restart the stopped DOC
training automation. No model training was performed in this version.

The analysis, figures and decisions are reproducible. Validation: 1,384 tests
passed, 2 skipped; Ruff passed; historical artifact audit exited 0 with its
documented historical limitations. New input/code/output receipts match files.
