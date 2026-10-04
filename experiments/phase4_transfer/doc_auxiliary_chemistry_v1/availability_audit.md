# Auxiliary chemistry availability audit

## Research opportunity

Observed pH and specific conductance can provide contemporaneous chemical information to the existing DOC model. The retained DOC inputs currently contain temperature/discharge and ecological descriptors; previous auxiliary-analyte experiments used station profile statistics to select source neighbors, not current-month values as DOC predictors.

This study concerns **DOC reconstruction where conventional water chemistry may be known at the target station**. It does not redefine an entirely unmonitored station. Current-month covariates support retrospective monthly reconstruction, not advance prediction before the month ends.

## Grid and definitions

The DOC, pH and EC tensors exactly align on 357 stations × 654 months (1972-04 through 2026-09), node order, edges/attributes and hydro values/masks. The four features are `pH/14`, `log1p(EC)`, pH visibility and EC visibility. No statistics are fitted. The `no_aux`, `masks` and `chemistry` arms all retain the same actual-availability gate; `no_aux` inputs are four zeros.

| Auxiliary information | Observed station-months | Stations | On observed DOC cells | On genuinely DOC-missing cells |
|---|---:|---:|---:|---:|
| ph | 54,630 | 356 | 22,070 / 22,571 (97.78%) | 32,560 / 210,907 (15.44%) |
| spec_conductance | 63,009 | 355 | 22,033 / 22,571 (97.62%) | 40,976 / 210,907 (19.43%) |
| either | 63,824 | 356 | 22,201 / 22,571 (98.36%) | 41,623 / 210,907 (19.74%) |
| both | 53,815 | 355 | 21,902 / 22,571 (97.04%) | 31,913 / 210,907 (15.13%) |

## Fixed DOC query availability

Only masks and cell identities were used; no DOC values or prediction outcomes were inspected. Training rows are all source training observations. Validation/test rows are the existing fixed queries after reserving five support cells per station. Seeds reuse these same cells.

| Partition | Role | Cells | pH | EC | Both | Either |
|---|---|---:|---:|---:|---:|---:|
| 142 | train | 15,958 | 15,715 | 15,609 | 15,570 | 15,754 |
| 142 | val | 2,013 | 2,000 | 2,012 | 2,000 | 2,012 |
| 142 | test | 3,975 | 3,749 | 3,796 | 3,726 | 3,819 |
| 143 | train | 15,241 | 14,845 | 14,749 | 14,682 | 14,912 |
| 143 | val | 2,798 | 2,793 | 2,795 | 2,790 | 2,798 |
| 143 | test | 3,907 | 3,812 | 3,865 | 3,810 | 3,867 |
| 144 | train | 13,727 | 13,479 | 13,414 | 13,364 | 13,529 |
| 144 | val | 3,236 | 3,225 | 3,215 | 3,205 | 3,235 |
| 144 | test | 4,983 | 4,765 | 4,792 | 4,735 | 4,822 |

## Raw-data reconciliation

The accepted WQP measurements were regrouped by station/calendar month using arithmetic means, then converted to float32 in frozen grid order. Both masks and **every pH/EC tensor value reproduced exactly**.

| Check | pH | Specific conductance |
|---|---:|---:|
| Accepted cached measurements (all dates) | 133,484 | 168,903 |
| Accepted measurements inside frozen calendar | 123,325 | 158,100 |
| Units | standard units | uS/cm |
| Preserved acceptance range | 0–14 | 0–100,000 |
| Duplicate/conflicting measurement IDs | 0 / 0 | 0 / 0 |
| Observed zero-valued months | 0 | 8 |

Nonempty detection-condition records are excluded by the existing builder; their raw unit/detection counts are retained in the JSON. The current definitions include multiple parameter codes, historical/provisional statuses and a small number of filtered samples. These categories were counted, not silently removed. Calendar-month co-observation need not mean the same sampling instant. Measurements outside the frozen calendar, including old 1901 cache records, do not enter the feature grid.

## Earlier experiment and next comparison

`spatial_adaptation/cross_analyte_source_v3_k_session_30/visibility_audit.md` describes the earlier station-profile comparison. Its source validation selected `profile_k=0, source_k=160`; that result does not test current-month chemical covariates. The earlier full-profile policy also used a different auxiliary-data availability setting.

A focused comparison can hold the neural representation fixed and contrast no auxiliary inputs, auxiliary availability alone, and actual pH/EC values. A matched tree model should receive the same auxiliary block. Benefits should be reported separately where chemistry exists and where it is absent; only 19.74% of actually DOC-missing grid cells currently have either auxiliary observation.

## Sources

- `data/processed/mississippi_graph_graphfix_st357.pt`
- `data/processed/mississippi_graph_ph_st357.pt` and `.provenance.json`
- `data/processed/mississippi_graph_spec_conductance_st357.pt` and `.provenance.json`
- `data/raw/wqp_results/*.csv` for the frozen stations
- `scripts/build_analyte_dataset.py`
- `src/river_graph/models/auxiliary_chemistry_features.py`

Dataset/provenance, mask, definition and raw-cache file hashes are recorded in `availability_audit.json`. Underlying datasets and prior experiments remain unchanged.
