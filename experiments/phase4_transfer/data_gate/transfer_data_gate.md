# Phase 4 transfer data gate

Generated: `2026-09-24T15:00:15.304975+00:00`
Stage-1 status: **external_pending**

The audit uses only dataset availability, metadata, graph identity, and frozen QC facts.
It does not load predictions or test labels for selection.

## ST357 local bundle

| analyte | stations | months | observed station-months | largest active component | sha256 |
|---|---:|---:|---:|---:|---|
| doc | 357 | 654 | 22571 | 0.608 | `33474f5584fd…` |
| ph | 357 | 654 | 54630 | 0.610 | `100be6eb967d…` |
| spec_conductance | 357 | 654 | 63009 | 0.611 | `5fc06dbaadb7…` |

Local errors: `0`

## QC replay and task availability

Raw-cache replay: **True** (370 files; retrospective only).

| analyte | task ID | HUC6 | K=5 task-months | post-2020 task-months | query cells |
|---|---|---|---:|---:|---:|
| doc | 101302 | 101302 | 28 | 0 | 72 |
| doc | 101900 | 101900 | 22 | 0 | 50 |
| doc | 102701 | 102701 | 32 | 0 | 78 |
| doc | 103001 | 103001 | 17 | 0 | 36 |
| doc | 510020 | 051002 | 34 | 0 | 68 |
| ph | 101302 | 101302 | 40 | 0 | 112 |
| ph | 101900 | 101900 | 26 | 0 | 65 |
| ph | 102701 | 102701 | 58 | 0 | 135 |
| ph | 103001 | 103001 | 23 | 0 | 63 |
| ph | 510020 | 051002 | 36 | 0 | 72 |
| spec_conductance | 101302 | 101302 | 42 | 0 | 118 |
| spec_conductance | 101900 | 101900 | 27 | 0 | 87 |
| spec_conductance | 102701 | 102701 | 62 | 0 | 143 |
| spec_conductance | 103001 | 103001 | 25 | 0 | 68 |
| spec_conductance | 510020 | 051002 | 57 | 0 | 114 |

K=5 inventories are availability diagnostics, not a new confirmatory split.
Existing task IDs are preserved; `510020` maps to canonical HUC6 `051002`.

## External candidates

No candidate manifest supplied; external validation is pending.

No model training is authorized by this report.
