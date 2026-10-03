# Station-support basis refresh: completed comparison

Nine packages are complete across three station partitions and three seeds.
This experiment performs no new neural or forest training. It refreshes the
station-support representation from 27 selected recurrent checkpoints while
retaining their native predictions, the frozen v4 two-dimensional readout and
the same 32-anchor normalization. Direct adapters and ecological mixers are
fitted on the existing source-validation episodes.

## Results

MAE in mg/L, seed-averaged within each partition and then partition-equal.
The table compares each expert with its own unchanged legacy support basis.

| Expert / pipeline | Legacy K3 | Refreshed K3 | Legacy K5 | Refreshed K5 |
| --- | ---: | ---: | ---: | ---: |
| Daily head / direct | 1.631317 | 1.607645 | 1.589387 | 1.590976 |
| Current GRU input / direct | 1.634744 | 1.611741 | 1.587180 | 1.590835 |
| Full GRU history / direct | 1.631764 | 1.619470 | 1.586407 | 1.593891 |
| Daily head / integrated | 1.620090 | 1.599087 | 1.564984 | 1.572464 |
| Current GRU input / integrated | 1.622109 | 1.599909 | 1.567740 | 1.570676 |
| Full GRU history / integrated | 1.620559 | 1.610500 | 1.567575 | 1.573159 |

The direct daily-head expert improves K3 by **1.45%**, with difference
**−0.023671 [−0.047630, −0.001123]**. The current-input expert improves
**1.41%**, with **−0.023003 [−0.046626, −0.000345]**. Full-history and
integrated K3 differences are numerically favorable but their overall intervals
span zero. The daily-head integrated K3 reduction is **1.30%**.

The K3 benefit primarily concerns ordinary DOC and partition 144. Direct
daily-head/current-input models improve in five/four of nine fits; about 58%
of positive station gain comes from five stations. The K3 Q90 point estimates
worsen, with intervals spanning zero. This is a conditional support-calibration
benefit rather than a broad improvement of the native prediction.

Every refreshed K5 overall estimate is worse than its matched legacy estimate;
their overall intervals span zero. Full-history direct Q90 error increases by
**+0.050762 [+0.004498, +0.098961]**. The unchanged daily-head integrated
model remains the main K5 candidate at **1.564984 mg/L**. No target-selected
K3/K5 switching rule is introduced.

## Why K0 and K1 are unchanged

All support bases preserve K0 predictions bitwise. K1 also reproduces the
legacy prediction exactly: the adapter enables temporal shape correction only
for more than one support reading, and centering a single support gives zero
shape rank. This is an algebraic control, not a low-power statistical result.

Source-validation K3 improves across all three experts, by approximately
0.013–0.015 mg/L. Integrated validation K5 changes are −0.001854 for current
input, −0.002495 for full history and +0.000233 for the daily-head expert.
The small current/full K5 validation improvements do not transfer to the
held stations. All selected alpha/ridge/gamma choices and candidate scores
are saved, so the representation change can be distinguished from parameter
reselection.

## What this experiment resolves

Updating the state read by station adaptation can improve three-reading
reconstruction. The fixed two-dimensional readout was learned in older hidden
coordinates, however; refreshing states alone does not establish that it remains
appropriate for the new coordinates. The next experiment in `next_iteration.md`
learns that readout from source-station support episodes while freezing the
prediction model. A failed fixed-readout refresh does not show that the new
hidden states contain no useful support information.

## Reproduction

```bash
uv run python scripts/run_ladder.py --experiment doc-daily-hydro-support-basis-v1
uv run python scripts/diagnose_doc_daily_hydro_support_basis_v1.py
uv run python scripts/verify_doc_daily_hydro_support_basis_v1.py
uv run python scripts/analyze_doc_daily_hydro_support_basis_v1.py
```

Independent replay reproduces every projected state, normalized basis and
anchor statistic bitwise. All 162 validation support/mixing refits and
2,778,840 query rows match exactly. Parent predictions and checkpoints remain
unchanged. The full suite passes **743 tests**, with two explicit skips;
ruff passes. Analysis bindings and all 18 fixed contrasts are verified.

See `PRODUCTS.md`, `diagnostics/source_validation.md`,
`analysis/interpretation.md`, `analysis/findings.md` and
`verification/replay_checks.json`. Raw states are causal, while unchanged
record-wide feature-anchor normalization and positive-K support are
retrospective. The station partitions remain development data.
