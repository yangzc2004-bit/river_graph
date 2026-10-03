# Daily hydrologic information improves support-assisted DOC reconstruction

Within-month discharge summaries produce a reproducible improvement in the
complete ecological and station-support reconstruction pipeline. This is a
more useful change than the preceding loss adjustment: the matched comparison
reduces both high-DOC and ordinary-concentration errors. The improvement comes
from additional hydrologic information in the existing model, with the largest
gain in support-assisted spatial transfer.

## What changed

The existing ecological/self encoder and observation-aware GRU are retained.
Three bounded daily descriptors describe flow distribution width, consecutive
day variation and rising-day frequency. Two coverage fractions and three
validity flags accompany them. The eight channels enter the existing native
residual readout, with three value-by-hidden-state interactions. Monthly hydro
inputs, source OOF forests, ecological memory and the GRU support basis remain
fixed.

Three matched models share the same expanded head and training objective:
zero daily channels, daily availability only, and complete daily descriptors.
All 27 fits use three station partitions and three seeds, with a common
120-epoch ceiling and patience5. Every fit stops normally; the daily arm stops
by epoch 77. Total executed epochs are 1,072 and recorded fitting/product time
is 375.5 seconds. Source-validation MAE improves in all nine daily fits, by
3.51% relative to the monthly control and3.47% relative to availability only.

## Reconstruction at new stations

| Ecological-integrated model, GRU support basis | K0 MAE | K1 MAE | K3 MAE | K5 MAE |
|---|---:|---:|---:|---:|
| Previous overall ecological-affine reference | 1.809217 | 1.791696 | 1.622073 | 1.579905 |
| Fresh monthly control | 1.809588 | 1.777090 | 1.626636 | 1.588987 |
| Daily availability only | 1.808665 | 1.775656 | 1.626066 | 1.588716 |
| **Complete daily information** | **1.803538** | **1.755368** | **1.620090** | **1.564984** |

Errors are in mg/L. Seeds are averaged within each partition, then the three
partitions receive equal weight. K counts retrospective target-station support
observations; the same reserved support schedule and fixed queries are used.

At K5, daily information reduces integrated MAE by **1.511%** versus monthly,
95% gain interval **0.664–2.331%**, and **1.494%** versus availability only,
interval **0.657–2.301%**. All three partition means and all nine fits improve
in both comparisons. This separates numerical daily information from mere
data availability.

Compared with the previous overall model, MAE falls from 1.579905 to 1.564984,
an improvement of **0.944%**, interval **0.194–1.711%**. All three partition
means and eight of nine fits improve. The constant-only support path also
improves, so the result does not depend solely on the GRU-shaped adapter.
Retain the daily integrated model as the next performance candidate, with the
monthly, availability and earlier reference models alongside it.

## High DOC and ordinary concentrations

At K5, against the matched monthly model:

- Q90 MAE falls from 6.660718 to **6.586994 mg/L**, delta interval
  **-0.138865 to -0.013241**.
- Ordinary MAE falls from 1.004322 to **0.985640 mg/L**, delta interval
  **-0.034521 to -0.005131**.
- Recall changes from 64.85% to 65.13%, and the false-high rate from 2.221%
  to 2.188%; both difference intervals span zero.

Against the earlier overall model, K5 Q90 MAE improves **1.403%**, interval
**0.888–1.913%**, with all nine fits improving. Tail bias becomes less negative,
from -5.132 to -4.966 mg/L.

K0 overall improvement remains uncertain: 1.809217 to 1.803538, delta interval
-0.061615 to +0.047049. Its Q90 MAE nonetheless improves **3.399%** relative to
the earlier model, and recall rises from 60.08% to 64.06%, a **3.99 percentage
point** difference, interval 1.67–7.07. The false-high difference interval spans
zero; this does not establish equality of false-high rates.

The isolated direct-residual K5 contrast remains uncertain. The stronger gain
is in the full integrated workflow, which also includes source-validation
ecological mixing and support calibration. It should not be attributed entirely
to the GRU or to graph messages; this experiment retains the no-message self
path. A method-level comparison with RF should supply the same daily covariates
to RF before claiming an architectural advantage.

## Where to improve next

Daily descriptors are valid in **8,141 of 10,520 unique fixed query cells
(77.39%)**. Among those cells, matched integrated K5 MAE falls from 1.746384
to 1.716752. The remaining cells are approximately unchanged at K5. At K0,
the insufficient-daily group instead worsens, from 1.155796 to 1.205614.
These descriptive strata point to protecting reconstruction where new
hydrologic information is absent, while retaining its benefit where available.

The independent source-validation coverage diagnostic supports this direction:
integrated MAE improves from 2.0071 to 1.9309 when all descriptors are valid
(nine of nine fits), but changes from 1.3651 to 1.3737 otherwise. The latter
effects vary by partition. Almost every insufficient-data validation cell has
all three numerical descriptor flags zero. A fixed fallback to the saved
monthly model at those cells is the next focused comparison, before expanding
the recurrent inputs or introducing a learned gate.

The next iteration should address that missing-information behavior on the
existing model. Further daily-memory integration can follow once its benefit
is separated from the ordinary-station performance change. The unchanged
overall reference remains available; no test-selected K-specific switching
policy is introduced by this experiment.

## Completion and reproduction

The full suite passes **698 tests, 2 skipped**; Ruff and the historical artifact
audit pass. The test configuration now explicitly includes the repository root,
so the documented `uv run pytest` command imports script modules consistently.

Independent replay covers 27 checkpoints, 2,101,302 full-grid rows and720
model-by-K panels (3,087,600 query rows), including source-validation-only
adapter/mixer refits. Neural and adapted predictions reproduce bitwise. All
eight daily descriptors independently rebuild bitwise from 78 cached files and
3,659,912 unique daily records. Forest recomputation differs by at most4.97e-14.
Historical reference predictions remain unchanged.

Current-month daily records are information for retrospective monthly
reconstruction, not month-start forecasting. All results remain same-cohort
development evidence, with 5,000 paired whole-station bootstrap draws and all
models retained.

```bash
uv run python scripts/build_doc_daily_flow_features_v1.py
uv run python scripts/run_ladder.py --experiment doc-daily-hydro-residual-v1
uv run python scripts/analyze_doc_daily_hydro_residual_v1.py
uv run python scripts/verify_doc_daily_hydro_residual_v1.py
```

[Complete analysis](analysis/findings.md) · [Interpretation](analysis/interpretation.md)
· [Availability appendix](analysis/availability_appendix.md)
· [Data feasibility](diagnostics/daily_hydro_feasibility.md)
· [Source validation](diagnostics/source_validation_training.md)
· [Source coverage diagnosis](diagnostics/source_validation_hydro_coverage.md)
· [Product guide](PRODUCTS.md) · [Replay](verification/replay_checks.json)
