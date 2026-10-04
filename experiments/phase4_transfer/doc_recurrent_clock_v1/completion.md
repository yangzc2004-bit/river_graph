# Recurrent decay clocks: completed DOC comparison

All nine packages are complete. Eighteen new neural fits compare unseen-neutral
and within-window discharge-age decay against nine reused, matched daily-head
legacy controls. The new fits run928 epochs in total, max94 under a120 ceiling,
in331.9 seconds of package execution. Forests, ecological residual profiles and
the old station-support basis are retained. The new arms refit the last spatial
self/ecology layers, GRU/decay and native head under the same initialization,
loss, learning rates, input views and stopping rule as legacy.

## Results

Integrated native MAE in mg/L; seeds averaged within each station partition,
then the three partitions equally weighted.

| Decay clock | K0 | K1 | K3 | K5 |
| --- | ---: | ---: | ---: | ---: |
| Legacy DOC age | **1.803538** | **1.755368** | 1.620090 | **1.564984** |
| Unseen neutral | 1.815765 | 1.771027 | 1.626570 | 1.570798 |
| Within-window flow age | 1.811108 | 1.764629 | **1.610432** | 1.568986 |

Neither new version replaces the integrated main candidate. At K5 neutral is
+0.005814 worse[+0.000946,+0.010517]; flow is+0.004002 worse
[−0.001616,+0.009594]. K0 differences are numerically unfavorable and their
overall intervals span zero. Flow's K3 estimate improves, but this experiment
does not construct a target-selected K-specific model route.

The direct recurrent models have numerically favorable K5 differences:
neutral MAE1.576967 versus legacy1.589387 (0.78% lower), flow1.572724
(1.05% lower). Their intervals span zero, and these gains do not carry into
ecological integration. Native prediction and mixture selection must be read
together; improved direct point estimates are not a new integrated winner.

## High DOC and source selection

Both interventions worsen K0 Q90 error, with station intervals entirely above
zero. Integrated K5 Q90 error also increases: neutral+0.021230
[+0.005404,+0.041212], flow+0.020420[+0.004840,+0.038314]. K0 high-DOC
recall drops by roughly1.4–1.5 percentage points. Their slightly lower
false-high rates accompany worse high-DOC recovery.

Mean selected source-validation native MAE is1.804428 for legacy,1.822566
for neutral and1.818777 for flow. No new fit reaches the budget ceiling.
The current evidence gives no reason to extend these same fits.

The neutral intervention changes realized model capacity: on all validation
query windows,62–64 of64 latent channels have gamma exactly1. All seed44
packages have identity decay in every channel. The intervention therefore
nearly removes external recurrent-state attenuation; GRU internal gates remain
active. It should not be described as merely deleting64 age coefficients or
as a matched-effective-capacity comparison. Flow's clock uses only the queried
12-month discharge-visibility window; it adds no pre-window history.

## Model decision and next step

Retain the daily-head neural expert with legacy support basis and ecological
integration, K5 MAE1.564984 mg/L. These controlled clock comparisons do not
identify the original clock as a performance bottleneck. They do not establish
that every alternative storage mechanism would fail.

The next bounded experiment is a conditional distributional residual head,
specified in `next_iteration.md`. It freezes the current trunk and compares
single- and two-component heads, reporting the predictive median. The goal is
to improve high-DOC reconstruction without raising ordinary false-high errors.
No new distributional fits have been performed in this clock experiment.

## Reproduction

```bash
uv run python scripts/run_ladder.py --experiment doc-recurrent-clock-v1
uv run python scripts/verify_doc_recurrent_clock_v1.py
uv run python scripts/analyze_doc_recurrent_clock_v1.py
```

The exact executed sources are in `code_snapshot/`; `execution_note.json`
records a later import-wrapping-only change to the current runner, with the
complete AST unchanged. Frozen archive contents are preserved. Verification
replays the archived execution's inputs, clocks and products; re-execution
with a changed checkout uses a fresh output root.

All14 model curves and eight specified5000-draw station-bootstrap contrasts
are saved, including unfavorable results. See `PRODUCTS.md`,
`analysis/interpretation.md`, `analysis/findings.md`,
`analysis/source_validation_decay_capacity.csv` and
`verification/replay_checks.json`. The full suite passes764 tests with two
explicit skips; current-code ruff and historical provenance audit pass.
Previously seen station partitions and positive-K retrospective reconstruction
retain their existing interpretation.
