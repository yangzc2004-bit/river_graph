# Daily hydrologic memory: completed DOC spatial reconstruction experiment

All nine station-partition/seed packages are complete: 27 neural fits and
18 matched ExtraTrees fits. The neural fits execute 1,381 epochs in total;
the longest run stops at epoch 77, below the common 120-epoch ceiling.

This experiment adds a zero-initialized eight-feature hydrologic projection
to the existing GRU. Current-only and full-history models have the same
512 additional parameters. The original daily-head-only model is retained
as an exactly reproduced control. All models use the same source OOF targets,
spatial/ecological initialization, query cells and validation-selection rules.

## Spatial reconstruction results

MAE in mg/L, averaging seeds within each station partition and then weighting
the three partitions equally. Integrated models use the established GRU support
basis and source-validation-selected ecological mixture.

| Integrated neural model | K0 | K1 | K3 | K5 |
| --- | ---: | ---: | ---: | ---: |
| Original daily head; no recurrent daily input | 1.803538 | 1.755368 | 1.620090 | 1.564984 |
| Daily input at the current GRU step | 1.807859 | 1.760206 | 1.622109 | 1.567740 |
| Daily input throughout the causal history | 1.797393 | 1.752831 | 1.620559 | 1.567575 |

Full history improves K0 relative to the matched current-only model by
**0.58%**, with paired difference **−0.010466 mg/L [−0.0189, −0.0022]**.
Before ecological mixing, the corresponding gain is **0.53%**:
**−0.009555 [−0.0178, −0.0013]**. Two of three station partitions and seven
of nine fits improve in the direct comparison. Historical water conditions
therefore contribute beyond adding the same projection only at the current
month; the gain is small and varies across partitions.

Relative to the original daily model, integrated full-history K0 improves
**0.34%**, but its interval spans zero:
**−0.006145 [−0.023423, +0.013350]**. At K5, full history has
**+0.002591 [−0.002107, +0.007728]** greater error than the original model.
Adding water history has not produced a replacement for the retained daily
K5 performance candidate. Full history is retained for the next memory
mechanism comparison rather than selected as an overall winner.

## Equal-information tree comparison

The two independent tree probes use the previously selected context
ExtraTrees configuration, without retuning. Both have 147 inputs; the
current probe zeros older daily slots and the history probe retains them.
They do not replace the frozen forest base used by the neural residual.

| Model, direct GRU support adaptation | K0 | K5 |
| --- | ---: | ---: |
| Original context forest | 1.9028 | 1.5960 |
| Tree with current daily information | 1.855769 | 1.573251 |
| Tree with daily history | 1.848927 | 1.577256 |
| Neural full-history residual | 1.800854 | 1.5864 |

Tree history versus tree current has K0 difference
**−0.006841 [−0.019709, +0.005357]**; it does not establish a general
historical benefit. The neural full-history integrated model improves K0
over the tree-history probe by **2.79%**, with difference
**−0.051535 [−0.1034, −0.0030]**. K5 neural–tree contrasts remain unresolved.
The direct neural residual improves K0 over the original context forest by
**5.36%**, with difference **−0.1020 [−0.1726, −0.0379]**.

## High DOC and source-validation context

Integrated K5 Q90 MAE is **6.586994** for the original daily model,
**6.5842** for current-only and **6.5949** for full history. The new history
does not improve its high-DOC tail. At K0, neural models still have lower
Q90 error and higher recall than matched tree probes, accompanied by a
small increase in false-high predictions. All tail, ordinary-error and
classification comparisons remain in the analysis tables.

Source-validation direct K0 MAE is 1.804428 / 1.801147 / 1.798270 for
off/current/history. The historical gain over current-only is concentrated
in partition 143 there; partitions 142 and 144 worsen. Validation-selected
integrated K5 MAE is nearly tied at 1.604602 / 1.604251 / 1.604384.
This source-side picture is consistent with a modest mechanism effect,
not a broad performance breakthrough.

All 10,520 unique held-station query cells have no visible DOC history
before support adaptation. Consequently the observed-age axis contains
only the never-visible group; it cannot identify effects of old versus
recent local DOC readings. Daily hydrologic history support is separately
reported, including empty and partly covered windows.

## Reproduction and saved products

```bash
uv run python scripts/run_ladder.py --experiment doc-daily-hydro-memory-v1
uv run python scripts/verify_doc_daily_hydro_memory_v1.py
uv run python scripts/diagnose_doc_daily_hydro_memory_v1.py
uv run python scripts/analyze_doc_daily_hydro_memory_v1.py
```

Independent replay covers 27 neural checkpoints, 18 trees, 2,101,302
full-grid rows and 3,705,120 query-product rows. Neural predictions and
adapted queries are bitwise exact. All nine original controls reproduce
their preceding model states, traces and outputs. Source-validation
refitting reproduces 108 direct adapters and 54 integrated mixers.
Tree summation differs only at floating-point roundoff under the recorded
tolerance. The full suite passes **733 tests**, with two explicit skips;
ruff and the historical artifact audit also pass.

See `PRODUCTS.md`, `diagnostics/source_validation.md`, `analysis/findings.md`,
`analysis/strata_appendix.md` and `verification/replay_checks.json`.
The label-free recurrence audit in `diagnostics/hydro_decay.md` quantifies
the inherited observation-clock attenuation. The immediate follow-up in
`next_iteration.md` first aligns support adaptation with the updated recurrent
state while retaining the same two-dimensional projection.
These station partitions remain development data. Same-month daily inputs
support month-end reconstruction, and positive-K support is retrospective.
The present expert uses the spatial self path; this comparison measures
hydrologic temporal memory rather than new river-message gain.
