# DOC regime readout: completed model iteration

Twenty-seven neural fits across three station partitions and three seeds tested
whether direct station-regime conditioning improves the existing native DOC
residual. The forest, spatial encoder and support basis were held fixed. Each
fit tuned the original observation-aware GRU/decay and a zero-initialized head.

## Why this change was tested

Source OOF and validation diagnostics showed persistent high-DOC underprediction
after station balancing. Many true high-DOC cells had context predictions below
the source Q90 threshold, with larger errors than high-DOC cells already predicted
above that threshold. A hard gate on an already high prediction would miss these
cases. We therefore used predicted concentration continuously, together with
direct watershed ecology, to condition the recurrent readout.

All three heads see the same 30 features. The additive head retains only the
existing hidden-state/flow interactions. The concentration head adds 64
hidden-state/predicted-concentration weights. The ecological head adds another
576 hidden-state/ecology weights. Source concentration features come from OOF
predictions, not source truth or the in-sample forest. Feature scales use source
stations only. Budget, initialization, twofold tail weighting, checkpoint
criterion and support queries are matched.

## Overall and high-DOC reconstruction

| Direct head with the existing GRU support adapter | K0 MAE | K5 MAE | K0 Q90 MAE | K5 Q90 MAE |
|---|---:|---:|---:|---:|
| Prior interaction | 1.835240 | 1.581605 | 7.437106 | 6.674577 |
| Additive regime inputs | 1.834500 | 1.587445 | 7.325763 | 6.646072 |
| Concentration conditioning | 1.843635 | 1.588311 | **7.284164** | **6.624432** |
| Full ecology conditioning | 1.879659 | 1.585962 | 7.298621 | 6.629646 |

Errors are in mg/L, seed-averaged within partition and then partition-equally
averaged. Q90 is derived separately from each partition's source labels.
Queries are the same at every K; reserved support is retrospective.

Concentration conditioning improves high-DOC MAE relative to the prior
interaction model by **2.06% at K0**, with a 95% whole-station bootstrap interval
of **1.03% to 2.81%**, and **0.75% at K5**, with an interval of **0.43% to 1.09%**.
All three partition means improve at both support levels; nine of nine fits
improve at K0 and seven at K5.

The matched concentration-versus-additive control gives a smaller tail
increment: Q90 MAE changes by -0.0416 mg/L at K0 (interval -0.0770 to 0.0018)
and -0.0216 at K5 (interval -0.0372 to -0.0002). Thus the total gain includes
direct regime features as well as the concentration interaction. Full ecology
interaction does not establish another improvement.

## The cost of this tail gain

Concentration conditioning raises non-tail MAE by **0.0282 mg/L at K0** and
**0.0138 at K5** relative to the prior interaction model. Its overall MAE is
0.46% and 0.42% higher, respectively; paired intervals span zero. These are
mean costs, rather than evidence of equal performance.

Tail signed bias improves from -6.439 to -6.066 mg/L at K0. At the same time,
the false-Q90 rate among ordinary concentrations rises from 1.950% to 2.082%
(+0.132 percentage points, interval +0.053 to +0.250). At K5 the increase is
0.067 percentage points, interval +0.007 to +0.153. The correction is more
sensitive to high values but also raises some ordinary predictions.

The broader ecological readout adds ordinary-concentration harm without a
clear tail advantage over the compact concentration head. Adding further
static interactions is not the next priority.

## Integration with the existing model

All direct heads were also combined with the unchanged ecological-affine memory
using source-validation mixing and support adaptation. Their K0/K5 MAE values
are 1.817633/1.584076 (additive), 1.824594/1.585896 (concentration), and
1.855548/1.585611 (ecological). The preceding support-aware ecological model
remains better overall at **1.809217/1.579905**.

The current overall model is retained; the compact concentration-conditioned
head is a candidate for high-DOC reconstruction. No new arm replaces the
overall model based on this development panel. The substantive next problem
is selective correction: improving high-DOC sensitivity without raising
ordinary concentrations. A representation update, comparing frozen versus
partially trained spatial/ecological encoding under the native residual
objective, is more informative than widening this scalar head again.

## Completion and reproduction

Nine packages contain 27 fits and 18 reported models. There were 603 training
epochs in total; recorded training/product time was **184.8 seconds**, excluding
earlier expert fitting, analysis and independent replay. Every candidate and
all K curves are retained. Analysis reports 39 predefined contrasts and 5,000
joint whole-station bootstrap draws. These previously used same-cohort station
partitions remain development data.

All 27 checkpoints and 648 model-by-K query groups reproduce bitwise exactly.
Independent replay covers 2,101,302 component-grid rows and 2,778,840 query rows.
Validation-only refits recover 72 direct adapters and 54 integrated mixers.
Forest recomputation differs by at most 4.97e-14, within the 1e-12 tolerance;
it is not called bitwise exact. All old reference predictions are unchanged.

The test suite passes **641 tests, 2 skipped**; Ruff passes and the historical
artifact audit exits zero. Source/full feature construction, normalization,
support-label isolation and missing ecology are checked explicitly.

```bash
uv run python scripts/run_ladder.py --experiment doc-regime-residual-v1
uv run python scripts/analyze_doc_regime_residual_v1.py
uv run python scripts/verify_doc_regime_residual_v1.py
```

[Results](analysis/findings.md) · [Source/validation diagnostic](diagnostics/native_regimes.md)
· [Replay](verification/replay_checks.json)
