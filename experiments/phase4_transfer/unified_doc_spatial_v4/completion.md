# Recurrent few-shot DOC adaptation: completed experiment

## Decision

All nine runs completed. The updated adapter is retained as a research candidate;
it does not replace the existing default model. It reaches essentially the same
K = 5 mean error as the existing tree representation, but updating recurrent
weights does not produce a clear gain over the matched frozen-GRU control.

The complete temporal-shape adapter improves on a simple station-level correction.
That is a useful result, but its improvement must be separated from the much
smaller effect of recurrent fine-tuning itself.

## What was trained

Only the existing GRUCell and observation-decay layer were updated: **25,472
parameters** per package. Spatial encodings were cached from the existing model.
Forests, the source-fitted two-dimensional readout and the fusion predictor stayed
fixed. Source station episodes used three or five observations to fit a ridge
correction, and the remaining observed months supplied the learning objective.

The new normalization uses the same 32 label-free calendar anchors in frozen
and updated controls. It removes uniform feature scaling above a numerical floor.
The recurrent states use causal 12-month windows; normalization and record-spanning
support make the complete reconstruction retrospective.

There were 104 executed epochs across nine packages, with a maximum budget of
30 epochs per package. Five selected updated weights, and four selected epoch 0.
Mean source-validation MAE under the checkpoint criterion changed from 1.7545 to
1.7520. No station hit the normalization floor in either the initial or selected
full-cohort products. Recorded run time was **74.4 seconds**, excluding earlier
expert training and analysis. A separate single-configuration, two-epoch smoke
run and full checkpoint replay also completed.

## Held-station performance

MAE in mg/L. Seeds are averaged within partition and the three station partitions
are equally weighted. All rows use identical query cells and support observations.

| Frozen-fusion adapter | K = 3 | K = 5 | K = 5 RMSE | K = 5 Q90 MAE |
|---|---:|---:|---:|---:|
| Constant station correction | 1.6392 | 1.6085 | 3.6113 | 6.7368 |
| Prior supervised GRU projection | 1.6358 | 1.5987 | 3.5961 | 6.6904 |
| Frozen GRU, new normalization | 1.6279 | 1.5973 | 3.5937 | 6.6941 |
| Updated GRU, same normalization | 1.6300 | 1.5952 | 3.5880 | 6.6886 |
| Existing supervised tree projection | 1.6307 | 1.5952 | 3.5994 | 6.7115 |

### Recurrent training itself

At K = 5, updating GRU weights reduces MAE relative to the frozen, identically
normalized GRU by **0.1295%**, with a joint station-bootstrap 95% interval of
**-0.0991% to 0.3655%**. Two of three partitions improve. Three of nine fitted
packages improve; four packages retain their initial checkpoint. At K = 3 the
update instead increases average MAE by 0.1277%, also with an interval across zero.

The environmental-only base gives the same pattern: a 0.1330% K = 5 improvement
and a 0.1322% K = 3 deterioration. These comparisons do not establish a reliable
increment from recurrent fine-tuning.

### Complete adaptation method

Relative to the previous GRU adapter, the new complete method reduces K = 5
MAE by **0.2173%**; all three partition means improve, but its interval spans
-0.0562% to 0.4907%. This comparison includes the normalization change.

Relative to constant station correction, its K = 5 improvement is **0.8254%**
(interval **0.1182% to 1.5964%**), with improvement in all three partitions and
all nine packages. The environmental-only version improves by 0.8313% with the
same direction pattern. This evidence supports a small benefit of the complete
shape-adaptation method on this development panel; it cannot be assigned solely
to the recurrent update.

The updated GRU and tree adapters have nearly identical K = 5 MAE:
**1.595209 versus 1.595199 mg/L**. Their uncertainty interval does not establish
superiority or statistical equivalence. At K = 0 and K = 1 predictions remain
unchanged: this experiment updates support adaptation, not the zero-observation
spatial predictor or the scalar temporal expert used in fusion.

### High DOC

The updated GRU has slightly lower Q90 MAE than the frozen normalized control
(0.08% reduction) and tree adapter (0.34%). Both paired tail intervals span zero.
The tail evaluation contains 1,017 distinct query station-months at 108 stations,
with all three partitions represented. Tail performance does not provide a
separate basis for promoting this update.

## Next research action

The source-validation error decomposition is now complete and saved separately
in [`diagnostics/diagnostic.md`](diagnostics/diagnostic.md). After GRU adaptation,
within-station variation accounts for **86.2% of raw squared error** and 87.7%
of log squared error. The **9.5% of queries above the source-training Q90 account
for 45.0% of raw absolute error**, as well as 96.4% of raw squared error. Their
partition-equal mean signed residual is -8.149 mg/L, indicating underprediction.

On that reused source-validation set, adaptation reduces the station-bias MSE
component by 4.37% relative to constant correction, but reduces the within-station
component by only 0.55%. Associations with observation gaps, hydro coverage and
ecological novelty are weak or inconsistent across partitions. These are descriptive
error components, not independently validated causes or an irreducible error floor.

The next concrete investigation is high-DOC month reconstruction: inspect the raw
records and hydrological timing of large residuals, then assess whether the current
residual branch and monthly drivers represent those changes. The absolute-error
share makes this relevant to the primary MAE objective, not only to squared-error
metrics dominated by extremes.

The three representation iterations now cover unsupervised temporal bases,
supervised projection and recurrent-state fine-tuning. Their small incremental
gains motivate an error-led next experiment rather than another parameter sweep.
This does not identify an irreducible error floor.

## Reproducibility

These are development results on previously evaluated ST357 station partitions.
There are 10,520 distinct held-station query cells across 172 stations; repeated
seeds and partitions are accounted for in the joint station bootstrap. Earlier
experiments and manuscript claims remain unchanged.

- 568 tests passed, 2 skipped; ruff passed.
- All nine memory checkpoints regenerated their complete feature arrays and
  query predictions bitwise from the saved inputs and weights.
- The six reused version-3 control arms remain bitwise unchanged at every K.
- Historical artifact audit completed with its documented historical exclusions.

```bash
uv run python scripts/run_ladder.py --experiment unified-doc-spatial-v4
uv run python scripts/analyze_unified_doc_spatial_v4.py
uv run python scripts/verify_unified_doc_spatial_v4.py
uv run python scripts/diagnose_unified_doc_spatial_v4.py
```

The full replay requires local source expert packages and the derived representation
caches. Fitted memory weights, readouts, training traces, adapter choices, query
products and analysis are retained. See [`analysis/findings.md`](analysis/findings.md)
for all primary, performance, normalization and conditional-tail comparisons.
