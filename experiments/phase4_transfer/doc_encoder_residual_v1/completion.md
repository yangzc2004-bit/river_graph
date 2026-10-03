# DOC encoder adaptation: completed iteration

The existing DOC representation benefits from adaptation to the native-scale
residual objective. Updating its final self layer and ecological encoder lowers
zero-observation spatial MAE in all three station partitions and all nine
partition/seed packages. This is a modest improvement to the current tail
candidate; the earlier support-aware ecological model remains better overall.

## Matched model comparison

| Model with the fixed GRU support adapter | K0 MAE | K5 MAE | K0 Q90 MAE | K5 Q90 MAE |
|---|---:|---:|---:|---:|
| Frozen concentration-conditioned encoder | 1.843635 | 1.588311 | 7.284164 | 6.624432 |
| Final self-layer update | 1.835723 | 1.588283 | 7.271988 | 6.622611 |
| Final self-layer + ecology update | **1.829927** | **1.586527** | **7.254645** | **6.620441** |

Errors are in mg/L, with seeds averaged within station partition and partitions
equally weighted. K is the number of reserved DOC observations per target
station. Query cells are fixed across K and models; support is retrospective.

Combined updating improves K0 MAE by **0.744%**, with a paired whole-station
bootstrap interval of **0.073% to 1.603%**. All three partition means and all
nine fits improve. Its K0 Q90 MAE drops by 0.0295 mg/L, interval -0.0573 to
-0.0005. Non-tail MAE falls by 0.0123 mg/L on average, with an interval spanning
zero. False-high rate rises slightly, by 0.030 percentage points.

At K5, the 0.112% overall improvement has an interval of -0.116% to 0.429%.
Most of the encoder benefit occurs before local observations can calibrate the
station. Extra ecological tuning improves over self-layer-only tuning, but
its added effect is small.

## Relation to the current overall model

Combining the updated residual with unchanged ecological memory gives K0/K5
MAE **1.813615/1.584405**, versus **1.809217/1.579905** for the earlier overall
model. The new means are slightly worse; their paired intervals span zero.

The integrated candidate does improve Q90 error relative to that model by
**2.11% at K0** and **0.62% at K5**, with positive paired intervals. These
benefits come with higher mean ordinary-concentration error and false-high
rates (+0.116/+0.073 percentage points). It remains a stronger high-DOC
candidate rather than replacing the overall model.

## What was trained

All modes share the original encoder/GRU initialization, concentration head,
30-epoch cap, patience 5, twofold source-tail weighting, source OOF context
bases and source-validation checkpoint/scale selection. Encoder dropout is
off in all modes. Only the requested weights can move: 0, 4,160 or 5,536
encoder parameters. GRU/decay and head learning rates are unchanged; selected
encoder weights use 1e-5.

Empty graph edges are retained, matching this spatial-transfer residual's
self path. Thus the gain concerns local ecological representation and memory,
not new river transport information. Existing input normalization, forests,
support representations and ecological-memory profiles remain unchanged.

## Next experiment from source-validation traces

Fourteen of 27 fits reach the 30-epoch ceiling before patience is exhausted;
ten set a new validation best at epoch 30. This includes frozen controls.
The next comparison extends only the maximum budget to 60 epochs in a new
directory, preserving all three matched modes and patience 5. Every fit starts
from its original initialization. This separates additional optimization from
the incremental value of encoder adaptation.

## Reproduction

Nine packages contain 27 neural fits (645 executed epochs; 237.6 seconds of
recorded fitting/product time) and 18 reported models, with 45 paired
contrasts and 5,000 whole-station bootstrap draws. These previously examined
station partitions remain model-development data.

All 27 checkpoints and 648 model-by-K outputs replay bitwise exactly, covering
2,101,302 grid rows and 2,778,840 query rows. Seventy-two direct adapters and 54
integrated mixers are exactly refitted using source validation. The new frozen
arm also reproduces the previous cached concentration model exactly in this
experiment. Initial representation equivalence was checked independently.
Forest recomputation differs by at most 4.97e-14.

**667 tests passed, 2 skipped**; Ruff passes and the historical artifact audit
exits zero. Every candidate and earlier reference is retained.

```bash
uv run python scripts/run_ladder.py --experiment doc-encoder-residual-v1
uv run python scripts/analyze_doc_encoder_residual_v1.py
uv run python scripts/verify_doc_encoder_residual_v1.py
```

[Results](analysis/findings.md) · [Source-validation diagnosis](diagnostics/source_validation_diagnostic.md)
· [Product guide](PRODUCTS.md) · [Replay](verification/replay_checks.json)
