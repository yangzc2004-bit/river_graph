# DOC spatial adaptation v2: completed experiment

## What changed

The existing environmental forest and observation-aware recurrent residual
experts remain fixed. Two fitted components have been added:

1. A station-cross-validated fusion that shrinks corrections toward the
   environmental prediction.
2. A support-fitted temporal residual head using two source-trained, whitened
   within-station components. Frozen GRU states and individual-tree prediction
   vectors receive the same adapter capacity and selection budget.

All nine expert packages were processed: three station partitions by three
training seeds, six adaptation arms, and K = 0, 1, 3, 5. The experiment produced
926,280 query prediction rows in approximately 155 seconds of fitting and
feature extraction. These rows reuse the original ecological observations;
model and seed repetitions do not increase the ecological sample size.

## Main results

MAE in mg/L; individual-seed errors are averaged within partition, followed by
equal partition weighting.

| Predictor | No target observations | Five target observations per station |
|---|---:|---:|
| Version-1 hybrid with constant calibration | 2.0015 | 1.6261 |
| Environmental ExtraTrees with constant calibration | 1.9028 | 1.6094 |
| Regularized fusion with constant calibration | 1.9136 | 1.6085 |
| Regularized fusion with GRU shape adaptation | 1.9136 | 1.5990 |
| Regularized fusion with tree shape adaptation | 1.9136 | 1.5961 |

**Fusion repair produced the clearest improvement.** At K = 0, the new fusion
reduces MAE by 4.39% relative to the previous hybrid (joint station-bootstrap
95% interval: 1.62–7.32%). Its difference from ExtraTrees is small: MAE is
0.57% higher, with a paired reduction interval spanning -2.16–1.09%.
Validation selected the exact environmental predictor in five of nine fits
and a ridge correction in four. The selected temporal coefficients in those
four corrections are small and negative; these are statistical corrections,
not evidence of stronger river transport.

**Shape adaptation adds a modest, consistently directed increment.** At K = 5,
GRU shape adaptation improves the same fusion's constant-calibrated MAE by
0.59% (-0.02–1.32%), with positive partition-average effects in all three
partitions and improvement in seven of nine fits. Matched tree shape adaptation
improves it by 0.77% (-0.01–1.66%), also positive in all three partitions and
eight of nine fits. The GRU-versus-tree difference is not resolved by the
station bootstrap. The temporal head is genuinely enabled in seven of nine
GRU fits at K = 5, rather than every fit falling back to a constant offset.
Each adapter selects its own level shrinkage on source validation, so these
differences compare complete adapters rather than removing the shape term
while holding the level correction fixed.

**Overall MAE and tail error differ.** The GRU shape arm lowers Q90 MAE from
6.7368 to 6.7060 relative to the new constant-calibrated fusion. The previous
hybrid's K = 5 Q90 MAE is still lower at 6.6067, despite its worse overall MAE.
Version 2 therefore does not improve every aspect of the old model.

These are development results on the previously evaluated station partitions.
The choices for each arm were made using source validation, and every arm and
partition is retained in the report. The smallest test MAE is not used to
retroactively choose a single winning model.

## Scientific interpretation and next model step

The new fusion substantially reduces the earlier spatial-transfer loss. Most
of that recovery comes from staying close to the strong environmental expert.
The sparse support observations also contain some time-varying correction
information, but the frozen GRU representation has not extracted more of it
than the matched forest representation.

The next useful training experiment is **episodic support-conditioned residual
learning across source stations**. Keep the environmental base fixed, simulate
an unmonitored source station, give the temporal adapter K source support
observations, and train its small representation/head on the remaining source
query residuals. This directly trains the representation for adaptation;
unsupervised PCA in the present experiment only retains directions of feature
variance. Keep the matched tree adapter and constant calibration as controls.
This is a research hypothesis motivated by the present comparison, not a
demonstrated explanation of its remaining errors.

## Reproduction

```bash
uv run python scripts/run_ladder.py --experiment unified-doc-spatial-v2
uv run python scripts/analyze_unified_doc_spatial_v2.py
uv run python scripts/verify_unified_doc_spatial_v2.py
```

The runner verifies and reuses completed outputs. Source expert packages and
the cached full-grid `representations.npz` files remain local artifacts;
the runner can regenerate the latter from the saved experts. Query predictions,
fitted fusion/adapter/projector parameters, execution snapshot, source bindings,
and analysis tables are retained with this experiment.

Verification completed:

- All 9 products have complete identities and reproduce their predictions
  bitwise after reloading fusion, adapter and cached representation states.
- The matched constant ExtraTrees predictions equal version 1 at every K.
- 528 tests passed; 2 were skipped under their existing local-data conditions.
- Ruff passed; the existing artifact audit exited successfully with its
  previously documented historical exceptions unchanged.

See `analysis/findings.md` for all comparisons and intervals, `k_curves.csv`
for the full curves, and `fusion_choices.csv` / `adapter_choices.csv` for the
source-selected mechanisms.
