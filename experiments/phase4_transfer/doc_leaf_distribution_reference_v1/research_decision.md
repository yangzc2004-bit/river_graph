# Research decision: fixed-leaf concentration distributions

## Source comparison

Nine packages (source partitions142/143/144 × seeds42/43/44) retain the exact
log-fitted ExtraTrees partitions and47 inputs. Source concentrations alone
populate their leaf distributions. The two readouts share that distribution:
native mean and its conditional median at the fixed probability0.5.

| Procedure | K0 MAE (mg/L) | Q90 MAE (mg/L) | Signed bias (mg/L) |
|---|---:|---:|---:|
| Retained complete model |1.769053|9.075420|-0.638608|
| Retained log ExtraTrees |1.866362|9.394458|-0.611014|
| Fixed-leaf native mean |2.131545|8.858379|+0.183859|
| Fixed-leaf conditional median |1.853846|9.541649|-0.783988|

Partition-equal paired station analysis uses5,000 draws. Conditional median
improves its matched native mean13.03%[6.94%,19.57%], all three partitions,
but its gain over the retained log forest is only0.67%[-0.84%,2.34%]: seven
of nine packages, all three training-seed averages and two of three partitions
are positive. This is a small development signal with unresolved uncertainty.
Its Q90 error worsens1.57%[0.62%,2.87%] against the log forest.

Against the actual complete model, the bare median reference is4.79% worse
[2.37%,7.31%], with Q90 also worse. Replacing a complete neural procedure
with this bare reference would therefore be an error. The source distribution
mean again shows the concentration-tail versus central-error trade-off.

## Decision and follow-up

Do not deploy either reference and do not claim that the reference-improvement
condition has been established. Keep the retained complete model as the current
best procedure. The median's modest central improvement justifies one exploratory
integration in the existing ecology/GNN/GRU native-residual pipeline: determine
whether its tail-aware neural correction can retain central gains and repair
the median's tail loss. This follow-up has its own source-only plan; it is not
geographical confirmation or a retroactive declaration that this study passed.
No quantile scan, region-specific winner selection or validation arm blend is
introduced. The unsuccessful native-mean reference will not receive a neural fit.

Nested source station-held-out forests already saved by the reference-trajectory
study can supply the same partition geometry for OOF distributional residuals,
after verifying their complementary station roles and fitted state. Their source
leaf concentrations must exclude each held fold as well. This avoids repeating
45 identical forest fits while preserving genuine OOF residual training.

## Evidence and reproducibility

Original predictions and tree states are unchanged. Source concentrations,
normalized leaf weights, parameter settings and saved predictions replay exactly
in all nine packages. The conditional median is verified against weighted
absolute loss on a hand-checkable example; arbitrary new query rows and saved
models are tested. Distribution weights follow the established forest leaf
mixture idea in [Meinshausen (2006)](https://www.jmlr.org/papers/volume7/meinshausen06a/meinshausen06a.pdf),
using ExtraTrees partitions here rather than claiming a new forest algorithm.

Analysis and figures are reproduced by
`scripts/analyze_doc_leaf_distribution_reference_v1.py --bootstrap-draws 5000`
and `scripts/plot_doc_leaf_distribution_reference_v1.py` through `uv run python`.
PNG/PDF/SVG were inspected.962 tests pass, two skip; Ruff and historical audit
pass. Large distribution/forest states remain local. Related small files are
saved in the whitelist because the current workspace policy denies Git index
writes. Existing geographical, external and paper claims are unchanged.
