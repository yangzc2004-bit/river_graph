# Station-regime-conditioned DOC residual readout

The existing native DOC residual improves ordinary concentrations but still
underpredicts high DOC at unseen stations. Source/validation diagnostics show
that high-DOC residuals remain large even when the context forest predicts an
ordinary concentration. A high-prediction-only correction is insufficient.

We retain the original spatial encoder and GRU-D memory, source station-fold
hidden inputs, source OOF context bases and the fixed forest. We extend only
the native residual readout with direct watershed regime and predicted DOC.
These information sources already enter earlier model components; this tests
how their interactions with recurrent state improve residual reconstruction.

## Matched neural heads

All heads receive the same 30 extra features: existing causal flow10; watershed
ecology9 and validity9; compressed standardized log1p predicted DOC and validity.
Ecology scaling uses unique source stations only. Predicted-DOC scaling uses
station-balanced selected source OOF predictions. Source loss cells use OOF
context input; validation/full grids use the full-source context forest.

1. **additive:** recurrent state plus extra features and existing state×flow.
2. **concentration:** also state×predicted-DOC, allowing concentration-dependent
   dynamic corrections.
3. **ecological:** also state×ecology, allowing station-regime-dependent dynamics.

All heads are zero-initialized, start from the same original GRU/decay, and use
30 maximum epochs, patience5, twofold source-Q90 loss weight and the same
overall validation MAE checkpoint/scale selection. No DOC upper clipping is used.

## Existing spatial adaptation

Each head is evaluated directly and integrated with the frozen v1 ecological
affine memory. Source validation chooses the K0 mixture and the existing joint
gamma/support-adapter settings at K1/3/5. The same frozen v4 GRU support basis,
nested support cells and fixed queries apply to all products. Current
interaction and ecological-v2 predictions remain unchanged references.

The experiment uses station partitions142/143/144 and seeds42/43/44. This is
development on previously examined partitions, with no outer-query choice of
mode, checkpoint or mixing weight. K-shot reconstruction is retrospective.

Report overall and log MAE, RMSE/R², Q90 MAE/bias, ordinary-concentration error,
false-high rate, partition/station gain distribution, and the complete K curve.
Compare matched additive/concentration/ecological heads before attributing a
gain to ecological dynamics. Integrated versus direct products separate the
neural-head change from ecological residual-memory mixing.

Products retain configs, model states, query/full-grid components and sidecars.
The full grid contains direct neural components and integrated K0 bases; K>0
final predictions are in predictions.parquet. Existing experiments are retained.
