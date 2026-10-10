# DOC river-message training update (2026-10-10)

Completed `experiments/phase4_transfer/doc_river_readout_crossfit_v2`: nine
source-development packages,54 conditional readout fits and45 river fits.
The same existing environmental-state operator was trained against errors
from either an identically refitted all-source readout or a whole-station-held
readout. Encoder/GRU, donor allocation/inputs, ecological coefficients,
observed-DOC branch and queries remain frozen historical source fits. Thus
the new readout loss is held-station, but the complete model is not OOF.

Primary MAE1.714736mg/L versus preceding upstream-state1.714729 and refitted
control1.713788. Incremental reductions are-0.0004%[-0.1375,0.1385] and
-0.0553%[-0.1752,0.0410]. High-DOC Q90 error is0.0883% worse than the previous
river model. Matched real/non-ancestor difference remains unresolved. Conditional
crossfit changed the training residual distribution but did not deliver a
reliable prediction improvement. Do not adopt this version or select a different
primary arm retrospectively; preceding candidate and released model stay fixed.

All nine input designs,54 heads and45 checkpoints replay; metrics and twelve
5,000-draw intervals independently recalculated; two scientific figures viewed.
pytest1463 passed,3 skipped; Ruff and historical audit pass. Partial v1 execution
is retained and explained: local nonnegative clipping must precede ecological
memory blending; v2 repaired the combination/loss and reran all packages.

Next: explicit confluence/storage modulation of upstream messages, compared
with the identical branch without that operator and matched non-upstream
sources. Existing path features already contain junctions, area ratios and
storage; change propagation/mixing, not add another feature list. Stay on source
roles142/143/144, keep receiving water-quality inputs hidden and local predictions
frozen. No geographic/external result is reused for tuning. No active training
process or recurring automation remains from this round.
