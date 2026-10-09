# Fixed geographical replication of current-source attention

The source-selected candidate uses the existing ecological self encoder,
observation GRU/native residual plus two32-dimensional donor attention heads.
Source development found0.701% complete gain over the retained release and
1.041% Q90 gain. Adaptive-versus-fixed complete gain remains uncertain;
matched neural-only Q90 gain is0.449% with a positive interval. Preserve this
distinction and test all three arms together.

## Fixed design

- HUC4 roles1013/1019/0708/1030/1101 in their established source/validation/test
  assignments; training seeds42–46 (25 packages).
- Learned current innovations, learned earlier-season innovations, and fixed
  ecological-prior current innovations;75 neural fits in total.
- Same source operator, dimensions,30epochs/patience5, learning rates, tail
  weight2, reference/OOF inputs,41-column readout, initial encoder/GRU/decay and
  memory fusion as source development. No new tuning or source selection.
- Reuse250 verified double-held-fold references from the preceding innovation
  geographical version; no new forest fits. Each source training library omits
  its query fold. No target DOC/pH/conductance enters K0.
- Current donor keys and availability match across real/historical controls;
  historical values also replace the aggregate innovation readout. Fixed prior
  retains the same allocated shape with zero query/key influence.
- Save point states before evaluating test DOC or designated support. K0 main
  population uses all observed target cells. K1/3/5 and diagnostic K0 share the
  same separate query excluding all five support candidates; existing simple
  validation-selected support adapters remain unchanged.

## Analysis

Five-region equal-weight K0 MAE, Q90 error/bias/recall, station-equal MAE,
K curves, source support/ecological/hydro strata and attention diagnostics.
Use5,000 paired station bootstrap draws. Compare current attention complete
with retained complete, preceding current-source complete, fixed/historical
complete and saved strong/enriched trees. Native contrasts remain separate.
Report all five regions, including regressions and unstable tail populations.

First complete one technical1013/42 package and saved-state replay; then execute
all25 packages without score-dependent continuation or per-region/K winners.
This is retrospective ST357 geographical replication. Existing external
products do not select this model and are not rescored for tuning. Portable
release and manuscript are not replaced during fitting.
