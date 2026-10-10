# Partial execution preserved; superseded by v2

Partition142 / seeds42,43,44 completed (18 scalar readout fits and15 river fits).
At partition143 / seed42, an independent source-anchor replay assertion failed
BEFORE that package's new fits. Export had omitted nonnegative clipping of the
local temporal prediction before ecological-memory blending. Two source rows
out of15,241 differed, with a maximum discrepancy of0.140793mg/L.

The three completed packages, original runtime/source snapshot, fitting caches
and failure log remain untouched. They are engineering records, not a complete
scientific comparison. No partial validation result selected the new arms or
hyperparameters. v2 fixes both the readout combination and training-loss order,
adds an explicit clipping test and reruns all nine packages from scratch.
The complete scientific results live in `../doc_river_readout_crossfit_v2/`.
