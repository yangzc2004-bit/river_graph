# K6 cross-analyte KGML pilot

K1--K5 establish the DOC decomposition. K6 asks whether the same local versus
river-message split is specific to DOC or recurs for pH and specific
conductance.

## Matrix

- analytes: `ph`, `spec_conductance`;
- masks: `e2a_strict`, `e3_spatial_seed42`;
- seeds: 42, 43, 44;
- arms: `residual_nomsg` (local temporal residual) and `residual_msgdelta`
  (all-input upstream message residual);
- 8 epochs, patience 3, hidden size 64.

The target transforms remain frozen: raw-scale standardization for pH and
`log1p` for specific conductance. The same RF-local OOF base, visibility
protocol and query cells are used for both arms within each analyte.

## Primary comparison

For each analyte and missingness family, compare the station-clustered,
seed-averaged absolute error of the message branch with the local residual
branch. Positive gain means the river-message correction improves on the local
temporal correction.

The purpose is mechanism transfer, not a new model-selection sweep. A result
can show that river messages are analyte-dependent without requiring every
analyte to benefit.
