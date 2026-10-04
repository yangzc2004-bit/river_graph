# Product roles

Each of nine `runs/split{142,143,144}_seed{42,43,44}` packages contains:

- `readout.pt` / `readout.json`: exact initial P0, selected128-parameter D,
  source/validation protocol, full traces, selected epoch and episode schedules.
- `readout_trace.csv`: source training and held-validation checkpoint losses;
  these are not target transfer metrics.
- `source_training.npz`: station identities, explicit source/validation masks
  and source native baseline. Baseline cells outside the source mask are NaN.
  Only the forest component is OOF.
- `input_definition.json`: dimensions and content identities of reconstructed
  source/full hidden states and source38-channel head inputs. Hidden states
  can be reconstructed from bound original/selected expert checkpoints.
- `representations.npz`: constant, legacy, refreshed_fixed and
  refreshed_learned bases, flat[station×month,2]; both refreshed raw bases and
  station anchor mean/RMS/scale/floor statistics; fixed calendar anchors.
- `basis_definition.json`: exact initial/selected64×2 maps and unchanged
  retrospective anchor-normalization definition.
- `adapters.json` / `mixers.json`: unchanged source-validation alpha/ridge and
  ecological mixture selection procedures for each basis.
- `source_validation.csv`: direct/integrated K0/1/3/5 validation MAE. It is
  separate from the four-loss readout checkpoint objective.
- `predictions.parquet`: eight direct/integrated models and all four K values
  on the same target queries. Only reserved support labels enter adaptation.
  `y_true` is added after all training and calibration selections.
- `full_grid.parquet`: byte copy of the parent native grid,233,478 rows. It
  includes previous memory-experiment components; it is not a new K-adapted
  full-grid imputation product. K-dependent results are in query predictions.
- `control_checks.json`, sidecars and `complete.json`: reproducible product
  identity and exact parent-control checks.

Raw hidden-state windows obey causal visibility. The32 calendar anchors span
the record, so the normalized support basis is a retrospective feature product.
K1 shape correction is structurally inactive; its exact equality to prior
predictions is a control, not a sample-size statement.

`analysis/` contains all eight specified contrasts, all model/K curves,
tail/ordinary errors, detection diagnostics, seed/partition/station consistency,
source-selection traces and source/output bindings. `verification/` contains
independent replay, including deterministic readout refitting.

`_smoke*` directories are engineering checks and excluded from scientific
aggregation. Earlier versions remain unchanged. No learned model is promoted
over the existing main candidate by this experiment.
