# Analysis execution

The25-package verifier is run before analysis. It replays every source role,
candidate, initial weight, whole-grid neural/memory component, support adapter
and attention diagnostic. The optional analyzer flag `--reuse-replay-report`
then verifies the complete report population/runtime and all bound product
files, avoiding a second identical full-grid inference pass. No model,
prediction, numerical endpoint, bootstrap or aggregation rule changes.

The original fitting-time analyzer and verifier snapshots remain. The current
analysis source is hashed in `analysis/sources.json`; training sources are not
edited during verification or analysis.
