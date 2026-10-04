# Product roles and reproduction

## Per completed run

- `config.json`, `complete.json`, parquet sidecars: configuration and bound
  input/output identities.
- `representations.npz`: unchanged two-dimensional GRU coordinates; two
  chemical or availability coordinates; their concatenated four-dimensional
  representations; auxiliary-active flags and source station identities.
- `basis_definition.json`: frozen chemical embeddings, source-only PCA
  covariance, centering, eigenvalues and projection definitions.
- `adapters.json`, `mixers.json`: source-validation-fitted calibration objects.
- `basis_selection.json`: all representation scores and the deterministic
  source-validation choices. No target query outcome selects a representation.
- `source_validation.csv`: final all-query and auxiliary-active validation
  MAE for all13 curves at K0/1/3/5.
- `predictions.parquet`: final observed-query products for the13 curves,
  including base prediction, adaptation, ecological mixture, representation,
  availability and original cell identities.
- `full_grid.parquet`: byte-identical parent native predictions and K0
  components. This is not a newly computed positive-K all-month product.
- `reference_checks.json`: legacy equality and exact K0/K1 invariance checks.
- `timing.json`: calibration execution time.

`analysis/` reports all13 curves and22 fixed contrasts. `diagnostics/` is a
source-validation-only residual-transfer diagnosis. `verification/` independently
reconstructs the source PCA, calibrators and query products. Runtime source
archives preserve the executed code. `_smoke/` is a local integration check
and is excluded from scientific analysis and committed experiment products.

## Commands

```bash
uv run python scripts/run_ladder.py --experiment doc-chemistry-support-v1
uv run python scripts/verify_doc_chemistry_support_v1.py
uv run python scripts/analyze_doc_chemistry_support_v1.py
uv run python scripts/diagnose_doc_chemistry_support_v1.py
```

Use archived execution source to regenerate a completed version after later
model code changes. Fresh execution with changed source belongs in a new root;
existing predictions and older experiment freezes are preserved.
