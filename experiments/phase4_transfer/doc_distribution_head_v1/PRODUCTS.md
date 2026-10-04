# Conditional DOC density-head products

## Scope

This experiment fits only a conditional residual head on the selected,
frozen daily-hydrology DOC model. Its encoder, recurrent state, forest,
ecological residual library and two support representations remain fixed.
The three station partitions and previously evaluated targets are development
data. Source training uses station-blocked forest OOF predictions plus a
source-trained neural correction; only the forest is OOF.

The Gaussian component parameters describe a distribution of log1p residuals.
They do not represent identified ecological processes. No calibrated prediction
interval or new monitoring claim is produced here.

## Run packages

`runs/split{142,143,144}_seed{42,43,44}/` contains:

- `single.pt`, `mixture.pt`: fitted linear density heads and source feature
  normalization. The selected point-correction scale is stored in each payload.
- `single.json`, `mixture.json`: initialization, parameter counts, source and
  validation distribution summaries, checkpoint choices and training traces.
- `single_trace.csv`, `mixture_trace.csv`: training batch likelihood and
  source-validation likelihood/native MAE, including epoch0 and scale0 fallback.
- `source_training.npz`: compact observed source identities and native base,
  with held-validation query identities and base. Feature identities and ordering
  are recorded in `input_definition.json`.
- `full_grid.parquet`: all 233,478 station-month cells, with original context
  and point predictions, Gaussian weights/locations/scales, conditional median
  residuals, density-head native predictions and integrated K0 predictions.
  It contains no withheld query truth.
- `predictions.parquet`: fixed held-station queries for K=0/1/3/5. Only the
  reserved support labels enter adaptation. `y_true` is attached after fitting
  and source-validation selection for final comparison.
- `adapters.json`, `mixers.json`, `source_validation.csv`: unchanged support
  adaptation and ecological integration rules, refitted on source validation.
- `point_checks.json`: exact reproduction of the retained parent point model
  across both support representations, direct/integrated routes and all K.
- configuration, timing, completion manifest and parquet sidecars: experiment
  settings and dependencies, including the frozen parent expert.

The two support representations are `constant` and the retained
`gru_tuned_anchor`. The latter uses the existing calendar anchor; positive-K
adaptation is retrospective station reconstruction. Historical support may
postdate a query. The raw recurrent encoder reads current/past covariates only.

## Point prediction and interpretation

The single Gaussian predicts its location. The mixture predicts its actual
conditional median, solved from the two-normal CDF. Neither a density mean nor
an average of component medians is substituted. A global correction scale is
selected on source-validation K0 MAE, and scale0 returns the original point
prediction exactly.

Both the density-head checkpoint and downstream adaptation choices use source
validation. Improved likelihood alone does not justify replacing the existing
point model. Ordinary DOC, high-DOC error and bias are reported with MAE.

## Reproduction

```bash
OMP_NUM_THREADS=2 uv run python scripts/run_ladder.py --experiment doc-distribution-head-v1
uv run python scripts/verify_doc_distribution_head_v1.py
uv run python scripts/analyze_doc_distribution_head_v1.py
```

`study_plan.md` records the twelve fixed comparisons. Runtime source is archived
under `code_snapshot/`. Engineering smoke packages use separate `_smoke*`
directories and do not enter the nine-package scientific analysis.
