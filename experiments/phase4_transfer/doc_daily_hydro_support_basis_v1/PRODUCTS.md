# Updated recurrent states for station-support adaptation

This is postprocessing of the completed daily-memory experiment. Neural
weights, forests, native bases and ecological memory profiles stay fixed.
The new representation connects each selected recurrent state to the same
two-dimensional readout already used by the station-support adapter.

## Product names

Experts are `off`, `current_only` and `full_history`, inherited from the parent.
Each has direct and ecological-integrated predictions with three support bases:

- `constant`: no temporal shape correction;
- `legacy`: the previous v4 recurrent basis;
- `refreshed`: the selected current expert hidden state through the fixed v4
  readout and calendar-anchor normalization.

Names follow `expert_basis` and `expert_integrated_basis`, for example
`full_history_integrated_refreshed`. There are 18 products at each of
K=0/1/3/5. Direct and integrated fits use the same source-validation grids.
K0 native predictions and ecological mixing are unchanged.

## Package files

- `representations.npz`: unchanged flat `constant` and `legacy` arrays, and
  each expert's `*_refreshed[N,T,2]`, `*_raw_basis[N,T,2]`, anchor means,
  RMS, applied scale and floor flags; one shared anchor-month array;
- `basis_definition.json`: unchanged readout, anchor count and scale floor;
- `adapters.json` / `mixers.json`: validation-selected direct and integrated
  support states;
- `source_validation.csv`: selected scores for all 18 model/K curves;
- `full_grid.parquet`: byte-identical parent native component grid;
- `predictions.parquet`: station/month query predictions with matching support
  budgets and component columns;
- `legacy_checks.json`: matched constant/legacy parent outputs at every K.

The parent's selected checkpoints and original v4 readout checkpoint are
linked from `config.json`. Refresh inference uses
`river_graph.models.support_basis_refresh.refresh_support_basis`; it performs
no training or PCA fit and preserves the checkpoint weights.

Raw projected states use causal 12-month windows. The subsequent station
normalization uses 32 label-free anchors across the record, following the
existing retrospective station-support protocol. It may consult hydrologic
features after a query month. Positive-K support is also retrospective; these
products are historical reconstruction rather than an online forecast.

```bash
uv run python scripts/run_ladder.py --experiment doc-daily-hydro-support-basis-v1
uv run python scripts/diagnose_doc_daily_hydro_support_basis_v1.py
uv run python scripts/verify_doc_daily_hydro_support_basis_v1.py
uv run python scripts/analyze_doc_daily_hydro_support_basis_v1.py
```
