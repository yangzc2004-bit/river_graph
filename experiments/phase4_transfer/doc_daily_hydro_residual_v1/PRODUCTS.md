# DOC daily-hydrology experiment: product guide

This experiment adds daily discharge summaries to the existing monthly DOC
residual readout. The source forest, target visibility, recurrent history,
ecological memory and support representation are unchanged.

`daily_features.npz` contains one float32 `[station, month, 8]` grid in the
dataset's exact node/month order. Its metadata describes extraction, duplicate
handling, signed-flow QC, calendar coverage, formulas and input files. It uses
no DOC labels, and no fitted normalization. All eight channels are zero outside
the frozen monthly discharge mask.

Three models use the same expanded head and hidden-feature products:

- `monthly`: eight zero channels, the fresh matched control.
- `availability`: three value channels zero; coverage/validity retained.
- `daily`: complete values and availability.

The current-month summaries enter the residual head, alongside the original
30 ecological/flow/concentration features. They do not replace the GRU input.
Current-month discharge is month-end reconstruction information; predictions
are not month-start forecasts.

Each package saves three neural checkpoints and their source-validation traces,
direct support adapters, ecological-integrated mixers, and prior references.
`full_grid.parquet` contains neural deltas, direct bases and integrated K0 bases.
`predictions.parquet` contains the final fixed-query predictions at K=0/1/3/5,
with retrospective target-station support where K>0. It is the query evaluation
product rather than a separate all-cell K-dependent reconstruction.

The eight canonical daily feature columns on product rows describe available
data for fixed strata. They are identical across arms; the monthly model's
actual eight inputs are zero, and the availability model's three values are
zero. Do not confuse saved data descriptors with the inputs of every arm.

All 20 model products are retained: two context models, six direct models,
six integrated models, and six prior encoder/ecological references. References
are copied unchanged; they do not constitute additional fits.

```bash
uv run python scripts/build_doc_daily_flow_features_v1.py
uv run python scripts/run_ladder.py --experiment doc-daily-hydro-residual-v1
uv run python scripts/analyze_doc_daily_hydro_residual_v1.py
uv run python scripts/verify_doc_daily_hydro_residual_v1.py
```
