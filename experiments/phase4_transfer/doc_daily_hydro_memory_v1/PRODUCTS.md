# Daily hydrologic memory: model and product guide

This iteration extends the existing ecological/self encoder plus
observation-aware GRU residual. All neural arms retain the preceding daily
model's current-month head features and source-OOF context prediction.

| Arm | Daily information entering GRU | Additional recurrent parameters |
| --- | --- | ---: |
| `off` | None; daily values enter the existing readout only | 0 |
| `current_only` | Target month only | 512 |
| `full_history` | Every valid month in the causal 12-month window | 512 |

Active modes use a bias-free, zero-initialized `Linear(8,64)` added to the
encoded monthly state before its existing GRU update. The GRU, age/support
decay, spatial/ecological parameters, residual head and source-validation
selection are trained with the same settings. Empty edges remain the current
local expert's architecture; this experiment measures hydrologic memory rather
than an additional river-message mechanism.

Two independent tree probes clone the saved selected context ExtraTrees
parameters and append matched daily input windows. Both have 147 features:
39 context features plus twelve slots with eight daily descriptors and a
history-valid indicator. `tree_current` zeros older daily slots;
`tree_history` retains them. These tree fits do not replace the frozen forest
base used to train the neural residual.

Each of nine packages contains three neural checkpoints/fit traces, two tree
checkpoints/records, the feature definition, direct support adapters and three
neural ecological-integration mixers. `full_grid.parquet` has 233,478 rows in
the original station/month order, with native components and integrated K0
bases. `predictions.parquet` contains the fixed observed query cells at all
K=0/1/3/5 and 24 model products, including copied prior daily/ecological
references. Positive K uses retrospective station support.

Product rows also describe information availability:

- `observation_age_months`: elapsed months since a visible target reading;
  −1 means no reading has ever been visible in that station's input view;
- `visible_target_history`: identifies the distinction between never-observed
  and old observations;
- `daily_numeric_valid_count`: current-month count of valid numerical descriptors;
- `daily_history_valid_months`: number of months in the causal window with any
  valid numerical daily descriptor;
- `daily_history_possible_months`: number of nonpadding months in the window.

All arms have identical descriptive columns; their actual use differs by mode.
The held-out stations have no visible target history before K-shot adaptation.
Their ages must not be interpreted as elapsed time since an actual measurement.
Same-month daily discharge is month-end reconstruction information.

```bash
uv run python scripts/run_ladder.py --experiment doc-daily-hydro-memory-v1
uv run python scripts/verify_doc_daily_hydro_memory_v1.py
uv run python scripts/diagnose_doc_daily_hydro_memory_v1.py
uv run python scripts/analyze_doc_daily_hydro_memory_v1.py
```

Load a neural checkpoint with `EncoderNativeResidual.from_payload(...)`.
Active modes require aligned `daily_history[N,T,8]` alongside the existing raw,
ecology, age, support and current-month extra inputs. The saved payload contains
both initial and selected hydrologic projection weights. `off` retains the
old checkpoint schema and behavior. Tree checkpoints use joblib and the saved
feature definitions; direct/support and integrated predictions use their saved
validation-selected adapters and mixers.
