# T8 matched-budget synthesis

All 300 source products passed the existing artifact-integrity audit.
144 products have declarative temporal metadata discrepancies; see training_metadata_ledger.csv. Historical sidecars are unchanged.
264/300 runs reached their maximum epoch count. These experiments establish a budget-specific comparison, not convergence.

## Five-seed temporal versus snapshot

| Analyte | Mask | Snapshot MAE | Temporal MAE | Reduction | Station-cluster delta CI | Month-cluster delta CI |
|---|---|---:|---:|---:|---|---|
| doc | e1_r20_seed42 | 4.285 | 2.625 | 38.75% | [-1.894, -1.415] | [-1.757, -1.564] |
| doc | e2a_strict | 2.679 | 1.417 | 47.12% | [-1.722, -0.7627] | [-1.355, -1.159] |
| doc | e2b_partial | 2.679 | 1.413 | 47.24% | [-1.725, -0.7395] | [-1.364, -1.154] |
| doc | e3_spatial_seed42 | 6.198 | 3.876 | 37.47% | [-2.77, -1.9] | [-2.417, -2.229] |
| ph | e1_r20_seed42 | 0.3468 | 0.3346 | 3.53% | [-0.01853, -0.006968] | [-0.01365, -0.01092] |
| ph | e2a_strict | 0.313 | 0.3018 | 3.58% | [-0.01732, -0.005343] | [-0.01271, -0.009693] |
| ph | e2b_partial | 0.3121 | 0.3003 | 3.77% | [-0.01773, -0.005958] | [-0.01339, -0.0102] |
| ph | e3_spatial_seed42 | 0.2859 | 0.2844 | 0.51% | [-0.004504, 0.001904] | [-0.002615, -0.0002858] |
| spec_conductance | e1_r20_seed42 | 683.2 | 374.8 | 45.14% | [-345.1, -272.8] | [-318.7, -297.7] |
| spec_conductance | e2a_strict | 572.7 | 247.9 | 56.71% | [-396.2, -247.1] | [-333.1, -317] |
| spec_conductance | e2b_partial | 568.9 | 243.5 | 57.19% | [-396.9, -245.3] | [-334.9, -315.5] |
| spec_conductance | e3_spatial_seed42 | 996.4 | 609.4 | 38.84% | [-443.2, -321.8] | [-397.1, -376.5] |

Delta = temporal minus snapshot; negative favors temporal.

## Interpretation and next decision

- Keep lookback=12 as the working configuration. The 1/3/6/12 curve shows most of the pH and conductance improvement by six months.
- Window length changes both available history and GRU computation. Repeated-current-month input over 12 steps has not been tested, so historical information alone is not identified by this comparison.
- Reversed history preserves a fixed, learnable order; it does not prove time ordering irrelevant. Hydro-only removes current and past target channels and cannot isolate the ecological contribution.
- Both cluster intervals are conditional on the chosen mask and fixed seed set. They do not estimate joint spatial-temporal dependence.
- Repeated inspection of these test sets makes this a descriptive synthesis. Do not use it as an untouched confirmatory test.
- Before a major training expansion, inspect validation learning curves and optimizer-update counts. Additional seeds do not fix undertraining.
- Seed dispersion is not a calibrated predictive interval; this stage makes no new uncertainty or active-sampling claim.

Reproduce: `.venv/bin/python scripts/analyze_temporal_synthesis.py` (or `uv run python scripts/analyze_temporal_synthesis.py` where uv is available).
