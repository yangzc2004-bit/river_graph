# Conditional local--transport expert route

The completed KGML pilots show a strong interaction between the missingness
family and the useful model component. The best model is not the same for
temporal and spatial extrapolation.

## Evidence from the existing three-seed pilot

| missingness family | strongest arm | mean MAE |
|---|---|---:|
| temporal holdout (`e2a_strict`) | RF-local + learned local residual (`residual_nomsg`) | 0.9816 |
| temporal holdout (`e2a_strict`) | RF-context | 1.0890 |
| spatial holdout (`e3_spatial_seed42`) | RF-context | 2.6838 |
| spatial holdout (`e3_spatial_seed42`) | RF-local + learned residual (`residual_nomsg`) | 2.9583 |

The matched-budget extension confirms the split on two additional families:
the local residual expert has MAE 0.8094 on partial-time missingness (`e2b`),
whereas RF-context has MAE 1.4518 on random point missingness (`e1`) against
1.4935 for the local residual expert.

The spatial support-matched confirmation gives the same ordering: RF-context
MAE 2.6536 versus 2.6526 for the upstream residual branch on the frozen test,
with the validation block selecting the exact RF-context null. The small test
difference is not large enough to replace the spatial baseline.

## Proposed model system

Use the missingness family, which is known at prediction time, as a router:

1. **Temporal expert:** RF-local plus the learned causal local residual. It
   uses station history, ecology, hydro variables and time memory.
2. **Spatial expert:** RF-context. It uses current-month visible network
   context and remains the strongest confirmed model for held-out stations.
3. **Conditional river correction:** retain the upstream residual branch as a
   diagnostic and optional correction when visible upstream support exists;
   do not force it into the spatial expert when the support pattern is absent.

This is a missingness-aware model system, not a post-hoc average of test
predictions. The router is defined from the observation design before looking
at query labels.

## Next experiment

Run the two experts on the same three-seed, four-family matrix, then report:

- each expert separately;
- the fixed routed system;
- a single-model baseline;
- gains and losses by missingness family;
- the routed system's pooled and family-stratified MAE.

The routed system succeeds if it preserves the temporal expert's advantage in
temporal holdouts while retaining the RF-context spatial result. If routing
does not improve the pooled result, the paper should still retain the finding
that information value is missingness-dependent.
