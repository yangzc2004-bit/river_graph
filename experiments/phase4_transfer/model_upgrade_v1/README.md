# EcoHydroGraph model-upgrade track

This directory contains the post-T9 model-improvement experiments. It does
not replace the frozen temporal benchmark under
`experiments/phase4_transfer/temporal_h2x_v1/`.

## U0 frozen reference

The reference model is the five-seed EcoHydroGraph matched comparison from
T8/T9: a two-layer ecology-aware transport graph encoder followed by a
12-month causal GRU, trained separately for DOC, pH, and specific conductance.
The reference uses the frozen ST357 dataset, target-specific masks, and the
existing endpoint definitions.

## U1 convergence pilot

U1 tests whether the 10-epoch matched budget constrains the temporal model.
It uses DOC and specific conductance, strict temporal extrapolation and
unmonitored stations, seeds 42--44, and 10/30/60 epoch budgets. This is a
diagnostic pilot; it does not alter T8/T9 claims.

## Upgrade order

1. U1 convergence and training-budget diagnosis.
2. U2 explicit visible-observation network context.
3. U3 temporal random-forest baseline.
4. U4 validation-selected RF/graph blend to test complementary information.
5. U5 one additional missingness family, then stop or promote the blend.

U1 established that 30 epochs is materially better than the original
10-epoch matched budget. U2 context results are mixed and incomplete. U3
Temporal RF is currently the strongest tested model on the targeted pilot
families. U4 closes part of this gap on strict temporal extrapolation: a
validation-selected convex blend improves all three seeds for DOC and
specific conductance, while its clustered intervals remain compatible with
zero. The first spatial holdout configuration selected pure RF and showed no
blend gain, but the remaining configurations were not completed; this is a
partial exploratory result, not a spatial gate. Deeper architectures and
broad searches are not authorized.

All products must carry dataset, mask, configuration, and runtime identities.
