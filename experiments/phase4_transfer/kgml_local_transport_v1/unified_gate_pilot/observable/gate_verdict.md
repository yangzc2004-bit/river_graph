# Unified observable-feature gate pilot

The gate was fit on validation errors and evaluated on terminal test predictions.

Feature set: `observable`.

feature_set              mask    n  gate_mean_local  rf_context_mae  residual_nomsg_mae  gate_mae
 observable     e1_r20_seed42 4514         0.461992        1.449039            1.490756  1.429198
 observable       e2b_partial 1778         0.495867        1.057108            0.802976  0.836413
 observable        e2a_strict 2223         0.534279        1.083995            0.976260  0.940115
 observable e3_spatial_seed42 2531         0.429172        2.679058            2.954307  2.727929

## Pooled test

        system      mae
          gate 1.532935
    rf_context 1.594326
residual_nomsg 1.611854

## Station-clustered gain

feature_set              mask candidate       baseline  gain_mae     ci_lo    ci_hi
 observable     e1_r20_seed42      gate     rf_context  0.033387 -0.010668 0.074490
 observable     e1_r20_seed42      gate residual_nomsg  0.071350  0.024836 0.132632
 observable       e2b_partial      gate     rf_context  0.221006  0.121703 0.337667
 observable       e2b_partial      gate residual_nomsg -0.055222 -0.174984 0.032294
 observable        e2a_strict      gate     rf_context  0.154415  0.023213 0.305610
 observable        e2a_strict      gate residual_nomsg  0.001223 -0.130415 0.116086
 observable e3_spatial_seed42      gate     rf_context -0.056090 -0.152647 0.037217
 observable e3_spatial_seed42      gate residual_nomsg  0.238217  0.119559 0.352642
