# Unified observable-feature gate pilot

The gate was fit on validation errors and evaluated on terminal test predictions.

             mask    n  gate_mean_local  rf_context_mae  residual_nomsg_mae  gate_mae
    e1_r20_seed42 4514         0.457914        1.449039            1.490756  1.427504
      e2b_partial 1778         0.564206        1.057108            0.802976  0.820021
       e2a_strict 2223         0.546326        1.083995            0.976260  0.938422
e3_spatial_seed42 2531         0.316393        2.679058            2.954307  2.704261

## Pooled test

        system      mae
          gate 1.523841
    rf_context 1.594326
residual_nomsg 1.611854
