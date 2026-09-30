# Unified observable-feature gate pilot

The gate was fit on validation errors and evaluated on terminal test predictions.

Feature set: `mask_disagreement`.

      feature_set              mask    n  gate_mean_local  rf_context_mae  residual_nomsg_mae  gate_mae
mask_disagreement     e1_r20_seed42 4514         0.460580        1.449039            1.490756  1.424528
mask_disagreement       e2b_partial 1778         0.563200        1.057108            0.802976  0.804190
mask_disagreement        e2a_strict 2223         0.545436        1.083995            0.976260  0.907933
mask_disagreement e3_spatial_seed42 2531         0.329524        2.679058            2.954307  2.724508

## Pooled test

        system      mae
          gate 1.518580
    rf_context 1.594326
residual_nomsg 1.611854

## Station-clustered gain

      feature_set              mask candidate       baseline  gain_mae     ci_lo    ci_hi
mask_disagreement     e1_r20_seed42      gate     rf_context  0.045189  0.014487 0.074597
mask_disagreement     e1_r20_seed42      gate residual_nomsg  0.083152  0.019486 0.180406
mask_disagreement       e2b_partial      gate     rf_context  0.213938  0.139669 0.287202
mask_disagreement       e2b_partial      gate residual_nomsg -0.062290 -0.244317 0.058871
mask_disagreement        e2a_strict      gate     rf_context  0.160531  0.071257 0.255591
mask_disagreement        e2a_strict      gate residual_nomsg  0.007340 -0.193954 0.173052
mask_disagreement e3_spatial_seed42      gate     rf_context -0.069042 -0.152322 0.014615
mask_disagreement e3_spatial_seed42      gate residual_nomsg  0.225265  0.085954 0.359132
