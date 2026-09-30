# Unified observable-feature gate pilot

The gate was fit on validation errors and evaluated on terminal test predictions.

Feature set: `observable`; objective: `direct`.

feature_set objective              mask    n  gate_mean_local  rf_context_mae  residual_nomsg_mae  gate_mae
 observable    direct     e1_r20_seed42 4514         0.514831        1.449039            1.490756  1.429502
 observable    direct       e2b_partial 1778         0.540636        1.057108            0.802976  0.824573
 observable    direct        e2a_strict 2223         0.597641        1.083995            0.976260  0.929530
 observable    direct e3_spatial_seed42 2531         0.549529        2.679058            2.954307  2.774268

## Pooled test

        system      mae
          gate 1.539641
    rf_context 1.594326
residual_nomsg 1.611854

## Station-clustered gain

feature_set objective              mask candidate       baseline  gain_mae     ci_lo    ci_hi
 observable    direct     e1_r20_seed42      gate     rf_context  0.035391 -0.010876 0.079540
 observable    direct     e1_r20_seed42      gate residual_nomsg  0.073355  0.028176 0.132626
 observable    direct       e2b_partial      gate     rf_context  0.236344  0.127176 0.369335
 observable    direct       e2b_partial      gate residual_nomsg -0.039884 -0.142636 0.038120
 observable    direct        e2a_strict      gate     rf_context  0.166105  0.018713 0.334439
 observable    direct        e2a_strict      gate residual_nomsg  0.012913 -0.103436 0.115343
 observable    direct e3_spatial_seed42      gate     rf_context -0.093574 -0.213841 0.019267
 observable    direct e3_spatial_seed42      gate residual_nomsg  0.200733  0.102943 0.295254
