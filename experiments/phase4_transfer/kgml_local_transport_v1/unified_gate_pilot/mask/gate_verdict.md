# Unified observable-feature gate pilot

The gate was fit on validation errors and evaluated on terminal test predictions.

Feature set: `mask`.

feature_set              mask    n  gate_mean_local  rf_context_mae  residual_nomsg_mae  gate_mae
       mask     e1_r20_seed42 4514         0.457914        1.449039            1.490756  1.427504
       mask       e2b_partial 1778         0.564206        1.057108            0.802976  0.820021
       mask        e2a_strict 2223         0.546326        1.083995            0.976260  0.938422
       mask e3_spatial_seed42 2531         0.316393        2.679058            2.954307  2.704261

## Pooled test

        system      mae
          gate 1.523841
    rf_context 1.594326
residual_nomsg 1.611854

## Station-clustered gain

feature_set              mask candidate       baseline  gain_mae     ci_lo    ci_hi
       mask     e1_r20_seed42      gate     rf_context  0.035809 -0.003307 0.071882
       mask     e1_r20_seed42      gate residual_nomsg  0.073773  0.023167 0.142055
       mask       e2b_partial      gate     rf_context  0.234995  0.127328 0.361273
       mask       e2b_partial      gate residual_nomsg -0.041233 -0.154760 0.042714
       mask        e2a_strict      gate     rf_context  0.156937  0.019598 0.317772
       mask        e2a_strict      gate residual_nomsg  0.003745 -0.119959 0.114110
       mask e3_spatial_seed42      gate     rf_context -0.030244 -0.102254 0.038255
       mask e3_spatial_seed42      gate residual_nomsg  0.264063  0.124725 0.401674
