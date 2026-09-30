# Unified observable-feature gate pilot

The gate was fit on validation errors and evaluated on terminal test predictions.

Feature set: `disagreement`.

 feature_set              mask    n  gate_mean_local  rf_context_mae  residual_nomsg_mae  gate_mae
disagreement     e1_r20_seed42 4514         0.456161        1.449039            1.490756  1.424476
disagreement       e2b_partial 1778         0.522493        1.057108            0.802976  0.810174
disagreement        e2a_strict 2223         0.538327        1.083995            0.976260  0.907694
disagreement e3_spatial_seed42 2531         0.405654        2.679058            2.954307  2.744191

## Pooled test

        system      mae
          gate 1.523984
    rf_context 1.594326
residual_nomsg 1.611854

## Station-clustered gain

 feature_set              mask candidate       baseline  gain_mae     ci_lo    ci_hi
disagreement     e1_r20_seed42      gate     rf_context  0.045605  0.014851 0.076188
disagreement     e1_r20_seed42      gate residual_nomsg  0.083569  0.019452 0.180044
disagreement       e2b_partial      gate     rf_context  0.208660  0.138188 0.277038
disagreement       e2b_partial      gate residual_nomsg -0.067568 -0.253500 0.055919
disagreement        e2a_strict      gate     rf_context  0.159948  0.073214 0.251462
disagreement        e2a_strict      gate residual_nomsg  0.006756 -0.199362 0.176771
disagreement e3_spatial_seed42      gate     rf_context -0.092954 -0.192643 0.006112
disagreement e3_spatial_seed42      gate residual_nomsg  0.201353  0.075187 0.318963
