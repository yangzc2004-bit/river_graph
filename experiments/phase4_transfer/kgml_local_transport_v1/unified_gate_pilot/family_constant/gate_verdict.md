# Unified observable-feature gate pilot

The gate was fit on validation errors and evaluated on terminal test predictions.

Feature set: `mask`; objective: `family_constant`.

feature_set       objective              mask    n  family_alpha  gate_mean_local  rf_context_mae  residual_nomsg_mae  gate_mae
       mask family_constant     e1_r20_seed42 4514          0.40             0.40        1.449039            1.490756  1.427674
       mask family_constant       e2b_partial 1778          0.74             0.74        1.057108            0.802976  0.793502
       mask family_constant        e2a_strict 2223          0.74             0.74        1.083995            0.976260  0.936246
       mask family_constant e3_spatial_seed42 2531          0.00             0.00        2.679058            2.954307  2.679058

## Pooled test

        system      mae
          gate 1.513429
    rf_context 1.594326
residual_nomsg 1.611854

## Station-clustered gain

feature_set       objective              mask candidate       baseline      gain_mae         ci_lo        ci_hi
       mask family_constant     e1_r20_seed42      gate     rf_context  3.431578e-02 -1.552031e-03 6.744682e-02
       mask family_constant     e1_r20_seed42      gate residual_nomsg  7.227968e-02  2.025763e-02 1.424586e-01
       mask family_constant       e2b_partial      gate     rf_context  2.650083e-01  1.287733e-01 4.222737e-01
       mask family_constant       e2b_partial      gate residual_nomsg -1.122011e-02 -8.460724e-02 4.179776e-02
       mask family_constant        e2a_strict      gate     rf_context  1.708025e-01 -4.032051e-03 3.759297e-01
       mask family_constant        e2a_strict      gate residual_nomsg  1.761087e-02 -6.346769e-02 8.844452e-02
       mask family_constant e3_spatial_seed42      gate     rf_context -1.478894e-17 -1.138855e-16 6.525214e-17
       mask family_constant e3_spatial_seed42      gate residual_nomsg  2.943072e-01  9.593388e-02 4.962907e-01
