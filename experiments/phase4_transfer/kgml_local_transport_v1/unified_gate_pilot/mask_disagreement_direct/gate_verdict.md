# Unified observable-feature gate pilot

The gate was fit on validation errors and evaluated on terminal test predictions.

Feature set: `mask_disagreement`; objective: `direct`.

      feature_set objective              mask    n  gate_mean_local  rf_context_mae  residual_nomsg_mae  gate_mae
mask_disagreement    direct     e1_r20_seed42 4514         0.545292        1.449039            1.490756  1.421408
mask_disagreement    direct       e2b_partial 1778         0.626178        1.057108            0.802976  0.815781
mask_disagreement    direct        e2a_strict 2223         0.661755        1.083995            0.976260  0.917688
mask_disagreement    direct e3_spatial_seed42 2531         0.488902        2.679058            2.954307  2.757156

## Pooled test

        system      mae
          gate 1.528615
    rf_context 1.594326
residual_nomsg 1.611854

## Station-clustered gain

      feature_set objective              mask candidate       baseline  gain_mae     ci_lo    ci_hi
mask_disagreement    direct     e1_r20_seed42      gate     rf_context  0.047005  0.024114 0.068109
mask_disagreement    direct     e1_r20_seed42      gate residual_nomsg  0.084969  0.015799 0.183860
mask_disagreement    direct       e2b_partial      gate     rf_context  0.219989  0.130820 0.311596
mask_disagreement    direct       e2b_partial      gate residual_nomsg -0.056239 -0.209182 0.047196
mask_disagreement    direct        e2a_strict      gate     rf_context  0.162321  0.050443 0.282856
mask_disagreement    direct        e2a_strict      gate residual_nomsg  0.009130 -0.157810 0.146877
mask_disagreement    direct e3_spatial_seed42      gate     rf_context -0.067883 -0.152645 0.014904
mask_disagreement    direct e3_spatial_seed42      gate residual_nomsg  0.226425  0.095506 0.353176
