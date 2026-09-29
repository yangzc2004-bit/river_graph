# K3 additive complementarity diagnostic

Alpha is selected from the validation query only. Test cells are used once for the rows below.

## Seed-level summary

             mask  alpha_val  val_mae_at_alpha  local_residual_mae  message_only_mae  additive_mae  rf_context_mae  additive_q90_mae
       e2a_strict   0.283333          1.334280            0.981643          0.981805      0.972957        1.088988          6.165219
e3_spatial_seed42   2.350000          1.353025            2.958316          2.942319      2.923126        2.683758          7.669157

## Paired station bootstrap

             mask                      comparison  gain_mae    ci_low   ci_high  gain_pct
       e2a_strict   additive_error_vs_local_error  0.008686 -0.002689  0.022327  0.884820
       e2a_strict additive_error_vs_context_error  0.116032 -0.113814  0.309889 10.654983
e3_spatial_seed42   additive_error_vs_local_error  0.035190  0.004616  0.069241  1.189526
e3_spatial_seed42 additive_error_vs_context_error -0.239368 -0.448994 -0.019715 -8.919138

## Interpretation

The additive arm tests whether the K1 learned local residual and the K2 upstream message correction carry complementary signal. A positive additive gain over local residual supports a joint model; a validation-selected alpha near zero means the message branch is redundant after local correction. RF-context remains a fixed reference for explicit spatial information.
