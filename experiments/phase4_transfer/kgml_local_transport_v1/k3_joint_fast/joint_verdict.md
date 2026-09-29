# K3 joint local--message verdict

             mask  seed  joint_mae  local_residual_mae  rf_context_mae  joint_graph_delta_sd  joint_message_delta_sd
       e2a_strict  43.0   1.024758            0.981643        1.088988              0.012994                0.010594
e3_spatial_seed42  43.0   2.956160            2.958316        2.683758              0.001468                0.001393

## Paired station bootstrap

             mask                   comparison  gain_mae    ci_low   ci_high   gain_pct
       e2a_strict   joint_error_vs_local_error -0.043115 -0.070177 -0.019930  -4.392174
       e2a_strict joint_error_vs_context_error  0.064230 -0.182580  0.271451   5.898163
e3_spatial_seed42   joint_error_vs_local_error  0.002156 -0.000674  0.004901   0.072890
e3_spatial_seed42 joint_error_vs_context_error -0.272402 -0.502878 -0.036945 -10.150010

The joint model is useful when its gain over K1 learned local residual is positive and stable in the temporal holdout. RF-context is retained as the explicit-context reference.
