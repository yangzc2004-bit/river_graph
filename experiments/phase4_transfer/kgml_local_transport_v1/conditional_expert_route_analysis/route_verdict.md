# Conditional expert route

Each family selects an expert from validation cells only.

             mask     chosen_arm  val_rf_context  val_residual_nomsg  test_rf_context  test_residual_nomsg  test_routed
    e1_r20_seed42     rf_context        1.315521            1.355292         1.451846             1.493502     1.451846
      e2b_partial residual_nomsg        1.506668            1.337256         1.060976             0.809362     0.809362
       e2a_strict residual_nomsg        1.506668            1.337256         1.088988             0.981643     0.981643
e3_spatial_seed42     rf_context        1.239786            1.357822         2.683758             2.958316     2.683758

## Pooled test

        system      mae     n
        routed 1.536073 11046
    rf_context 1.598177 11046
residual_nomsg 1.616006 11046
