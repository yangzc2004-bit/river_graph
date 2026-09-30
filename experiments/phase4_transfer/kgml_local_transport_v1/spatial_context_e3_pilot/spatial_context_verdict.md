# Spatial context residual pilot verdict

The RF-context arm is the explicit current-month network baseline. The context_nomsg arm is an exact zero-message null: its prediction equals RF-context. Positive gain means a graph candidate has lower absolute error.

## Mean test MAE

                      arm  n_cells  n_stations      mae
               rf_context     2531          43 2.683758
   residual_context_nomsg     2531          43 2.683758
residual_context_msgdelta     2531          43 2.663624
    residual_context_both     2531          43 2.672454

## Paired station bootstrap

                candidate     reference  mae_candidate  mae_reference  gain_mae  gain_pct    ci_low  ci_high
residual_context_msgdelta    rf_context       2.663624       2.683758  0.020135  0.750236  0.001264 0.040152
residual_context_msgdelta context_nomsg       2.663624       2.683758  0.020135  0.750236  0.001420 0.039001
    residual_context_both    rf_context       2.672454       2.683758  0.011304  0.421209 -0.010507 0.031295
    residual_context_both context_nomsg       2.672454       2.683758  0.011304  0.421209 -0.010060 0.031058

## Upstream support strata

   support_group                       arm  n_cells  n_stations      mae
            none                rf_context     1510          42 3.290023
            none    residual_context_nomsg     1510          42 3.290023
            none residual_context_msgdelta     1510          42 3.278208
            none     residual_context_both     1510          42 3.289079
visible_upstream                rf_context     1021          15 1.787127
visible_upstream    residual_context_nomsg     1021          15 1.787127
visible_upstream residual_context_msgdelta     1021          15 1.754688
visible_upstream     residual_context_both     1021          15 1.760500

## Interpretation

The upstream residual branch is the primary spatial candidate. Its gain must be read together with the support strata: stations without a visible upstream source cannot receive a target-message correction. The both-direction branch is a spatial interpolation diagnostic because held-out stations can have visible downstream neighbors; it is not a one-way transport claim.
