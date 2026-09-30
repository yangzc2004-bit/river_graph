# Spatial context residual pilot verdict

The RF-context arm is the explicit current-month network baseline. The context_nomsg arm is an exact zero-message null: its prediction equals RF-context. Positive gain means a graph candidate has lower absolute error.

## Mean test MAE

                      arm  n_cells  n_stations      mae
               rf_context     2531          43 2.669285
   residual_context_nomsg     2531          43 2.669285
residual_context_msgdelta     2531          43 2.668822
    residual_context_both     2531          43 2.670137

## Paired station bootstrap

                candidate     reference  mae_candidate  mae_reference  gain_mae  gain_pct    ci_low  ci_high
residual_context_msgdelta    rf_context       2.668822       2.669285  0.000463  0.017355 -0.000965 0.001962
residual_context_msgdelta context_nomsg       2.668822       2.669285  0.000463  0.017355 -0.000953 0.001970
    residual_context_both    rf_context       2.670137       2.669285 -0.000851 -0.031895 -0.007164 0.006079
    residual_context_both context_nomsg       2.670137       2.669285 -0.000851 -0.031895 -0.007258 0.005721

## Upstream support strata

   support_group                       arm  n_cells  n_stations      mae
            none                rf_context     1510          42 3.298711
            none    residual_context_nomsg     1510          42 3.298711
            none residual_context_msgdelta     1510          42 3.297974
            none     residual_context_both     1510          42 3.301000
visible_upstream                rf_context     1021          15 1.738402
visible_upstream    residual_context_nomsg     1021          15 1.738402
visible_upstream residual_context_msgdelta     1021          15 1.738343
visible_upstream     residual_context_both     1021          15 1.737127

## Interpretation

The upstream residual branch is the primary spatial candidate. Its gain must be read together with the support strata: stations without a visible upstream source cannot receive a target-message correction. The both-direction branch is a spatial interpolation diagnostic because held-out stations can have visible downstream neighbors; it is not a one-way transport claim.
