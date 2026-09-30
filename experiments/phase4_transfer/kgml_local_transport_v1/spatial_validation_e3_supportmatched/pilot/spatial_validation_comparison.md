# Spatial validation comparison

The validation block is used for arm selection; the original E3 test is terminal.

## val

### Mean MAE

role                       arm  n_cells  n_stations      mae
 val                rf_context      796          20 2.526837
 val    residual_context_nomsg      796          20 2.526837
 val residual_context_msgdelta      796          20 2.527658
 val     residual_context_both      796          20 2.534959

### Paired station bootstrap

role                 candidate  reference  mae_candidate  mae_reference  gain_mae  gain_pct    ci_low   ci_high
 val residual_context_msgdelta rf_context       2.527658       2.526837 -0.000821 -0.032496 -0.001843 -0.000011
 val     residual_context_both rf_context       2.534959       2.526837 -0.008123 -0.321464 -0.014140 -0.002824

### Upstream support strata

role    support_group                       arm  n_cells  n_stations      mae
 val             none                rf_context      475          19 3.029415
 val             none    residual_context_nomsg      475          19 3.029415
 val             none residual_context_msgdelta      475          19 3.029727
 val             none     residual_context_both      475          19 3.038376
 val visible_upstream                rf_context      321           7 1.783146
 val visible_upstream    residual_context_nomsg      321           7 1.783146
 val visible_upstream residual_context_msgdelta      321           7 1.784720
 val visible_upstream     residual_context_both      321           7 1.790029

## test

### Mean MAE

role                       arm  n_cells  n_stations      mae
test                rf_context     2531          43 2.653636
test    residual_context_nomsg     2531          43 2.653636
test residual_context_msgdelta     2531          43 2.652572
test     residual_context_both     2531          43 2.651859

### Paired station bootstrap

role                 candidate  reference  mae_candidate  mae_reference  gain_mae  gain_pct    ci_low  ci_high
test residual_context_msgdelta rf_context       2.652572       2.653636  0.001064  0.040095  0.000016 0.002149
test     residual_context_both rf_context       2.651859       2.653636  0.001777  0.066952 -0.004369 0.006847

### Upstream support strata

role    support_group                       arm  n_cells  n_stations      mae
test             none                rf_context     1510          42 3.406055
test             none    residual_context_nomsg     1510          42 3.406055
test             none residual_context_msgdelta     1510          42 3.405360
test             none     residual_context_both     1510          42 3.406491
test visible_upstream                rf_context     1021          15 1.540852
test visible_upstream    residual_context_nomsg     1021          15 1.540852
test visible_upstream residual_context_msgdelta     1021          15 1.539243
test visible_upstream     residual_context_both     1021          15 1.535802
