# DOC three-expert validation stack

Weights were selected on each family validation split using a 0.01 simplex grid, then applied once to the five-seed terminal predictions.

           family  w_rf_local  w_rf_context  w_residual  val_mae
    e1_r20_seed42         0.0          0.60        0.40 1.289122
       e2a_strict         0.0          0.21        0.79 1.313218
      e2b_partial         0.0          0.21        0.79 1.313218
e3_spatial_seed42         0.0          1.00        0.00 1.237298

## Pooled test

     model       n      mae
  residual 11046.0 1.607890
rf_context 11046.0 1.592266
  rf_local 11046.0 1.686240
     stack 11046.0 1.518129
