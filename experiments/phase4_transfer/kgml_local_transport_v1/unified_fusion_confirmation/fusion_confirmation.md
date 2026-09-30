# Five-seed DOC unified fusion confirmation

Both frozen experts use seeds 42--46. The conditional gate is fit separately per missingness family; the global gate is one model across all families. Both gates use validation labels only and are evaluated once on terminal test predictions.

## Conditional gate

             mask    n  alpha_init  alpha_mean  alpha_sd   rf_mae  local_mae  fusion_mae
    e1_r20_seed42 4514        0.39    0.449866  0.090123 1.450394   1.490635    1.429221
      e2b_partial 1778        0.75    0.662767  0.104836 1.053114   0.802349    0.800561
       e2a_strict 2223        0.75    0.675888  0.111649 1.070204   0.977766    0.894529
e3_spatial_seed42 2531        0.00    0.019781  0.000759 2.682570   2.936342    2.681961

## Conditional station-clustered effects

             mask       baseline  gain_mae     ci_lo    ci_hi
    e1_r20_seed42     rf_context  0.033714 -0.001459 0.065984
    e1_r20_seed42 residual_nomsg  0.071606  0.015643 0.138536
      e2b_partial     rf_context  0.221227  0.141420 0.313785
      e2b_partial residual_nomsg -0.050541 -0.218267 0.064919
       e2a_strict     rf_context  0.167359  0.060780 0.279986
       e2a_strict residual_nomsg  0.024469 -0.172085 0.187825
e3_spatial_seed42     rf_context  0.001082 -0.003828 0.005808
e3_spatial_seed42 residual_nomsg  0.291597  0.095907 0.486517

## Global gate

             mask    n  alpha_mean  alpha_sd   rf_mae  local_mae  fusion_mae
    e1_r20_seed42 4514    0.515679  0.137018 1.450394   1.490635    1.423396
      e2b_partial 1778    0.637152  0.094873 1.053114   0.802349    0.814496
       e2a_strict 2223    0.687580  0.092202 1.070204   0.977766    0.908490
e3_spatial_seed42 2531    0.436196  0.128868 2.682570   2.936342    2.742893

## Pooled

            system      mae
conditional_fusion 1.507467
     global_fusion 1.524101
        rf_context 1.592266
    residual_nomsg 1.607890
