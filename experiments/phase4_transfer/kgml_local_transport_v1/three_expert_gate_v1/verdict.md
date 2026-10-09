# DOC three-expert feature gate

A linear softmax gate was fitted separately on each family validation split in log1p space and evaluated once on the five-seed terminal ensemble.

           family  w_rf_local_mean  w_rf_context_mean  w_residual_mean  w_rf_local_sd  w_rf_context_sd  w_residual_sd
    e1_r20_seed42         0.223133           0.531051         0.245816       0.066346         0.102170       0.102132
       e2a_strict         0.231853           0.306226         0.461921       0.181361         0.257560       0.164615
      e2b_partial         0.240371           0.311950         0.447679       0.191566         0.261064       0.162458
e3_spatial_seed42         0.173022           0.643205         0.183773       0.047010         0.102911       0.055980

## Pooled test

            model       n      mae
         residual 11046.0 1.607890
       rf_context 11046.0 1.592266
         rf_local 11046.0 1.686240
three_expert_gate 11046.0 1.520107
