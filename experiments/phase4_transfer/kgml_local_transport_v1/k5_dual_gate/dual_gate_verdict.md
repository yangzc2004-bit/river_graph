# K5 dual-channel gate verdict

Positive gain means the gated dual-channel model has lower error.

             mask            comparison  n_cells  n_stations  dual_mae  reference_mae  gain_mae    ci_low   ci_high  gain_pct
       e2a_strict          dual_vs_null     2223          69  1.162316       1.207908  0.045592  0.030095  0.062999  3.774454
       e2a_strict           dual_vs_all     2223          69  1.162316       1.093137 -0.069178 -0.104285 -0.039230 -6.328425
       e2a_strict   dual_vs_target_only     2223          69  1.162316       1.163416  0.001100 -0.000335  0.002839  0.094590
       e2a_strict dual_vs_hydro_ecology     2223          69  1.162316       1.152325 -0.009990 -0.014957 -0.005865 -0.866960
e3_spatial_seed42          dual_vs_null     2531          43  2.958212       2.974866  0.016654  0.003342  0.030939  0.559820
e3_spatial_seed42           dual_vs_all     2531          43  2.958212       2.958085 -0.000127 -0.001027  0.000945 -0.004290
e3_spatial_seed42   dual_vs_target_only     2531          43  2.958212       2.957818 -0.000394 -0.002210  0.001323 -0.013305
e3_spatial_seed42 dual_vs_hydro_ecology     2531          43  2.958212       2.956946 -0.001266 -0.003719  0.000772 -0.042822

The dual branch is interpreted as a predictive mixture of input channels; gate weights are not causal transport coefficients.
