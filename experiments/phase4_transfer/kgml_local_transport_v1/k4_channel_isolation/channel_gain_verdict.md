# KGML channel-isolation verdict

Positive gain means the candidate message channel has lower error.
`mode_vs_null` compares with the exact zero-message null; `mode_vs_all` compares with K2's all-input message branch.

         mode              mask   comparison  n_cells  n_stations  message_mae  null_mae  all_input_mae  gain_mae    ci_low   ci_high  gain_pct_vs_null
  target_only        e2a_strict mode_vs_null     2223          69     1.163416  1.207908       1.093137  0.044491  0.029392  0.061364          3.683349
  target_only        e2a_strict  mode_vs_all     2223          69     1.163416  1.207908       1.093137 -0.070279 -0.106304 -0.039755         -5.818230
  target_only e3_spatial_seed42 mode_vs_null     2531          43     2.957818  2.974866       2.958085  0.017047  0.003150  0.031886          0.573048
  target_only e3_spatial_seed42  mode_vs_all     2531          43     2.957818  2.974866       2.958085  0.000267 -0.000924  0.001888          0.008963
hydro_ecology        e2a_strict mode_vs_null     2223          69     1.152325  1.207908       1.093137  0.055582  0.036112  0.077943          4.601521
hydro_ecology        e2a_strict  mode_vs_all     2223          69     1.152325  1.207908       1.093137 -0.059188 -0.090377 -0.033164         -4.900058
hydro_ecology e3_spatial_seed42 mode_vs_null     2531          43     2.956946  2.974866       2.958085  0.017920  0.003306  0.034302          0.602384
hydro_ecology e3_spatial_seed42  mode_vs_all     2531          43     2.956946  2.974866       2.958085  0.001139 -0.000778  0.003153          0.038298

The null comparison identifies information retained by a channel set. The all-input comparison identifies information lost when the message input is restricted.
