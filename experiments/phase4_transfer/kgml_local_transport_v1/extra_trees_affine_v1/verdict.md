# Validation-only affine expert stack

The log-space affine stack is fit on validation predictions only. The selected variant is the lowest-validation-MAE candidate among context, residual, geometric blend, and affine stack; terminal test labels are never used for fitting or selection.

             mask  n_seeds  selected_test_mae  context_test_mae  gain_vs_context_pct
    e1_r20_seed42        5           1.436170          1.436170             0.000000
       e2a_strict        5           0.887071          1.047455            15.311813
      e2b_partial        5           0.783808          1.016471            22.889313
e3_spatial_seed42        5           2.545262          2.545262             0.000000

 rows  selected_mae_equal_seed_family  context_mae_equal_seed_family  gain_equal_seed_family_pct  selected_mae_cell_weighted  context_mae_cell_weighted
   20                        1.413077                       1.511339                    6.501644                    1.474787                   1.544514
