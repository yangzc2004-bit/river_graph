# KGML source-isolation verdict

Estimator: cell-weighted MAE averaged over training seeds; paired station bootstrap. This is not the error of an averaged prediction ensemble.

## Paired station bootstrap

             mask  mae_message  mae_null  gain_mae  gain_pct   ci_low  ci_high  message_delta_abs_mean  null_delta_abs_max
       e2a_strict     1.093137  1.207908  0.114770  9.501579 0.069632 0.167503                0.044944                 0.0
e3_spatial_seed42     2.958085  2.974866  0.016781  0.564085 0.003242 0.031430                0.004832                 0.0

## Interpretation

The message-only arm is compared with an identical empty-message null. The null is structurally RF-local, not an independently learned local residual. A positive interval establishes an improvement over that base in this design; it does not by itself distinguish correct upstream correspondence from topology-dependent bias correction. K1's learned no-message arm remains the stronger comparison. The null graph delta must be exactly zero by construction.
