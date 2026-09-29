# KGML source-isolation verdict

Estimator: cell-weighted MAE averaged over training seeds; paired station bootstrap. This is not the error of an averaged prediction ensemble.

## Paired station bootstrap

             mask  mae_message  mae_null  gain_mae  gain_pct   ci_low  ci_high  message_delta_abs_mean  null_delta_abs_max
       e2a_strict     1.163416  1.207908  0.044491  3.683349 0.029392 0.061364                0.013945                 0.0
e3_spatial_seed42     2.957818  2.974866  0.017047  0.573048 0.003150 0.031886                0.005188                 0.0

## Interpretation

The message-only arm is compared with an identical empty-message null. The null is structurally RF-local, not an independently learned local residual. A positive interval establishes an improvement over that base in this design; it does not by itself distinguish correct upstream correspondence from topology-dependent bias correction. K1's learned no-message arm remains the stronger comparison. The null graph delta must be exactly zero by construction.
