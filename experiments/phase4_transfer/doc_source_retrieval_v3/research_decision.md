# Robust donor response: source-validation result

Nine source-only development packages completed and replayed. Replacing the
ridge-10 least-squares donor profile with smooth-native-MAE population-shrunk
season/hydro profiles yields K0 MAE **1.828808 mg/L**, versus **1.829579** for the
current complete model: **0.042%** reduction, 95% paired station-bootstrap
interval **−0.145–0.256%**. Only one of three partition means improves.

Q90 MAE changes from **9.225863** to **9.201238 mg/L**. Uniform source weights and
donor-value removal do not show a reliable contrast advantage: the respective
overall gain intervals are **−0.069–0.163%** and **−0.056–0.193%**. Better source
profile recovery in the synthetic check does not imply useful spatial transfer
in these real data. Keep the result; do not promote it to HUC4 confirmation.

Nested source-only OOF residual arrays are now retained locally so a future
representation can reuse the same forest errors. Source/validation roles and
the fixed current comparator remain intact; outer target DOC has not been scored.

The next development experiment isolates source/query input availability in
the environmental tree base. Previous source tree rows could use local DOC
lags, whereas entirely unmonitored queries have none. Train the same strong
trees on station-fold-hidden source rows, then use source validation to decide
whether the existing ecology/GRU residual should be rebuilt on that base.
This changes the training information regime, not model depth or attention size.
