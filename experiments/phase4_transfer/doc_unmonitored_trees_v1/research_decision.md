# Station-hidden source tree training

Nine source-only development packages completed. Training the same trees on
station-fold-hidden source rows yields K0 MAE **1.866362 mg/L**, versus **1.963925**
for the old matched-input tree reference: **4.968%** reduction (5,000 paired
station-bootstrap 95% interval **3.059–7.026%**). Q90 MAE changes from **9.618575**
to **9.394458 mg/L**, a **2.330%** reduction (interval **0.240–4.805%**).

The current complete ecology/GRU model still has lower MAE, **1.829579 mg/L**.
The new tree is **2.010% worse**, so do not describe the tree result as an
improvement over the complete model. Every prediction uses the same validation
DOC cells and input channels. All nine fitted tree checkpoints independently
replay within 1e-12 numerical tolerance.

The input-regime change supplies a stronger base for entirely unmonitored sites.
Next, generate station-blocked OOF predictions from this base, hiding the outer
fold from all fitted targets and covariates, then refit the **existing** ecology
encoder/observation-GRU residual. Keep depth, lookback, daily descriptors and
tail weighting fixed. Continue source-role development; no geographical or
external target results have been used to select this change.
