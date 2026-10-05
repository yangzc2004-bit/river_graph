# Source response representation, version 3

Version 2 improved K0 by 0.055% on the selected development panel and did not
beat its source-value controls. This version changes only the **donor response
profile**. The contrast integration, attention size, candidate retrieval and
training/validation roles stay fixed.

Use source partitions 142/143/144, seeds 42/43/44. Regenerate the same nested
station-OOF forest predictions with 300-tree source forests. Entire outer folds
are excluded from the donor library and every inner fitted forest. Save source
OOF residual arrays for later representations. There is no outer target scoring.

Fit donor seasonal/hydro response with station-balanced smooth native MAE,
epsilon 0.05 in source residual units and fixed ridge 0.1. Each donor shrinks
toward a station-balanced population profile. Use the same six design terms:
intercept, compressed log forest prediction, seasonal sine/cosine, temperature
and log discharge. Temperature/discharge absence retains explicit key flags.
The old bank's dominant ridge-10 least-squares profile is retained in versions
1/2; it is not overwritten.

Retrieve up to 20 ecological candidates, two 32-dimensional heads, 30 epochs,
patience 5, station-balanced native MAE. The zero-initialized donor-minus-local
contrast reproduces the current predictor. Include uniform weights and source
value removal while retaining local subtraction. Current model, static memory
and matched daily-input trees remain fixed references. Retained source-trained
neural weights have fold-hidden inputs, not independently fitted OOF checkpoints.

Analyze all validation DOC cells with equal-partition K0 means and 5,000 paired
station-bootstrap draws, overall and source-Q90 tail. Source validation selects
the candidate; whole-HUC4 and external targets are reserved for later tests.
