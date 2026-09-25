# Conditional-fusion oracle ceiling (exploratory)

This is a post-hoc design diagnostic using the frozen H2X K=5 and
`analytic_blend` products. Target query labels were opened intentionally. The
results are not a primary endpoint and did not select a reported model.

## What it shows

The fixed H2X route is not the ceiling. A convex blend has room to adapt:

- DOC HUC6 `101900`: oracle H2X weight 0.30, reducing MAE from 1.98 for the
  baseline and 2.30 for H2X to 1.90.
- DOC HUC6 `102701`: oracle weight 0.00, selecting the ecology baseline and
  removing the large H2X harm in this exploratory comparison.
- pH: oracle weights vary from 0.15 to 0.90 across HUC6 tasks.
- Specific conductance: H2X is favored in three basins, while the oracle
  weight is 0.35 in HUC6 `103001` and 0.75 in `102701`.

The fixed analyte-level weights are 0.60 for DOC, 0.60 for pH, and 0.95 for
specific conductance. This heterogeneity is exactly the signal a source-only
reliability gate should learn.

## Interpretation

The result motivates a conditional-fusion feasibility experiment. It does not
prove that a gate can recover the oracle weights without target labels. The
next experiment must estimate the gate entirely from source pseudo-target
episodes and evaluate it once on the frozen target queries.

