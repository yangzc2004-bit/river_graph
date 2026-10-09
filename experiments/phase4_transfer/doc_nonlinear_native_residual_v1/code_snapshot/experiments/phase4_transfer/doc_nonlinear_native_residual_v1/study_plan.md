# A nonlinear residual readout on the retained DOC model

## Scientific change

Keep the retained station-hidden environmental tree and native mg/L residual
framework. The ecological encoder and observation-aware GRU remain unchanged.
Replace the scalar linear readout with that same linear path plus a small
nonlinear correction over its existing features:

```
existing GRU state + existing current hydro/regime features and interactions
                     |                              |
                linear path             LayerNorm -> Linear(32) -> GELU
                     |                              |
                     |                       zero Linear(1)
                     +------------------------------+
                                    |
                         signed native DOC residual
```

The new final projection starts at zero. A nonzero supplied linear path is
therefore preserved exactly at initialization. Full training starts from the
preceding candidate's initial ecological/GRU/decay states, with both output
paths zero. This is a nonlinear hydro-ecological correction on the existing
model, not a separate backbone. No dropout, gate or new input is added.

## Matched experiment

Use source-development partitions142/143/144 × seeds42/43/44, with the existing
source station-fold-hidden views, nested forest OOF residual reference and full
source-validation input views. Receiving sites have no local DOC/pH/conductance
or their history/availability features. Old geographical and external queries
do not enter model selection or scoring.

Keep30 epochs, patience5, native MAE, tail weight2, learning rates, batch size,
residual-scale candidates and ecological-memory fusion. Persist the neural
stage before memory export. The matched retained linear model has the same
starting backbone and training budget; it is already saved in the parent.
The nonlinear branch is the only changed model mechanism. It adds32 hidden
units at the readout, rather than enlarging spatial or temporal depth.

Preserve five parent arms and add `nonlinear_native_residual` and
`nonlinear_native_integrated`. Compare complete against complete and native
against native; also show strong trees and the preceding older model. Report
MAE, RMSE, bias, Q90 and per-partition effects with5,000 paired station draws,
averaging seeds within partition and weighting partitions equally. Do not
present an improvement over the older model as an improvement over the retained
release. Record useful source results before separately fixed confirmation.

## Expected evidence

If the new head reliably improves source-validation new-site error, the linear
readout was limiting the use of existing learned states and hydro-ecological
interactions. If it does not, retain the simpler head and investigate additional
predictive information rather than extrapolating from model size. These reused
development roles estimate candidate behavior; they are not independent tests.
