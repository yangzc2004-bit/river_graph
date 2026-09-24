# Target-unseen K=0 decision record (2026-09-24)

## Blocker

The proposed leave-one-analyte task hides every target-analyte label outside
the K support cells. At `K=0`, a predictor trained only on DOC, pH, and
conductance source tasks has no legal target-analyte value or scale from which
to produce a native-unit prediction. The three units are not interchangeable,
so a pooled mean or standard deviation would be an unregistered target prior.

The privileged same-analyte climatology in `stage2a_same_analyte_v2/` is not a
solution: it uses target-analyte labels outside the target HUC6 and is clearly
labelled as a diagnostic reference.

## Options considered

| option | consequence | decision |
|---|---|---|
| Use a target-analyte global mean/scale at K=0 | leaks a target prior and makes the output mapping target-informed | reject |
| Standardize all analytes to a shared latent score and report native-unit MAE | target-to-native inverse mapping is still unavailable at K=0 | reject |
| Evaluate only K≥1 for a genuinely unseen analyte | removes the promised K=0 base and changes the frozen endpoint | possible only in a new spec version |
| Let the target analyte be observed in source basins and hold out only the target basin | K=0 has a legal target-specific head; the claim becomes cross-basin transfer with shared multi-analyte representation | viable fallback, requires new charter/spec |
| Supply a preregistered analyte metadata-to-scale mapping | needs a defensible metadata source and an independent validation of that mapping | not available yet |

## Current decision

Do not start a shared transfer model or silently redefine K=0. Stage 3 remains
locked. The internal same-analyte diagnostic may be reported as a feasibility
reference, but it cannot be promoted to leave-one-analyte evidence.

Before transfer training, choose one of the two viable versioned routes:

1. **Cross-basin multi-analyte route:** keep target-analyte labels in source
   basins, hold out the target HUC6, and compare a shared representation with
   single-analyte heads. K=0 is native-unit identifiable.
2. **Strict unseen-analyte route:** issue a new endpoint specification that
   evaluates only K≥1 and defines the K=1 output mapping before any query
   labels are read. K=0 must be removed rather than filled with a target prior.

Any choice must be recorded as a new charter/spec version with the fact that
the Stage-2A exploratory outputs were already observed. Until then, no
cross-analyte transfer claim is enabled.
