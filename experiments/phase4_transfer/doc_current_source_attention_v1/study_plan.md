# Hydrology-conditioned use of current source DOC

## Scientific question

The completed geographical study finds a0.412% overall and0.464% Q90 error
improvement from fixed ecological weighting of current source innovations.
The overall interval crosses zero; the tail interval is positive. Equally
informed trees also benefit, and neural-only overall improvement is absent.
The existing portable model remains the release.

Test whether the receiving site's present GRU state and donor hydrological
conditions can select more useful current source departures. Earlier learned
retrieval used fitted donor residual profiles. This study uses actual same-month
source OOF residual innovations, while receiving DOC/pH/conductance remain hidden.

## One model extension

Keep the existing ecological self encoder, observation-aware GRU, environmental
reference,12-month window, native residual objective, seven old interactions
plus innovation interaction, daily hydro and static memory fusion.
The current41-feature readout remains. Add a two-head donor attention residual
to its scalar head. Each head has32 projection dimensions.

- Query: the SAME current GRU hidden state, nine ecological features and eight
  existing bounded daily-flow descriptors.
- Keys: each candidate's ecology and same-month daily-flow descriptors.
- Values: same-month native source innovations, fixed1mg/L scale.
- Prior: the existing exp(-ecological distance/sigma) weighting, plus the same
  unit-weight zero-innovation prior.
- Candidate pool: the same at-most20 source stations and matched current/
  earlier-season availability used in the completed information study.
- Fusion: a zero-initialized projection of the two weighted innovations into
  the existing scalar residual; GRU and ecology are trained jointly as before.

This reallocates ecological source information. It is not a physical upstream
attention operator or a causal-effect estimator.

## Matched source development

Use source partitions142/143/144 ×seeds42/43/44,30epochs/patience5, existing
learning rates and tail weight2. Reuse the90 verified double-held-fold forests.
Their query/donor exclusion and source library roles remain unchanged.

Fit three arms (27 neural fits):

1. Learned attention, real current innovations.
2. Learned attention, earlier-season innovations; current keys/availability match.
3. Fixed ecological attention prior, real innovations; identical allocated shape.

In the fixed-prior arm, query/key scores have zero influence; their connected
gradients are zero. The original scalar readout and attention-value projection
train under the same budget. This separates adaptive donor choice from extra
readout capacity. Keep the retained complete, preceding source-informed complete
and equally informed trees as saved references. No new forest or neighbour scan.

The attention module begins with zero output contribution. Save/reload the full
head, masked donor inputs and diagnostics. A station excluded from an episode
cannot supply library values or reference predictions. Current input gathering
uses that month only; the existing GRU supplies causal history.

## Evidence and decision

Evaluate source-validation overall MAE, Q90 MAE, bias, station-equal error and
partition/seed directions with5,000 paired station draws. Report comparisons
with real fixed prior, historical attention, preceding source-informed complete
and retained complete; neural-only contrasts remain separate. Inspect prior
mass, entropy and donor use by hydro availability/support. Preserve null results.

First complete input/zero-fusion/normalization/future/exclusion/save-load tests
and one technical package, then all nine packages. Proceed to a separate fixed
geographical version only if source evidence improves the actual complete
model and adaptive weighting adds value over matched controls. The completed
geographical and external products do not select dimensions or checkpoints.
