# Ecology-conditioned transfer charter v2

Status: **new research direction, protocol design stage, 2026-09-25**.

## Working title

**When does sparse local evidence help? Ecology-conditioned transfer for water-quality reconstruction**

## Central idea

Sparse target observations are not uniformly useful. Ecological and temporal
context can provide a transferable prior, while local target support can either
correct the prior or amplify its error. The paper will build a reliability gate
that learns when to trust local support and when to fall back toward the
ecology-time baseline.

The scientific contribution is the **conditionality of transfer** and an
operational response to it. The GNN remains one representation inside the
comparison, not the headline contribution.

## Testable claims

1. Support gains vary systematically across analytes and held-out HUC6 basins.
2. Source-only signals such as ecology novelty, support density, support spread,
   and model disagreement predict when support-based correction is reliable.
3. A source-calibrated reliability gate can preserve pooled K=5 gains while
   reducing the worst-basin harm seen in the fixed H2X route.
4. The remaining errors define measurable boundaries of ecological transfer;
   they are not averaged away into a universal claim.

## Main paper result

The primary comparison is:

```text
fixed H2X K=5  ->  source-calibrated reliability-gated blend
```

against the frozen ecology-time baseline. The gate is evaluated once on the
same held-out HUC6 tasks and analytes. The existing DOC 102701 failure is a
predeclared stress test: a successful gate must reduce that harm without
discarding the gains in the other basins.

## Scope boundaries

- No causal, mechanistic, or universal claim.
- No new external basin until the internal conditional-transfer gate passes.
- No blind-spot or active-sampling claim from uncertainty ranking alone.
- No hidden-test tuning: gate candidates and endpoints are frozen before the
  final target query labels are opened.

## Decision rule

The gate is worth a full confirmation experiment only if a feasibility pilot
meets all three conditions:

1. worst analyte × HUC6 relative harm is above -5%;
2. pooled K=5 improvement remains at least 10% for at least two analytes;
3. the gate weight is selected from source-only pseudo-target episodes and
   remains finite, non-degenerate, and label-free on target tasks.

If it fails, the paper still reports the conditional transfer boundary and
returns to the DOC reconstruction mainline. The gate failure is itself a
useful boundary result, not a reason to add an unconstrained model family.

