# Stage-2D conditional-fusion feasibility specification (v1)

Status: **protocol design freeze before training, 2026-09-25**.

This is a new, versioned feasibility route after the completed Stage-2 support
integrity audit. It does not alter any Phase 0--3 file, Stage-2 endpoint, or
completed prediction product.

## Hypothesis

The local target support signal is conditional. A source-calibrated gate can
combine the H2X K-shot prediction with a simple ecology-time baseline and avoid
large basin-specific harm without giving up the broad K=5 improvement.

## Candidate ladder

Candidates are deliberately ordered from transparent to learned:

1. Fixed convex blend, weight selected on source-only validation.
2. Disagreement gate: reduce the H2X weight when H2X and the ecology-time
   baseline disagree beyond a source-calibrated quantile.
3. Support-quality gate: add support count, same-month spatial spread, temporal
   spread, ecological novelty, and source-support agreement.
4. A small logistic or isotonic gate fitted on source pseudo-target episodes.

No candidate may use target query labels, target query errors, or target basin
identifiers. The gate output is clipped to `[0, 1]` and K=0 must exactly return
the ecology-time baseline or the frozen source-only prediction, depending on
the candidate definition.

## Source-only pseudo-target calibration

For every held-out target HUC6/analyte task, source HUC6 basins are split into
meta-train and meta-validation pseudo-target episodes. Support and query cells
are drawn with the same-month rule, but all pseudo-query labels are used only
to fit/select the gate inside the source partition. The real target HUC6 query
labels remain unopened until final scoring.

The gate feature schema is frozen as:

```text
analyte, K, support_count, support_month_span, support_site_span,
ecological_novelty, h2x_baseline_disagreement, source_support_residual,
visibility_role
```

Target basin identity, target query labels, and any statistic computed from the
target query set are forbidden.

## Primary feasibility endpoints

- paired K=5 MAE versus the frozen ecology-time baseline;
- worst analyte × HUC6 relative harm;
- pooled relative improvement by analyte;
- gate-weight distribution and fraction of queries using support;
- support-shuffle gap as a diagnostic, not a new endpoint.

All native-unit summaries remain within analyte. HUC6 is the fixed stratum and
calendar month is the bootstrap cluster.

## Pass / stop rule

Pass to confirmation only if:

- worst cell is no worse than 5% relative to the selected baseline;
- at least two analytes retain at least 10% pooled improvement;
- at least three of five HUC6 tasks improve for each retained analyte;
- the gate is non-degenerate (not always 0 or always 1) and passes the
  pseudo-target label-perturbation contract;
- true support remains better than value-shuffled support in the same audit
  subset.

The previously observed DOC 102701 harm is therefore a named stress-test
endpoint. Passing requires fixing that case or explicitly showing why the
failure is an unavoidable, measurable transfer boundary.

## Provenance

Every candidate and product binds the task manifest, dataset, mask, feature
schema, gate candidate, runtime snapshot, and source pseudo-target split hash.
All outputs live under a new `stage2d_conditional_fusion_v1/` directory. No
existing endpoint or artifact is overwritten.

