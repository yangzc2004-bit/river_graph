# Reproduction and figure checks

- **Full project tests:** 1,247 passed, 2 skipped, 8 existing warnings. Thirteen
  new checks cover source-phase compensation, conserved moments, pure transit,
  volume/flow cancellation, shared-path absence and survey-proxy grain/missingness.
- **Ruff:** all checks passed.
- **Historical artifact audit:** exit 0, retaining 83 parquet-only verifications,
  one reported zero-coverage artifact, the excluded known G0 conflict and the
  85 historical no-sidecar statuses. No old artifacts were changed.
- **Full new replay:** passed for all 33,984 scenario rows, four-example numerical
  refinement, the five field proxy rows, original 22 elongated/broad pairs and
  geometry/cohort/form tables. Source snapshots and both figure receipts match.
- **Visual inspection:** both English and both Chinese PNG figures were actually
  viewed. Maps use the stored real flowlines, complete boundaries and selected
  corridors. Curve axes, legends, labels and field-junction point labels are
  readable; export PDFs come from the same plots.

The 295 junction corridors belong to original whole-network form classes; two
sparse instances without an eligible junction retain exclusion records. The
59 monitored connections are summarized separately after receiver aggregation.
Repeated scenario settings are parameter controls, not independent field samples.

HUC4 bootstrap intervals describe this geometry ensemble. Field DOC is neither
used to choose pulse phases/mixing strengths nor fitted by the conservative
kernel. The rectangular width/depth proxy is not substituted for tracer timing.
Constant-flow concentration-time conservation does not imply equal input carbon
mass across the alternative discharge settings.

No training, endpoint replacement or model selection was performed. Older
geometry classes, river-form experiments and model results remain unchanged.
