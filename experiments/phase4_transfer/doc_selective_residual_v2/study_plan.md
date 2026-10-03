# Converged-budget comparison of DOC residual losses

The [four-loss experiment](../doc_selective_residual_v1/study_plan.md) changes
only source training loss/sampling weights in the existing encoder/GRU model.
Source-validation traces show that some new objectives still improve at the
60-epoch ceiling, unlike the unchanged tail2 control. Before reading their
target-comparison results, allow a common maximum of **120 epochs**, retaining
patience5 and every other setting.

All four objectives and nine partition/seed packages start from the same
original expert weights. The frozen source forest, support basis, ecological
memory, raw input views, validation criterion, thresholds and target queries
are unchanged. Training ends sooner when patience is exhausted. The earlier
60-epoch experiment is preserved. No new architecture, loss weight, learning
rate or model choice is introduced by the extension.

Use the original fixed loss contrasts and report all K curves. Compare each
longer-budget arm with its own earlier counterpart; distinguish optimization
duration from the loss-definition effect. Check the old trace prefixes and
the unchanged tail2 control. If a fit still reaches120, report its optimization
status rather than automatically launching another budget expansion.

The per-run implementation identifier remains `doc_selective_residual_v1`;
the new root, `epochs:120`, wrapper and runtime snapshot identify this duration
completion. These remain same-cohort model-development experiments.
