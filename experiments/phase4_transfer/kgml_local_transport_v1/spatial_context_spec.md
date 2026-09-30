# Spatial context residual pilot

## Question

The E3 spatial holdout currently favors RF-context, which uses explicit
current-month global, upstream and downstream observed-value summaries. The
existing KGML residuals use RF-local as their base, so their weak E3 result does
not test whether a graph branch can improve on that stronger spatial baseline.

This pilot fits the residual branch on out-of-fold RF-context errors and asks
whether a directed graph correction improves the already context-aware base.

## Design

- analyte: DOC
- masks: `e2a_strict`, `e3_spatial_seed42`
- seeds: 42, 43, 44
- arms: `rf_context`, `residual_context_nomsg`,
  `residual_context_msgdelta`, `residual_context_both`
- RF: 200 trees; station-blocked five-fold OOF predictions
- residual network: the existing observation-aware 12-month temporal trunk,
  hidden size 64, two spatial layers, dropout 0.1
- `context_mode=all` for the residual trunk, so the model sees the same
  visible global/upstream/downstream context family as the RF-context base
- `residual_context_nomsg` has an empty message edge set and is the matched
  null; `residual_context_msgdelta` uses upstream-only directed messages
- `residual_context_both` uses both edge directions as a spatial interpolation
  diagnostic; it is not interpreted as one-way transport
- no layer-count, lag or hyperparameter search is included

The context-base residual is trained on

```text
z_true - z_RF-context,OOF
```

where each OOF RF-context prediction is produced without labels from the held
stations. Validation and test prediction use the RF-context forest fitted on
all training cells, with only visible labels entering context features.

## Primary comparison

The primary spatial question is whether
`residual_context_msgdelta` improves on `residual_context_nomsg` and on the
standalone `rf_context` baseline in E3. A positive graph-vs-null difference
would show that the learned directed branch adds information beyond explicit
neighbor summaries. A null result would indicate that the current spatial
signal is already captured by the explicit context features.

All results are exploratory model-development evidence until a later
multi-seed confirmation is run. The pilot does not modify Phase 0--3 frozen
endpoints or previous KGML verdicts.
