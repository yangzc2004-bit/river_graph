# Local--Transport KGML v1

This experiment separates a local ecological/hydrological baseline from a
directed river-message residual.  RF-local uses station history, hydro,
season, static and ecological features; RF-context additionally uses current
upstream/downstream observed-value summaries.  The residual arms use the
causal observation-aware temporal graph trunk and differ only in edge
direction or edge availability.

## K1 pilot

- analyte: DOC
- masks: `e2a_strict`, `e3_spatial_seed42`
- seeds: 42, 43, 44
- arms: `rf_local`, `rf_context`, `h2x_t`, `residual_upstream`,
  `residual_both`, `residual_nomsg`
- epochs: 30; patience: 5
- RF: 200 trees; station-blocked 5-fold OOF residual targets

The residual is trained in the train-derived standardized log1p target space.
Validation predictions expose train/context labels; test predictions expose
train/val/context labels.  No target query label is used as a feature.

The pilot is a matched mechanism experiment.  It does not overwrite the
Phase 0--3 frozen products or the earlier graph-upgrade results.
