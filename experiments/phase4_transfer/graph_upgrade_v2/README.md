# Observation-driven graph upgrade

This directory contains the second-generation H2X experiments. The released
H2X-T runs under `temporal_h2x_v1/` remain the matched baseline. The upgrade
asks whether a graph model can use three pieces of information that a static
snapshot model leaves implicit:

1. when a nearby target observation was last made;
2. when an upstream signal can reach a downstream station; and
3. whether short, seasonal and long-term dynamics need separate memories.

The mechanism sequence is:

* **M1 — observation-aware memory**: last value, observation age, recent
  support counts, upstream/downstream visibility and a GRU-D-style recurrent
  decay. Training alternates point, contiguous-month and whole-station masks.
* **M2 — lagged transport**: directed upstream messages in the `{0, 1, 3,
  6, 12}` month buckets. Lag weights use edge attributes and the observed
  hydro/support state at the source and destination.
* **M3 — multi-scale time**: causal short and seasonal convolutions combined
  with a trend GRU.

The initial pilot uses DOC, pH and specific conductance, the temporal and
spatial holdout families (`e2a_strict`, `e3_spatial_seed42`) and three seeds.
Each mechanism is compared with the existing H2X-T and temporal random forest
using the same target masks. M1 is the first scientific run; M2/M3 smoke runs
only establish that the independent components are executable until M1 is
reviewed.

All products are full station-month grids with hidden cells retained as
predictions and `y_true` present only for final evaluation. Existing Phase 0–3
artifacts and the model-upgrade-v1 results are not overwritten.

