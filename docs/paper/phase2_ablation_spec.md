# Phase-2 ablation spec (v1, frozen 2026-09-23)

Status: Phase-2A preflight companion to `docs/paper/primary_endpoints.json`.
**Nothing in this document changes a primary endpoint, gate, or claim**; it
only freezes the arm definitions, information set, training protocol and
staging of the Phase-2 same-protocol ablation. Machine-readable twin:
`configs/phase2_ablation_stcore_v1.json`. Contract tests:
`tests/test_phase2_controls.py`. Any change here needs a new version and a
recorded reason, like the endpoints file.

## 1. Staging (internal gates; not paper claims)

| stage | content | may it enter paper claims? |
|---|---|---|
| **2A preflight** | this spec + config + tests; synthetic forwards; short trainings (4 GNN arms, 1 seed, 1–2 masks, ≤12 short runs) | **no** |
| **2B directional pilot** | 6 arms × 3 seeds (42/43/44) × 8 key masks (≈144 configs) | **no** — compute/implementation screening only |
| **2C final matrix** | 6 arms × 5 seeds (42–46) × 8 key masks = **240 configs** | yes — the only matrix that can support Phase-2 claims |

A stage may start only after the previous stage's pass criteria are met. A
failed arm means "fix implementation and re-run", never "change the science".
Pilot outcomes must not trigger endpoint changes (`primary_endpoints.json`
anti-switching clause applies unchanged).

## 2. Cohort, masks, seeds

- Cohort: ST-only `data/processed/mississippi_graph_graphfix_st357.pt`
  (sha256 `33474f55…e77c0`), nodes `graph_nodes_graphfix_st357.csv`. 357
  stations × 654 months, 22,571 observed DOC cells.
- Masks: `experiments/masks_stcore_v1/` (with `masks_provenance.json`). The
  **8 key masks** are
  `e1_r20_seed{42,43,44}`, `e2a_strict`, `e2b_partial`,
  `e3_spatial_seed{42,43,44}` — the same 8 used by the historical multi-seed
  refresh. `e1_r40/r60` are appendix-only and are not part of the 240-config
  matrix.
- Training seeds: 2A `[42]`; 2B `[42, 43, 44]`; 2C `[42, 43, 44, 45, 46]`.
  `train_split_seed = 42` everywhere (masks are pre-frozen; the field exists
  for provenance parity with `BASE_MODEL_CONFIG`).
- Regime layout (13 columns of `dataset["regime"]`, standardized per column
  inside the model): `0:4` hydro (streamorde, log1p totdasqkm, slope,
  is_headwater); `4:8` landcover (forest, crop/hay, urban, wetland);
  `8:10` climate (precip, log1p tmean+20); `10:11` soil (SOM);
  `11:13` topo (elevws, bfiws).

## 3. Arm definitions (exact)

Base channels (every GNN arm, from `gcn.py::_build_inputs`): standardized
temp, temp-availability flag, standardized discharge, discharge-availability
flag, month sin/cos, standardized lat/lon, DOC-observation channel + visibility
flag (filled at train/inference time). The historical comment naming the mask
channels "temp_m/flow_m" is a misnomer: they are availability **masks**.

| arm | paper name | architecture | env_groups | env_encoder | edge_set | node input |
|---|---|---|---|---|---|---|
| `H2` | transport, no StreamCat ecology | `transport` | `["hydro"]` | false | `river` | 10 base + 4 hydro = 14 |
| `H2E` | transport + raw ecological context | `transport` | `None` | false | `river` | 10 base + 13 regime = 23 |
| `H2X` | transport + ecological encoder (ours) | `transport_enc` | `None` | true (`env_emb=32`) | `river` | 10 base + 4 hydro = 14; encoder input = 9 ecological cols (`4:13`) |
| `H2X_nomsg` | **no-message control** | `transport_enc` | `None` | true (`env_emb=32`) | `empty` | identical to H2X |
| `eco_RF` | ecological random forest | tabular (`sklearn.ensemble.RandomForestRegressor`, `n_estimators=200`) | — | — | none | §4 feature set |
| `eco_MLP` | ecological MLP | tabular (`sklearn.neural_network.MLPRegressor`, hidden `(128, 64)`) | — | — | none | §4 feature set |

H2E definition, in words: `architecture="transport"` plus the **full** ST-core
regime block as raw standardized channels concatenated into the node input,
with **no** ecological encoder. H2X: same regime block and same training
budget, but cols `4:13` go through the ecological encoder (`env_encoder=True`)
instead of being concatenated raw.

**Naming rule for the paper:** `edge_set="empty"` disables edge messages but
**retains the self path** (`GatedDirectedConv` computes `self_lin(x) + up +
down` and `_agg` returns zeros for an empty edge tensor). It is therefore
called the **no-message control**, never "no-graph model" or "no-topology
model": node features, regime, and the self transform all remain. It is not a
model with node identity or node information removed.

Shared GNN training config (identical for all four GNN arms):
`hidden=64, layers=2, dropout=0.1, lr=1e-3, weight_decay=0, edge_dropout=0,
share_weights=false, max_epochs=200, patience=20, edge_direction="both",
loss="log1p_mse"` (MSE in log1p mg/L, the existing `GCNDocModel` objective),
prediction clamped to the train log-space `[min, q99.5]` then `expm1` (the
existing `predict()` convention). `env_emb=32` applies only to encoder arms.

## 4. Information set (parity rule)

Every arm receives exactly the same per-(station, month) information; the only
allowed difference between arms is access to the river graph.

| block | content | H2 | H2E | H2X / nomsg | eco_RF / eco_MLP |
|---|---|---|---|---|---|
| dynamic hydro | temp + availability flag, discharge + availability flag (month t) | ✓ | ✓ | ✓ | ✓ |
| season | month sin/cos | ✓ | ✓ | ✓ | ✓ |
| geography | lat, lon | ✓ | ✓ | ✓ | ✓ |
| regime | 13 static cols | cols `0:4` only | all 13 (raw, no encoder) | cols `0:4` in node input, `4:13` in encoder | all 13 |
| visible DOC field | observed DOC of cells in `obs_mask`, month t | via graph + own channel | same | same | via graph-free aggregates (below) |
| river graph | `edge_index` + `edge_attr` | ✓ | ✓ | ✓ | **never** |

Frozen clarifications:

1. **Visibility rule** (identical everywhere): `obs_mask = train ∪ val ∪
   context` — the inference-time convention shared by `GCNDocModel.predict`
   and `baselines::_ctx`. Test-cell DOC is never visible to any arm.
2. **No self-label leakage:** a cell's own DOC is never a feature of its own
   row. GNN arms achieve this because evaluated (test) cells sit outside
   `obs_mask` and training targets are re-masked; tabular arms exclude self
   from every DOC-derived feature on fit rows.
3. **Visible-DOC channel for tabular arms:** per-cell tabular rows cannot hold
   the cross-station DOC field the GNN reads through messages. `eco_RF`/
   `eco_MLP` therefore receive two permutation-invariant aggregates of the
   month-t visible-DOC field, computed without the graph and without self:
   mean and count of visible DOC across stations at month t (0 when empty).
   They never receive neighbour identity, distances, or edges — cross-station
   structure is exactly what the GNN arms add, and the **no-message control**
   is the strict information-parity ablation for it (identical inputs to H2X,
   messages zeroed).
4. **Tabular training/eval convention:** fit rows = train cells only (the same
   cells GNN loss is computed on); `eco_MLP` early-stops on the **frozen val
   cells** with the same patience rule as the GNN (`min_delta=1e-6`), never on
   a random carve of train; `eco_RF` has a fixed seed (`42`) and no val-based
   selection. Both predict test cells only (NaN elsewhere), matching the
   historical B2/B3 export convention.

## 5. Provenance and output rules

- All Phase-2 outputs go to `experiments/phase2_ablation_stcore_v1/`
  (subdirs `results/`, `predictions/`), written with the standard
  parquet + `.meta.json` sidecar (`predictions.py` + `provenance.py`).
  Historical directories (`experiments/results/`, `experiments/predictions/`,
  `experiments/frozen_results/`) are **never written to**.
- Every run records `config_hash` over the full arm config including
  `dataset_sha256`/`mask_sha256`; distinct arms must produce distinct config
  hashes; overwriting a different config under the same name raises
  `PredictionConflictError` (names: `P2X_<arm>[_s<seed>]__<mask>`).
- Split, early stopping, seed and clamp conventions are exactly as §3–§4; a
  run that deviates is not a Phase-2 run.

## 6. Phase-2A checklist and pass criteria

Deliverables (this stage): `docs/paper/phase2_ablation_spec.md`,
`configs/phase2_ablation_stcore_v1.json`, `tests/test_phase2_controls.py`,
plus the minimal plumbing they pin (`ecological_tabular_features`,
`EcoRandomForest`, `EcoMLP` in `src/river_graph/baselines/baselines.py`).
**No model-architecture change is part of 2A.**

Budget: 4 GNN arms × 1 seed × 1–2 representative masks (`e2b_partial`,
`e3_spatial_seed42`) short trainings (few epochs), plus tabular fits — at most
≈8–12 short runs.

Pass criteria (all required):

1. every arm completes a synthetic forward;
2. every arm completes a short training on a real mask;
3. outputs contain no NaN on evaluated cells;
4. prediction coverage and test-cell counts are identical across arms on the
   same mask (and identical to the baseline convention);
5. provenance config hashes are pairwise distinct and no hash conflict occurs;
6. H2E / H2X / no-message input dimensions and actually-used feature sets
   match §3 (14 / 23 / 14+9 as specified);
7. no-message behaviour tests pass: edge tensor empty at the model boundary,
   output independent of edge-attribute values, self path still effective,
   inputs/mask/objective identical to H2X;
8. **zero scientific conclusions are produced or recorded.**

## 7. Phase-2B pilot (after 2A passes)

6 arms × 3 seeds × 8 key masks (96 GNN configs + 48 tabular fits). It may only
decide: does H2E run; does H2X still gain on H2E; does no-message actually
change results; do eco RF/MLP become strong baselines; which arms deserve the
5-seed matrix. Failure handling: fix implementation, re-run; never swap
endpoints. Pilot numbers never appear as paper claims.

## 8. Phase-2C final matrix and decision rules

240 configs (6 × 5 × 8). Gates (unchanged from `primary_endpoints.json`):
H2X vs best no-graph model ≥10% MAE reduction in ≥2 of {E2a, E2b, E3}; stable
gain vs H2E in ≥1 extrapolation scenario; direction consistent in ≥3/5 seeds;
E1 within 5% of the best baseline; station-clustered paired bootstrap on every
main comparison; no claim carried by one HUC2 or one station type; high-DOC
top 5%/10% precision, recall and tail error reported separately.

Interpretation table (fixed before results):

| outcome | reading |
|---|---|
| H2X > H2E | the encoder adds value beyond raw ecological context |
| H2E > H2X | ecological context is valuable; the encoder is not the contribution |
| eco_RF/MLP > H2X | pivot: ecology-aware missing-data / monitoring-design paper, models become baselines |
| no-message ≈ H2X | river topology has not been shown necessary; do not sell topology as the core contribution |
| no-message clearly worse than H2X | edge messages carry predictive value beyond the self path |
