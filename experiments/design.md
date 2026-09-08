# Milestone 3: DOC Reconstruction Benchmark — Experiment Design

Status: fixed before implementation. Changes here require a stated reason.

## Task definition

Spatiotemporal masked reconstruction on the station graph.

- Data: `mississippi_graph_v02.pt` — N=571 stations, E=562 directed edges
  (upstream → downstream), T=652 months (1972-04 .. 2026-07).
- Unit of prediction: one (station, month) cell.
- Label: monthly-mean DOC (mg/L). Observed on 8.9% of cells.
- Model input at month t: node features `[temperature, discharge]`,
  the observed-DOC channel (0 where unobserved) + its mask, and
  month-of-year (sin/cos). The DOC channel lets the GNN do what the task
  sketch describes: given A=5, C=8 at month t, predict B=?.
- DOC is modeled in log space (`log1p`); metrics are reported in mg/L
  after `expm1`.

## Research questions

- RQ1: Does real river topology improve DOC reconstruction?
  (topology ablation: no graph vs random graph vs river graph)
- RQ2: How much missing data can be recovered? (mask rates 20/40/60%)
- RQ3: Does the learned structure generalize in time and space?
  (temporal and spatial extrapolation)

## Experiments

### E1 — Random mask (RQ1, RQ2)

Split observed cells into train / val / test. Test fraction =
mask rate r ∈ {0.2, 0.4, 0.6}; of the remaining cells, 10% are validation
(early stopping), the rest train. Seeds 42, 43, 44 → mask files
`experiments/masks/e1_r{r}_seed{s}.npz` (arrays of cell indices).

### E2 — Temporal extrapolation (RQ3)

Train: cells in 1972-04 .. 2020-12. Test: cells in 2021-01 .. 2026-07.
Validation: last 10% of train months. Answers "can the model predict
future months", and guards against the trivial seasonal shortcut.

### E3 — Spatial extrapolation (RQ3)

Hold out 20% of stations (all their cells) as test, sampled from the
largest connected component (407 nodes) with seed 42; training stations
stay in the graph as context. Answers "can information propagate along
the river network to unmonitored stations".

## Models and baselines

| id | model | inputs | graph |
|----|-------|--------|-------|
| B0 | station mean | train cells of the station (global mean fallback) | — |
| B1 | kriging (PyKrige, lat/lon) | stations observed in that month | — |
| B2 | random forest | temp, discharge, month-of-year, lat, lon | — |
| B3 | MLP | same as B2 | — |
| G0 | GCN | full monthly feature set incl. DOC channel | real |
| G1 | GCN on random graph (same N, E, seed 42) | same | random |
| G2 | GCN variant = MLP on same features (no message passing) | same | none |

G0 vs G2 isolates message passing; G0 vs G1 isolates *real* topology.
A degree-preserving random graph is a noted refinement of G1 if reviewers
push on it.

Notes:
- Kriging is Euclidean; river-distance spatial models (SSN) exist and are
  cited as related work, not implemented in the MVP.
- GNN stays simple (GCN; GAT as a robustness check only). The contribution
  is topology as ecological prior, not a new architecture.

## Leakage rules

- Static station-level DOC statistics (if ever used as features) are
  computed on train cells only.
- In E1, the DOC input channel at month t contains train/val cells only;
  test cells are zeroed and masked out.
- Feature channels (temperature, discharge) are treated as observed
  inputs everywhere; they never contain DOC-derived quantities.
- All splits are generated once and stored under `experiments/masks/`;
  every reported number names its mask file and seed.

## Metrics

RMSE (primary) and MAE on test cells, in mg/L after `expm1`; R² as
secondary. Reported per experiment × model × mask rate (E1), mean ± std
over seeds 42/43/44.

## Expected output table (E1 example)

| model | r=0.2 | r=0.4 | r=0.6 |
|-------|-------|-------|-------|
| B0 station mean | | | |
| B1 kriging | | | |
| B2 RF | | | |
| B3 MLP | | | |
| G2 no-topology | | | |
| G1 random graph | | | |
| G0 river graph | | | |
