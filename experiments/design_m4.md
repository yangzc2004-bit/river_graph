# Milestone 4: River-aware GNN — Model Design

Status: proposal. Milestone 3 established that topology helps (river >
none > random on E1; only river transfers spatially on E3) but a plain
GCN does not yet exploit the *physics* of rivers. This document fixes the
design of the actual contribution model.

## Motivation (what the M3 results say)

- Plain GCN treats neighbors as an undirected average. Rivers are
  directed transport systems: upstream observations are *causal inputs*
  to a station, downstream observations are *inversely informative*
  (a downstream measurement constrains what could have come from above).
- Edges are not interchangeable: an edge means "B is the next monitored
  station downstream of A along the mainstem", and it carries physical
  attributes (distance along the network, intervening drainage area,
  mean flow).
- DOC at month t is not independent of t-1: transport and memory exist,
  and our labels are monthly means on a shared calendar.

## Model: HydroGRN (working name)

Per-month directed message passing + temporal recurrence:

1. **Directed relational encoder** (per month t)
   - Two relations: upstream (j→i where j is upstream of i) and
     downstream (j→i where j is downstream of i).
   - `h_i^t = W_self x_i^t + σ( W_up Σ_{j∈up(i)} f(e_{ji}) h_j^t + W_down Σ_{j∈down(i)} f(e_{ji}) h_j^t )`
   - Upstream/downstream use separate weights — this is the "not just
     neighbor averaging" part. 2 layers.
2. **Edge gating** `f(e)` = MLP over edge features → per-edge scalar/vector
   weight ("river transport" instead of uniform mean).
   - v1 edge features: river hop distance (from NLDI flowline walks,
     already cached), log geographic distance, and (follow-up) NHDPlus
     attributes per reach: slope / mean annual flow / total drainage area.
3. **Temporal module**: per-node GRU over months on top of the spatial
   embeddings; prediction head reads the GRU state at month t.
   - Handles irregular observation via the existing obs channel + mask.
4. **Training**: identical protocol to M3 (same masks, same leakage rules,
   per-epoch re-masking of train cells, log-space MSE, early stopping on
   hidden-from-input target loss, clamp to train range at inference).

## Ablation ladder (each rung isolates one design choice)

| id | model | what it tests |
|----|-------|---------------|
| G0 | GCN river (M3, done) | undirected plain baseline |
| H1 | directed R-GCN | + direction |
| H2 | H1 + edge features | + transport attributes |
| H3 | H2 + GRU | + temporal memory |

Success criterion: H1 > G0 on E1/E3; H3 > H2 on E2-b (temporal context
should matter most where the benchmark already has partial observation).

## Data work needed

- [x] edge hop distance from cached NLDI flowline walks (no new downloads)
- [ ] NHDPlus reach attributes (TotDASqKM, slope, mean annual flow) per
      COMID — small fetch, resumable
- [ ] (later, post-MVP) HydroATLAS land cover / soil carbon / precipitation

## H1 results and the H1.5 stabilization step (added after the H1 run)

H1 (directed) beat G0 on all 9 E1 masks and became the best model on both
E2 variants (E2a MAE 1.094 vs RF 1.115; E2b 1.067 vs RF 1.114), but
collapsed on E3 spatial transfer (R2 -2.31 vs G0 river 0.32).

Interpretation (a finding, not just a failure): directional message
passing substantially improves temporal reconstruction but reduces spatial
transferability — explicit flow direction captures local transport
dynamics while increasing sensitivity to regional topology. The directed
encoder has ~3x the parameters of G0 on only 571 nodes, so it memorizes
regional neighborhood patterns that do not exist at unseen stations.

H1.5 keeps direction but cuts overfitting:

1. **Shared relation weights + multiplicative direction gates** — one
   conv for both relations, per-relation gates keep up/down
   distinguishable. (An additive direction embedding does NOT work:
   summing both relations makes the layer invariant to edge direction.
   Multiplicative interaction is what preserves direction.)
2. **Edge dropout (p=0.2)** — training with randomly deleted river edges
   forces the model not to memorize specific reaches.
3. **Weight decay 1e-4**.
4. **E3 promoted to 3 seeds (42/43/44)** to separate signal from luck.

H2/H3 are postponed until H1.5 either fixes E3 or tells us direction
inherently trades off against spatial transfer.
