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
