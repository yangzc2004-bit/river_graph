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

## H2: Physics-informed asymmetric transport (revised after E3 diagnosis)

The E3 seed42 diagnosis localized the H1 failure: not a model-size problem
(H1.5 regularization did nothing), not out-of-distribution labels, but
**headwater stations** (upstream_degree=0, high-DOC mountain streams in
HUC2 10/11) — nodes where the upstream transport signal does not exist.
Directed message passing assumes every node has an informative upstream;
river networks have heterogeneous regimes.

H2 therefore has two coupled parts:

1. **Edge transport gate**: message from j to i is
   `m_ji = gate(e_ji) * W h_j` with `gate(e) = MLP(edge features)`, so
   propagation strength is set by the physical reach, not uniform.
   Edge features (v1, ≤5, no more): river hop distance, flowline length,
   stream order, drainage area, slope.
2. **Node hydrologic regime encoding**: node features additionally include
   is_headwater / stream order / drainage area of the node's own reach, so
   the model can represent "this is a headwater mountain stream" instead
   of memorizing one.

Evaluation focus: E3 three seeds (esp. seed42) and the headwater subgroup
(upstream_degree=0), plus the E1/E2b retention check.

Naming: working name HydroGRN → H2 stage. (GRN collides with gene
regulatory networks; final name deferred.)

## H1 results and the H1.5 stabilization step (added after the H1 run)

H1 (directed) beat G0 on all 9 E1 masks and became the best model on both
E2 variants (E2a MAE 1.094 vs RF 1.115; E2b 1.067 vs RF 1.114), but
collapsed on E3 spatial transfer (R2 -2.31 vs G0 river 0.32).

Resolution (after H1.5 + diagnosis): the collapse was NOT structural.
Regularization (H1.5) and physics features (H2) did not help, and the
training curve was stable for 250 epochs. Root cause: a handful of
held-out high-DOC headwater stations got unbounded extrapolations, and
the inference clamp ceiling was the train-set MAX (445 mg/L, a 1970s-era
high-DOC regime value, NOT a flood pulse — see
experiments/analysis/extreme/P0_label_audit_06438000.md), so a few cells produced squared errors large enough to destroy
the mg/L-space R2. Fix: clamp to the train 99.5th percentile. After the
fix, H1 on E3 seed42 is R2 0.27 (was -2.28).

E3 three-seed summary (R2): kriging 0.27, G0 0.25, H1 0.25, H2 0.23 —
no directed advantage on spatial transfer, but no pathology either. The
direction advantage lives in E1/E2 (known stations, temporal context).
Headwater regime awareness (H2's node regime encoding) remains motivated
by the diagnosis: errors still concentrate in HUC2 10/11 headwaters.

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

## M5: Ecological context for spatial transfer (added after H2)

H2 proved the *propagation* mechanism; E3 remains flat (all models
R2 0.2-0.27) because an unseen station's identity is unknown — topology
says who it connects to, not *what* it is. Two headwater stations with
identical graph roles (alpine forest vs agricultural hillslope) have
different DOC regimes.

M5 adds a static **ecological context** block per node (kept separate from
dynamic channels): drainage area / elevation / slope / stream order
(hydrology), precipitation + temperature climatology (climate),
forest/agriculture/urban/wetland % (land cover), soil carbon (soil).

Data source: EPA StreamCat (catchment attributes keyed by NHDPlusV2
COMID — direct join, no spatial overlay) + NHDPlus VAA (already fetched).

Design rule: environment defines the node state, the river graph defines
propagation. Validation experiment is E3-focused: G0 vs H2 vs H2+Env,
success = E3 R2 improvement while keeping E1/E2b.

## M5 terminology and ablation rules (added after review)

- Call it **ecological context** (catchment ecological identity), never
  "environment parameters" — it encodes what a reach *is*, not model knobs.
- Leakage note: StreamCat attributes are static catchment properties
  (NLCD 2019 land cover, 1991-2020 climate normals, STATSGO soils). They
  contain no information from the evaluation period, so E2/E3 remain clean.
- Environmental ablation (answer "what actually helps"): H2+Env groups —
  hydro / landcover / climate / soil+topo / all — on E3 seeds first.
- If the hypothesis holds, the next upgrade is a real **ecological context
  encoder** (MLP embedding of the context block) rather than raw feature
  concatenation.
