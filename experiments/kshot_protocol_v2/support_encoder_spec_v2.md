# Support Encoder v2 — Frozen Design Specification

Date: 2026-09-22. Status: **specification only** — no training code exists yet.
Parent protocol: `experiments/kshot_protocol_v2/protocol.json` (tasks, K ladder,
success gate are already frozen and must not be re-derived here).
Machine-readable config for later runs: write `configs/support_encoder_v2.json`
in Step 4 and keep it consistent with this file.

This supersedes the learned `SupportEncoder` gate in
`src/river_graph/models/support_encoder.py` (Task 3 case B, v1), which failed:
it predicted a blend gate `g` between `base` and `local_mean` and got **worse**
than K=0 as K grew (see `experiments/kshot_support_st357/support_metrics.csv`).
The closed-form `analytic_blend` / `residual_idw` stay in the repo as
**baselines only**, not as the model claim (already required by
`protocol.json → judge_requires_support_encoder`).

---

## 1. Problem and notation

| symbol | meaning |
|---|---|
| `r` | target basin (HUC6), all DOC labels hidden during training |
| `s` | source basin (HUC6, `s ≠ r`) supplying training episodes |
| `j` | month index (v2 episodes are same-month) |
| `support` | K observed cells of basin `r` (or `s` during training) at month `j` |
| `query` | fixed set of remaining observed cells at month `j` (v2 definition) |
| `base_q` | H2X prediction at query cell `q` with **no** local support |
| `y` | label in mg/L; **every label enters the model as `log1p(y)`** |

Goal: given `base_q` and up to K support observations, predict
`log1p(ŷ_q) = log1p(base_q) + Δ_q`, where `Δ_q` is a residual correction that
uses the support set. K=0 must reproduce `base` exactly.

---

## 2. Inputs

### 2.1 Support token (one per support cell `s_i`)

| field | definition | notes |
|---|---|---|
| `log1p_y` | `log1p(y_{s_i})` | |
| `resid` | `log1p(y_{s_i}) - log1p(base_{s_i})` | `base_{s_i}` from the **cross-fitted** base (§4) |
| `hop` | undirected shortest-path hops `s_i → q` on the station graph | `min(hop, hop_cap)` with `hop_cap=10`; use `hop_cap+1` for disconnected |
| `dir` | 3-way: `query_downstream_of_support` / `query_upstream_of_support` / `other_or_multi_hop` | same predicate as `scripts/diagnose_kshot_v2_reachability.py` |
| `streamorde` | node `streamorde` | from `data/processed/reach_attributes.csv` (authoritative) |
| `totdasqkm` | node `totdasqkm`, standardized | same file |
| `time_gap` | `|j_s - j_q|` in months | **always 0 under v2** (same-month); channel reserved for Case C |
| `regime_s` | 9-dim ecological block (regime cols 4..12) | standardised with train-basin stats |
| `regime_diff` | `|regime_s - regime_q|` | elementwise, 9-dim |

HUC codes come from `graph_nodes*.huc_cd`, never from station-id prefixes.

### 2.2 Query token (one per query cell)

| field | definition | notes |
|---|---|---|
| `log1p_base` | `log1p(base_q)` | |
| `regime_q` | 9-dim ecological block | same standardisation as above |
| `month_sin/cos` | month-of-year sin/cos | same convention as `GCNDocModel._build_inputs` |
| `K` | `|support|` | scalar, standardised over training episodes |
| `target_meta` | reserved analyte embedding slot | zeros now; leaves room for leave-one-analyte-out (Task 6) without changing the input shape |

### 2.3 Normalisation

Standardisation statistics are computed **once per fold** on the training
episodes of that fold and stored with the checkpoint. No test-basin statistics.

---

## 3. Model

**Set encoder with pairwise attention and geometric biases** (a DeepSets mean
pool is the pre-registered ablation, not the main model).

```
h_i  = MLP_sup(token_i)          # d = 64
hq   = MLP_qry(query_token)      # d = 64
bias = Linear([hop_emb, dir_emb, |streamorde_s - streamorde_q|,
               log1p(1+totdasqkm_s) - log1p(1+totdasqkm_q)])   # scalar per pair
α_i  = softmax_i( (hq · h_i)/√d + bias_i )
c    = Σ_i α_i h_i
Δ_q  = MLP_out([hq, c])          # scalar; linear head initialised at 0
```

- `hop_emb = Embedding(hop_cap+2, 8)`; `dir_emb = Embedding(3, 4)`.
- `MLP_*`: 2 hidden layers width 64, ReLU, dropout 0.1 (match H3A convention).
- **K=0 is exact:** if `|support| = 0`, return `Δ_q = 0` by construction (skip
  the module; never rely on softmax over an empty set).
- Output is a **log-space residual**, not a gate. This is the single most
  important difference from failed v1: the head cannot collapse toward
  `local_mean`, and `analytic_blend` is not baked into the architecture.

Frozen hyperparameters (any change is a new spec version):
Adam lr=1e-3, weight_decay=0, batch = one episode (one month of one basin),
max_epochs=200, patience=20, min_delta=1e-6, gradient clip global norm 1.0,
dropout 0.1, hidden 64, seed list `{42, 43, 44}`.

---

## 4. Cross-fitted base (anti-leakage)

Two exclusions, both mandatory:

| level | rule |
|---|---|
| outer | target basin `r` never enters any model that produces a `base` used to train or evaluate the encoder for `r` |
| inner | source basin `s` must not enter the base model that produces `base_{s_i}` / `base_q` inside episodes drawn from `s` |

**Clean definition:** `base_{r,s}` = H2X trained on all data **except** basins
`r ∪ s`. Encoder training for target `r` uses, for every episode from source
`s`, the pair-specific `base_{r,s}`.

**Evaluation base** for target `r` is `base_{-r}` (H2X excluding `r` only) —
the same base used by every K-shot baseline on the frozen v2 tasks.

**Compute note (pre-registered approximation).** Full cross-fitting costs one
H2X run per `{r, s}` pair plus one per `r`. With 5 targets × ≤4 in-protocol
sources that is ≤25 runs. If this is too expensive, the only allowed
approximation is `base_{-r}` for encoder training episodes (drop the inner
exclusion) **plus a leakage sensitivity check**: re-train the encoder on a
stratified subset of episodes with true `base_{r,s}` and report the metric
shift. The approximation must be labeled in the run manifest
(`"inner_exclusion": "approximate"`); results without the sensitivity check
cannot support the "new task adaptation" claim.

---

## 5. Training episodes

- Sources: HUC6 basins other than `r` that pass the v2 eligibility rule
  (≥7 stations in the task component, ≥1 valid same-month task). The four
  non-target primary basins may serve as sources when they are not `r`.
- Episode construction **copies the frozen v2 task builder**
  (`river_graph.experiments.kshot`): same-month, fixed query set, nested
  support `K ∈ {1, 3, 5}` (K=0 has no encoder input and is not sampled),
  `min_query = 2`, support ∩ query = ∅.
- Target basin `r` labels, support, and query cells **never** appear in
  training or hyperparameter selection for that `r`.
- Hyperparameter selection uses a validation slice of **source** episodes only
  (e.g. one held-out source basin or 10% of source months). Never the target.

---

## 6. Evaluation

Frozen v2 tasks only (`experiments/kshot_protocol_v2/tasks/`), untouched.

**Comparators** (all on the same fixed query cells):

| method | role |
|---|---|
| `K=0 base` | `base_{-r}`, no support |
| `local_mean` | mean of support values |
| `mean_bias` | `base` + mean support residual |
| `IDW` | hop-inverse-distance weighted support residual (`residual_idw`) |
| `nearest` | nearest support by river hops |
| `gnn_true` | H2X forward with support values in the DOC channel |
| `analytic_blend` | closed-form gate (baseline only — not a model claim) |
| `support_encoder_v2` | this spec |

**Shuffles on the encoder** (counterfactuals, both required):

- `value-shuffle`: keep support sites, permute `y_s` within the support set.
- `site-shuffle`: keep `y_s`, move them to non-query sites of the same basin
  and month.

**Uncertainty and gate:**

- paired bootstrap ΔMAE vs each comparator, resampling **task-months**
  (cluster bootstrap), 95% CI; success requires the upper bound of
  ΔMAE(ours − best simple baseline) < 0.
- effect size: MAE reduction vs best simple baseline ≥ 10% at K=5.
- curve: `MAE(K=5) < MAE(K=0)` and `MAE(K=5) ≤ MAE(K=1)`.
- shuffle: ours with true support must beat both shuffles.
- **pass = at least 4 of 5 primary basins**; no basin may worsen >5% while
  still claiming overall success. Otherwise fall back to the
  "river-type transferability" story (`protocol.json → fallback_story`).

If fewer than 4 basins pass, do **not** proceed to Task 4 expansion — the
failure mode (which basin, which hop bin, shuffle gap) is the deliverable.

---

## 7. What this spec deliberately does not do

- No cross-time support (Case C): `time_gap` is a reserved channel only.
- No dynamic gates, no multi-scale edges, no GRU — those are H3-line questions.
- No new basins, no new analytes (Task 4/5/6 come after the 4/5 gate).
- No full 571×652 imputation product; predictions remain per-observed-cell.

---

## 8. Acceptance for starting Step 4

1. This file and `protocol.json` agree on tasks, K list, and the success gate.
2. Clean-commit H2X smoke (`tests/test_h2x_smoke.py`) passes.
3. `runtime_code_snapshot` in `manifest.json` is current (re-freeze after any
   code change that affects runs, then re-run `tests/test_kshot_protocol_v2.py`).
