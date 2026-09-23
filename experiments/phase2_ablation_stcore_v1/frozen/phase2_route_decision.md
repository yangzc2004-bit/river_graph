# Phase-2 route decision (frozen 2026-09-24)

This file records the research-route decision taken after the corrected 2B
pilot (R1/R2/R3) and **is the execution authorization** for what follows.
It supersedes the execution scope implied by
`docs/paper/phase2_ablation_spec.md` §1 and `configs/phase2_2c_policy.json`
wherever they conflict (see item 6).

Accepted 2B characterization: **2B passes the engineering and directional
screening, but produced no evidence supporting architecture or topology
claims.** Core facts (pilot only, never paper claims):

1. H2X's advantage over eco_RF is concentrated in E2a/E2b (temporal
   extrapolation / running-network families).
2. E1 clearly loses to eco_RF by 6.1% (stable, 0/3 seeds, CI excludes 0).
3. H2X and no-message show no reliable difference on E2a/E2b/E3 — river
   topology is not shown necessary.
4. H2X and H2E show no stable difference — the encoder is not an
   independent contribution.
5. EcoMLP exits the main line: eco_RF is the trustworthy tabular baseline;
   MLP is kept only as an implementation-stability record.

Main line, further converged: **ecology-informed DOC gap-filling, predictive
uncertainty, and monitoring blind-spot identification.** Model architecture is
no longer a selling point.

## Decisions

1. **The original 240-config Phase-2C confirmatory ablation is NOT started.**
   Its three scientific questions already have sufficient directional
   evidence; more seeds would mostly re-measure near-zero deltas.
2. **2B results are pilot / engineering evidence only** (spec §7) and cannot
   support paper claims.
3. **48 new configurations run as Phase-3 ensemble completion** — explicitly:
   *"Phase 3 uncertainty ensemble completion; not a post hoc Phase 2 claim
   test."* They exist to satisfy the frozen five-seed legal-ensemble
   requirement in `docs/paper/primary_endpoints.json` (Phase-3 uncertainty
   product), not to re-test H2X.

   | model | added seeds | masks | configs |
   |---|---|---|---|
   | H2X | 45, 46 | 8 key | 16 |
   | H2X no-message | 45, 46 | 8 key | 16 |
   | EcoRF | 45, 46 | 8 key | 16 |
   | **total** | | | **48** |

4. **`primary_endpoints.json` stays at v1.** The no-message control is a
   matched-input message-ablation control, not a plain no-graph baseline;
   the pilot has been seen, so redefining the primary gate now would
   introduce a new estimand. All three comparison sets continue to be
   reported separately (`tabular`, `message_ablation`, `encoder`). If
   no-message is ever made a primary-gate object, that requires endpoints v2
   with a revision reason and pilot-seen acknowledgement — and it cannot
   retroactively change 2B conclusions.
5. **Three tools are locked before Phase 3** (Phase 3A model lock; no winner
   is selected on hidden test data): H2X (primary GNN treatment), H2X
   no-message (structural sensitivity control), EcoRF (strong ecological
   tabular baseline). Blind-spot identification is two-layer: each model
   produces its own uncertainty/risk map; only regions where at least two
   models agree in direction are called stable blind spots; regions with
   large model disagreement are reported as `model-sensitive` and excluded
   from core ecological conclusions.
6. **Historical-plan inconsistency, recorded:** `phase2_ablation_spec.md`
   still describes "6 arms × 5 seeds × 8 masks" and
   `configs/phase2_2c_policy.json` freezes 4 GNN arms for the 240-config
   matrix. That matrix is not authorized by this decision; those texts are
   historical plans and remain for provenance. Authorized executions are
   only: the completed 2B pilot (6 arms × 3 seeds), the 48 configs above,
   and then Phases 3B/3C (uncertainty product, high-DOC handling) and
   Phases 4–5 per `primary_endpoints.json`.

## What runs next (after the 48 configs)

- **3B uncertainty product:** full station-month export (the Phase-2
  parquets are observed-cells-only and must not be reused as a full-grid
  product), 5-seed ensemble, central 90% PI, per-cell `uncertainty`,
  `support_count`, `network_distance`, `ecological_novelty`; calibration
  uses train/val information only; hidden DOC only for final evaluation.
  Acceptance = endpoints v1 (overall coverage 80–95%, strata 70–98%,
  top5/top10 reported separately, monotone uncertainty vs distance /
  novelty / support). Failed hidden validation blocks active sampling.
- **3C high-DOC:** report top5 and top10; keep n<20 marked unstable;
  primary high-DOC analyses use Q90 or continuous tail error; seeds never
  count as extra ecological samples.
- **Phase 4 blind spots** only after 3B passes: definition frozen as
  (low support + high uncertainty + ecological novelty) or high-DOC risk;
  acceptance: ≥3/5 basins with adequate strata, direction stable under
  bootstrap and three masks, ≥3/5 basins direction-consistent,
  model-sensitive regions reported separately.
- **Phase 5 budgeted monitoring design:** strategies
  random / spatial-uniform / uncertainty-only / ecological-rarity-only /
  uncertainty+representativeness / network-coverage; K = {0,1,3,5}
  (K=10 exploratory); acceptance: ≥20% improvement vs random in ≥4/5
  basins, no basin worse than 5%, stable improvement in at least one of
  high-DOC recall or MAE, no strategy sees query ground truth.
