# Paper charter (v1, frozen 2026-09-23)

Status: Phase-0 freeze. Hypotheses, primary endpoints, and failure conditions
must not be swapped after results are seen. Any change requires a new version
of `primary_endpoints.json` with a recorded reason. Machine-readable endpoints:
`docs/paper/primary_endpoints.json`. Claim tracking:
`docs/paper/claim_evidence_matrix.csv`.

## Main idea

In a sparse, ecologically non-random river monitoring network, reconstruct the
spatial pattern of dissolved organic carbon (DOC) from river-network topology,
watershed ecological context, and predictive uncertainty — and decide where the
next samples should be taken. Models are instruments for this goal, not the
contribution by themselves.

## Paper question (one sentence)

**Can AI identify ecological blind spots in river DOC monitoring and restore
the key ecological patterns with fewer new samples?**

Target venue: *Patterns* — data-science methodology with scientific,
environmental, and management impact. The contribution is the
uncertainty-aware, ecology-aware reconstruction and sampling pipeline plus the
evidence discipline around it, not a new architecture.

## Hypotheses (fixed)

- **H1.** DOC monitoring gaps have ecological and river-network structure.
- **H2.** River-network topology and ecological context jointly determine DOC
  predictability.
- **H3.** Active sampling that combines uncertainty with ecological
  representativeness is more effective than random sampling.

## Primary endpoints (summary; exact definitions in `primary_endpoints.json`)

| id | endpoint | gate owner |
|---|---|---|
| `doc_mae` | DOC MAE (mg/L), log1p-space MAE co-reported | Phase 2 |
| `high_doc_identification` | high-DOC hotspot precision / recall (top 5% and 10%) | Phases 2/4/5 |
| `uncertainty_coverage` | 90% prediction-interval coverage (overall 80–95%) | Phase 3 |
| `budget_error` | error drop at fixed sampling budget K ∈ {0,1,3,5} | Phase 5 |

Secondary metrics are listed in the JSON and never replace a primary endpoint.

## Experiment map and failure conditions

| hypothesis | experiment | primary scenarios | fail condition (then narrow the paper) |
|---|---|---|---|
| H1 | Phase 4 blind-spot analysis (blind spot := low observation support ∧ high predictive uncertainty ∧ ecological novelty, or high-DOC risk; definition uses training-available information only) | E2b, E3; 5 target HUC6 basins | fewer than 3 adequately sampled ecological strata; direction inconsistent in ≥3/5 basins; unstable under bootstrap/mask resampling → drop ecological-discovery claims, keep a pure data-science methods paper |
| H2 | Phase 2 same-protocol ablation: H2 vs H2E vs H2X vs no-graph vs ecological RF/MLP vs simple baselines; paired ΔMAE with station-clustered bootstrap; mg/L and log space; high-DOC tail; river role × ecology strata | E1, E2a, E2b, E3 | H2X shows no incremental value over ecological-only or no-graph models → stop complex-GNN development; the main line becomes ecology-aware missing-data / monitoring design with models as baselines |
| H3 | Phase 5 active-sampling simulation on a fixed hidden query set: random / spatially uniform / uncertainty-only / ecological-rarity-only / uncertainty+representativeness / network-coverage, K ∈ {0,1,3,5} | 5 target HUC6 basins | no stable gain over random sampling → claim only ecological error structure and blind-spot identification; do not claim sampling optimization |
| support | Phase 3 uncertainty product: full station-month predictions, multi-seed ensemble, calibrated 90% intervals, per-cell `uncertainty / support_count / network_distance / ecological_novelty` | E1, E2a, E2b, E3 | Phase 3 gate not met → Phase 5 does not start (active sampling needs trustworthy uncertainty) |
| scope | Phase 6 external validation (second basin preferred; else held-out HUC6 inside Mississippi; second analyte is out of the main line) | external | external MAE degradation >25% relative to internal, or intervals uncalibrated → narrow scope to within-basin monitoring design |

Phase-2 numeric gate (from the roadmap, fixed here): H2X ≥10% MAE reduction vs
the best no-graph model in at least 2 of {E2a, E2b, E3}; stable gain vs H2E in
at least one extrapolation scenario; gain direction consistent in ≥3 training
seeds; E1 must not degrade by more than 5%; no overall claim resting on one
HUC2 or one station type; every comparison reports paired differences with
bootstrap intervals.

Phase-5 numeric gate (from the roadmap, fixed here): in at least 4 of 5 target
basins, ecology-aware sampling vs random at the same budget achieves ≥20% MAE
reduction, **or** ≥20% high-DOC recall improvement, **or** ≥20% fewer samples
to reach the same error; no basin may worsen by >5% while claiming overall
success; the K ∈ {0,1,3,5} curve must be monotone or free of clear anomalies;
at least one simple baseline must be stably beaten; strategies never see query
labels.

The k-shot protocol v2 `success_gate` (10% at K=5) is **pilot evidence only**
and is superseded for paper claims by the 20% Phase-5 gate above. K = 10 is
exploratory/appendix only.

## Data and evaluation freeze

- **Primary cohort (decision, frozen):** ST-only. Stream stations
  (`site_tp_cd == "ST"`) of the audited graphfix lineage, largest available
  frozen build `data/processed/mississippi_graph_graphfix_st357.pt`
  (sha256 `33474f5584fd99cd6a95289d0e7046ce07f5e2d011857618c67a1c467c4e77c0`).
  357 stations, 324 directed edges (upstream → downstream), 654 monthly steps
  (1972-04 … 2026-09), 22,571 observed DOC cells = 9.67% of the grid.
  Provenance: derived from `graphfix_cached370` (covariate quality rules
  `covariate_quality_v1`, raw-input digest recorded); 13 non-ST sites excluded
  (LK 10, ST-DCH 1, FA-QC 1, SP 1). Station inclusion from the parent build:
  HUC2 in {05, 06, 07, 08, 10, 11}, min 20 DOC samples, min 2-year span.
- **Sensitivity / historical reference only:** the mixed-type 571-station
  results (`experiments/frozen_results/benchmark*.csv`). Paper claims are
  restricted to ST stations; mixed-cohort tables are robustness context and
  carry no headline claim.
- **Scenarios** (semantics of `src/river_graph/experiments/masks.py`): E1
  random missing cells (20/40/60% × seeds 42/43/44); E2a strict future
  forecasting (test = observed months after 2020-12, no future DOC context);
  E2b future reconstruction with a running network (20% post-cutoff cells kept
  as context); E3 spatial extrapolation (20% of stations in the largest
  component held out × 3 seeds). Frozen mask set:
  `experiments/masks_stcore_v1/` (with `masks_provenance.json`); E3
  connectivity is defined on the evaluation graph itself. The older
  `experiments/masks_graphfix_st357/` E3 splits computed the largest component
  on a foreign edge list and are superseded for paper claims.
- **Target basins (H3/Phase 5–6):** the five k-shot v2 primary HUC6 basins
  {103001, 510020, 102701, 101302, 101900} (`experiments/kshot_protocol_v2/protocol.json`).
- **Training seeds (Phases 2–3):** 5 seeds, identical data/mask/early-stopping
  rules across all compared models.
- Prediction artifacts: parquet + provenance sidecar
  (`src/river_graph/experiments/predictions.py`) with config/dataset/mask
  content hashes; benchmark predictions cover observed cells only (the full
  station-month product is a separate Phase-3 export and must not overwrite
  them).

## Claim boundaries (non-claims)

Adapted from `docs/conclusions_and_limitations_20260912.md`; wording in the
paper must respect all of them.

1. No causal, mechanistic, or universal claims — predictive/descriptive
   language only ("associated with", "structured by", "predictable from").
2. H2X gains cannot be attributed to the encoder alone where ecology and
   encoder are added together; only the Phase-2 H2E control licenses that
   split.
3. Transport gating (H2 vs H1) is not shown to improve future reconstruction
   or spatial extrapolation unless Phase 2 reproduces it under the frozen
   protocol.
4. No claim about hydrological headwaters from graph degree alone ("no
   upstream monitoring neighbour" is a monitoring-graph property).
5. ST vs non-ST differences in the mixed cohort are descriptive sensitivity,
   not ST-only training validation; the paper's population is ST stations.
6. Causes of extreme stations (e.g. 06438000) are not established;
   land-use/sampling correlations are not mechanisms.
7. No significance claims from 5 seeds and shared masks beyond the reported
   paired counts and cluster-bootstrap intervals.
8. Group rankings produced under the `env_groups` column-selection defect stay
   invalidated; AI-generated illustrative figures
   (`docs/figures/imagegen_frozen_20260911/`) are never evidence.
9. The overall mean advantage of a model does not imply it wins in every
   region (HUC2=8 ranking reversal is on record).

## Failure → narrowing (fixed)

| stage | if it fails | the paper becomes |
|---|---|---|
| Phase 1 | evidence chain broken | no training; fix provenance first |
| Phase 2 | no incremental H2X value | ecology-aware missing-data / monitoring design; GNN as baseline only |
| Phase 3 | uncertainty not calibrated | no active-sampling stage; reconstruction + descriptive uncertainty only |
| Phase 4 | no stable ecological structure | pure data-science methods paper; no ecological-discovery framing |
| Phase 5 | no sampling gain | blind-spot identification paper; no sampling-decision claims |
| Phase 6 | external validation fails | within-basin monitoring design scope |

## Submission gate (Phase 7)

Every main claim maps to a frozen artifact recomputable in one command; code,
environment, data processing and mask rules public; no illustrative AI figure
used as data; banned vocabulary absent; the pre-submission abstract states the
data-science contribution, the ecological question, and the sampling-decision
value explicitly.
