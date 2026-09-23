# AGENTS.md

High-signal facts for working in this repo. Project background and results live
in `README.md`; this file covers what an agent would otherwise get wrong.

## Commands

```bash
uv sync --extra gnn --extra dev          # torch + torch-geometric are the `gnn` extra
uv run pytest                            # full suite
uv run pytest tests/test_hydro.py::test_forward_shape   # single test
uv run ruff check .                      # lint only (ruff target py310); no formatter/typecheck configured
```

- Always run scripts via `uv run python scripts/<x>.py` (uv-managed `.venv`).
- Training entry point is `scripts/run_ladder.py`, **not** `run_gnn.py`:
  `run_ladder` forces `--save-predictions`, writes a provenance sidecar, and
  refuses to overwrite a prediction whose provenance config hash differs
  (unless `--force`). `run_gnn.py --save-predictions` is the legacy opt-in path.

## Data (gitignored, re-derivable)

`data/raw/`, `data/processed/`, `cache/` (~500MB pynhd NHDPlus VAA parquet) are
gitignored. Rebuild order (README "Reproducibility"):
`fetch_doc_inventory → build_graph → build_dataset → fetch_streamcat →
generate_masks → run_freeze`. Raw WQP pulls are slow; `warm_wqp_cache.py` exists.

Tests guard missing local data with `pytest.mark.skipif(..., reason=...)` (e.g.
`tests/test_h3_repair.py`, `tests/test_kshot_protocol_v2.py` needs
`data/processed/mississippi_graph_graphfix_st357.pt`). Follow that pattern:
skip with an explicit reason, never fail or silently pass when local artifacts
are absent.

## Provenance discipline (this repo's core convention)

- Every prediction is a parquet + provenance sidecar with config/dataset/mask
  hashes (`src/river_graph/experiments/provenance.py`, `predictions.py`).
- Protocol freezes (`scripts/freeze_kshot_protocol_v2.py`) write
  `experiments/kshot_protocol_v2/manifest.json` recording `code_commit`,
  `workspace_dirty`, `generator_script_sha256`, dataset/nodes sha256, and
  artifact hashes. **Re-freeze and re-run contract tests
  (`tests/test_kshot_protocol_v2.py`) after any code change that affects runs** —
  a manifest from a dirty/old tree is a known footgun.
- `experiments/frozen_results/` tables are authoritative; recomputed metrics go
  through `scripts/audit_artifacts.py --verify` /
  `scripts/recompute_metrics.py`. The one known corrupt artifact
  (`G0_gcn_none__e1_r20_seed42.parquet`, overwritten in commit `2b67bf0`) is
  excluded from analysis — don't "fix" it silently.

## Repo quirks (would be missed without help)

- **HUC codes come from the `huc_cd` column** of `graph_nodes*.csv`, never from
  station-id prefixes: station 06438000 is HUC2 10, not 06.
- `edge_direction` accepts exactly `both | upstream | downstream` in
  `models/hydro.py` and `models/gcn.py`; anything else raises.
- `env_groups` column selection in `src/river_graph/models/gcn.py` was fixed on
  2026-09-23 (subset columns were mis-indexed; the non-encoder path ignored the
  subset). Group rankings produced before the fix remain invalid — don't build
  on them; re-claim only from fresh runs (`tests/test_env_groups.py` pins the
  semantics).
- Paper claims are frozen in `docs/paper/` (`paper_charter.md`,
  `claim_evidence_matrix.csv`, `primary_endpoints.json`). Primary endpoints must
  not be swapped after results are seen; changes need a new version with a
  recorded reason. The paper cohort is ST-only
  (`data/processed/mississippi_graph_graphfix_st357.pt`) with masks in
  `experiments/masks_stcore_v1/`.
- Predictions are stored only for **observed** cells, not a full 571×652
  imputation grid. Kriging has partial coverage; don't compare its MAE with
  full-coverage models without noting coverage.
- `docs/figures/imagegen_frozen_20260911/` figures are illustrative AI images —
  never cite them as data.
- Model vocabulary: G0 (plain GCN) → H1 (`hydro.DirectedGCNImputer`) →
  H2/H2E (`hydro.TransportGCNImputer`, edge-physics gates) → H2X
  (`gcn.py`, architecture `transport_enc`, env encoder). Frozen H2X base config
  is `BASE_MODEL_CONFIG` in `scripts/freeze_kshot_protocol_v2.py`.
- Phase-2 ablation is frozen in `docs/paper/phase2_ablation_spec.md` +
  `configs/phase2_ablation_stcore_v1.json` (contract tests:
  `tests/test_phase2_controls.py`). Arms: H2 (`env_groups=["hydro"]`), H2E
  (full regime raw, no encoder), H2X (encoder), `H2X_nomsg` (`edge_set="empty"`
  — the **no-message control**: self path retained, only edge messages zeroed),
  `EcoRandomForest`/`EcoMLP` (`baselines.py`, ecological feature set). Phase-2
  outputs go to `experiments/phase2_ablation_stcore_v1/` only; preflight/pilot
  results are never paper claims.
- K-shot v2 protocol: 5 target HUC6 basins
  {103001, 510020, 102701, 101302, 101900}, K ∈ (0,1,3,5), nested same-month
  support with fixed query, task seeds 42/43/44; see
  `experiments/kshot_protocol_v2/protocol.json` — do not re-derive by hand.

## Layout

```
configs/            experiment configs (h3a_v1.json, mvp.yaml)
docs/               execution plans, repair logs, phase reports (dated, English)
experiments/        design docs, masks, predictions, frozen_results, analysis — committed artifacts
scripts/            pipeline + analysis entry points (thin wrappers over src/)
src/river_graph/    data/ topology/ models/ baselines/ experiments/ (library code)
tests/              unit + contract tests
```

Report language: repo docs and committed artifacts are English; commit messages
in this repo mix English and Chinese — match the surrounding history.
