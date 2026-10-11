# Official-backbone DOC comparison, 2026-10-11

The user authorized direct comparison with related-work models. This experiment
runs pinned official SAITS and GRIN backbones on the same river-DOC tasks as the
fixed environment-MLP + driver-GRU + graph-Transformer + concatenation model.
Published MAEs from different datasets are not comparable to this experiment.

## Competitors and adaptations

| Family | Official source | Version | Retained backbone and changed interface |
| --- | --- | --- | --- |
| SAITS | [Paper](https://arxiv.org/abs/2202.08516), [official repository](https://github.com/WenjieDu/SAITS) | `660b87f19c1277065f314f24134f646229e89ca9` | Two diagonally masked self-attention blocks and attention-weighted combination; 23 covariate channels plus an always-hidden DOC channel; station windows treated as local samples. |
| GRIN | [Paper](https://arxiv.org/abs/2108.00298), [official repository](https://github.com/Graph-Machine-Learning-Group/grin) | `4a28afbb092600b6e6abeeabaaf67e87dbd1ed6e` | Official bidirectional graph-recurrent imputation core with its exogenous-input interface; zero initial state (`n_nodes=None`) and no station embeddings (`embedding_size=0`) permit changing node counts. |

No core architecture files are rewritten. A namespace loader bypasses optional
package re-exports that otherwise require unrelated training frameworks.
`einops==0.8.1` is installed in an isolated cache, without changing the shared
project environment. Executed upstream files and dependency files are hashed
and copied into the isolated runtime workspace before fitting.

These methods are named `saits_doc_covariates` and `grin_doc_covariates` in
results. They are **covariate-only adapters**, not reproductions of the papers'
original missing-value experiments. Both papers primarily address imputation
with observations of the variable being reconstructed. Our zero-DOC-context,
unmonitored-node deployment changes that information regime. The shared-input
track measures transfer of their backbones under this declared regime; it does
not establish superiority over their native imputation settings.

[CoFILL (IJCAI 2025)](https://www.ijcai.org/proceedings/2025/320) and its
[official implementation](https://github.com/joyHJL/CoFILL), commit
`d461621b213df7d682034a1da99721f2ba65b1ab`, were inspected. Its implementation
uses feature/node embeddings, adaptive node vectors, and dataset-specific graph
loaders. It is excluded from this first run before observing new target results:
it needs additional adaptation for this changing-node covariate-only interface,
or a separate within-network DOC-assisted imputation track. Further recent
competitors and reserved external river systems remain necessary for a SOTA
assessment.

## Common evaluation contract

- ST-only dataset: `data/processed/mississippi_graph_graphfix_st357.pt`.
- Five previously inspected development HUC4 tasks: 1013, 1019, 0708, 1030, 1101.
  Existing geographical masks are reused, with disjoint source, source-validation,
  and target station sets. This is not a fresh independent confirmation cohort.
- The same 15 environmental features and 12-month, eight-channel driver windows
  supply all models. Window channels retain temperature/discharge values,
  observation masks, observation age, and seasonal encoding. Static environment
  features are repeated along the window for official models' native interfaces.
- DOC is a supervision target only. Neither source nor target DOC history enters
  an input. Covariate normalization and log1p DOC normalization use source data.
  Target labels enter only scoring after all source-side choices are frozen.
- Training uses the source-induced graph. GRIN inference uses source plus the
  relevant held-out nodes' covariates. Its adjacency supports are recomputed for
  each graph; learned parameters do not contain a station axis. SAITS operates
  on individual station windows and receives no adjacency.
- Both new families run three training seeds (42, 43, 44) at two frozen widths:
  SAITS 64/128, GRIN 32/64. Total: **60 new candidate fits**, plus 15 replayed
  checkpoints for the fixed fusion reference. Parameter counts, epochs and
  elapsed costs are reported. The reference benefited from earlier development;
  this is not an equal-compute or exhaustive tuning comparison.
- Shared optimizer: AdamW, learning rate 0.001, weight decay 0.0001, gradient
  norm cap 1, maximum 60 epochs, minimum 15, early-stop patience 10. SAITS batches
  contain up to 256 observed source station-months; GRIN batches contain eight
  window-ending months. Per-batch loss is weighted by observed source-label count.
- SAITS retains official observed-reconstruction loss plus masked-imputation
  loss; only observed source DOC at the final month is the DOC loss target.
  GRIN retains main-output MAE plus the four auxiliary-head masked MAE terms at
  that month. Labels are normalized log1p DOC. These target, optimizer and
  checkpoint choices differ from the original papers' training recipes.
- Checkpoints minimize source-validation native MAE. Width selection uses mean
  validation native MAE across three seeds, with exact ties resolved by log1p MAE
  and then lexical recipe ID. Every basin's selections are hashed before any
  new target prediction is scored.
- Joint primary endpoints: native MAE and log1p MAE. Secondary endpoints:
  RMSE, predictive R², signed bias, station/basin macro MAE, source-Q90 tail MAE,
  and observed temporal/spatial structure diagnostics. Original paper endpoints
  in `docs/paper/` are not changed.
- Compare the identical 7,262 hidden observed station-months at full coverage.
  Metrics are computed per seed and then averaged; repeated seeds are not an
  implicit prediction ensemble. Five-basin paired bootstrap intervals use 5,000
  draws and are exploratory, unadjusted intervals with only five clusters.
- Naturally unobserved river cells have no ground truth here. These comparisons
  do not validate arbitrary virtual-station locations or a continuous river field.

## Provenance and validation

Output root: `experiments/phase4_transfer/doc_related_work_v1/`.
Runtime: `cache/doc_related_work_v1/runtime/`.

The original `scripts/run_ladder.py` and its frozen historical SHA remain intact.
An isolated dispatcher with that entry-point name invokes the new runner.
The protocol records the original and isolated code hashes, official revisions,
source file hashes, data/mask hashes and all reference-checkpoint dependencies.
The manifest records repository commit, dirty-workspace status and generator
hash. Every prediction parquet has a provenance sidecar. Completed fits and
test products reject changed hashes or configuration on resumption.

Tests exercise finite forward/backward computations, zero DOC inputs, final-month
supervision, GRIN permutation equivariance and changing node counts, checkpoint
replay, target-label poisoning, and corrupted/changed-run rejection. Contract
tests recheck the freeze and historical masks/dispatcher before launch. The
existing generic-baseline batch continues independently.

```bash
uv run pytest tests/test_doc_related_work_v1.py
uv run ruff check .
uv run python scripts/start_doc_related_work_v1.py --freeze-only
uv run pytest tests/test_doc_related_work_v1.py tests/test_kshot_protocol_v2.py
uv run python scripts/start_doc_related_work_v1.py
uv run python scripts/status_doc_related_work_v1.py
```

## Launch verification

The suite passed with **1,545 tests passed, four skipped** and eight existing
warnings. Ruff passed. The post-freeze adapter/baseline/K-shot contracts passed
with 25 tests. Protocol hash:
`8d8cb04e075cd261a1da42e42924337fbabdf431c2a64d5efdccc485a4e0c763`.
The detached worker was launched with PID 61671; use the status script to check
its current state. The baseline worker remains separate.

No result ranking or SOTA claim is recorded in this execution note. Status,
trace files and receipts identify the actual completed work. Once all fits and
the audit succeed, `analysis/summary.csv`, `paired_comparisons.json` and
`research_decision.md` hold the comparison, with explicit scope limits.
