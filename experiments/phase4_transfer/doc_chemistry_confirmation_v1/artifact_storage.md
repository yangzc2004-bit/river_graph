# Confirmation artifact storage

Inventory taken after all nine production packages completed, before final
analysis/figure export. The completed study uses the retention scope below;
large local artifacts are preserved without deletion or relocation. Sizes
below use MiB = 2^20 bytes.

## Inventory

The experiment directory occupied **4,170,405,242 bytes (3.884 GiB)** at inspection.
The nine production runs contain **801 files / 3,952,313,279 bytes (3.681 GiB)**;
the separate smoke tree accounts for another 205.66 MiB.

| Production artifacts | Files | MiB | Retention location |
|---|---:|---:|---|
| Fitted ExtraTrees bundles (`*.joblib`) | 36 | 1,978.95 | Keep locally |
| Encoded inputs, native feature caches and representation grids | 45 | 1,079.60 | Keep locally |
| Intermediate/duplicate parquet exports | 27 | 384.55 | Keep locally |
| Canonical observed-query predictions + sidecars | 18 | 57.84 | Commit |
| Canonical full-grid predictions + sidecars | 18 | 231.18 | Commit |
| Neural checkpoints (`*.pt`) | 63 | 10.29 | Commit |
| Small OOF/support/readout arrays | 27 | 2.70 | Commit |
| Configurations, model/calibration JSON and training traces | 567 | 24.13 | Commit |

This retains **693 production files / 341,966,303 bytes (326.12 MiB)** in Git,
before top-level documents, masks, the 1.21 MiB execution snapshot, verification,
analysis and figures. Retaining canonical full-grid products follows the recent
decoder/support experiments; intermediate copies are unnecessary for reading
the results.

The largest individual file is
`runs/split244_seed43/chemical/tree_chemistry.joblib` (91.49 MiB), followed by its
current-daily tree (91.36 MiB). Across all nine runs, forest bundles occupy
1.93 GiB. The two largest array-cache families are
`basis/inputs/temporal_inputs.npz` (493.44 MiB total) and
`native/source_views.npz` (347.26 MiB total).

## Explicit Git retention scope

Paths below are relative to this experiment directory. Expand `RUN` over
`runs/split{242,243,244}_seed{42,43,44}` only.

```text
study_plan.md
execution_note.md
artifact_storage.md
research_decision.md
runtime_snapshot.json
batch_complete.json
progress.json
masks/*
code_snapshot/**
verification/**
analysis/**
figures/**

RUN/**/*.json
RUN/**/*.csv
RUN/**/*.pt
RUN/predictions.parquet
RUN/full_grid.parquet
RUN/source_oof/source_oof.npz
RUN/basis/projector/readout.npz
RUN/chemical/source_training.npz
```

The recursive JSON pattern includes both canonical provenance sidecars and
component completion manifests. Do not use `git add` on the whole experiment
directory. Leave the following outputs local:

```text
_smoke/**
*.log
RUN/**/*.joblib
RUN/basis/inputs/temporal_inputs.npz
RUN/basis/memory/representations.npz
RUN/native/source_views.npz
RUN/native/native_components.npz
RUN/chemical/representations.npz
RUN/initial/full_grid.parquet
RUN/chemical/full_grid.parquet
RUN/chemical/predictions.parquet
```

## Reproduction and model use

The local files remain available for cache reuse, full numerical replay and
inference. Source data/covariate packs follow the repository's existing local
data convention; the confirmation driver reconstructs these fitted stages from
the recorded splits, settings and execution source. Small neural/calibration
states alone are not a complete deployable predictor: the fitted forests are
also required.

Completion manifests and prediction sidecars intentionally retain hashes of
all execution dependencies, including local artifacts. Consequently, the
current full replay and analysis loaders need those dependencies present; a
clean Git checkout containing only this retained subset cannot pass the full
dependency audit until the local bundle is restored or regenerated. Stored
analysis tables and the standalone figure script remain inspectable without
loading the forests. Preserve original hashes; do not weaken sidecars merely
to make a partial checkout look complete.
