# Unified DOC reconstruction and station adaptation

## Research question

Can a fitted environmental–temporal hybrid use a small number of local DOC
observations to reconstruct records at withheld stations more accurately than
an equally calibrated environmental tree model?

The experiment integrates the existing environmental expert, observation-aware
temporal residual expert, validation-selected fusion, and station calibration in
one saved predictor. The temporal expert uses the strongest existing self-node
configuration. Learned river messages and attention are not added to this batch.

## Design

- Cohort: the existing ST357 DOC dataset, 357 stations and 654 months.
- New station partitions: seeds 142, 143, and 144; each has 232 training,
  54 validation, and 71 test stations.
- Training seeds: 42, 43, and 44 within each partition, giving nine fits.
- Training budget: up to 20 epochs, patience 5; 300 trees per forest.
- Support budgets: 0, 1, 3, and 5 observations per test station.
- Four primary arms: ExtraTrees, calibrated ExtraTrees, hybrid, and calibrated
  hybrid. Both calibrated arms select their own shrinkage on the same validation
  tasks from the same candidate grid.

These are new station partitions of the existing development cohort, not new
observational data, geographically contiguous holdouts, or an external basin.

## Model fitting and information availability

The environmental context expert is selected on validation stations from
ExtraTrees with minimum leaf sizes 4, 2, 1, and a square-root feature variant.
The temporal expert corrects a fixed local ExtraTrees model using station-blocked
five-fold out-of-fold residual targets, ecological inputs, and a 12-month
observation-aware GRU. Fusion candidates are the context expert, temporal expert,
convex log-space mixtures, and an affine log-space combination; validation MAE
selects the final configuration.

Both experts receive training DOC only during validation and test inference.
All target-station DOC remains hidden from expert features at every support
budget. Support labels enter only the explicit station-calibration component:

    log(1 + adapted prediction)
      = log(1 + base prediction)
        + alpha × mean support residual in log space.

Validation stations select alpha separately for each arm and support budget from
0, 0.25, 0.5, 0.75, and 1. No query labels enter adaptation.

Five support candidates are selected from observed dates in the order first,
middle, last, first quarter, and third quarter. Smaller budgets are nested
prefixes. All five candidates are excluded from the query even at K=0, keeping
the query fixed across models and budgets. Support dates may follow query dates;
this experiment evaluates retrospective reconstruction.

## Analysis and interpretation

Primary comparisons are calibrated hybrid versus calibrated ExtraTrees at K=5,
hybrid versus ExtraTrees at K=0, and each calibrated arm at K=5 versus K=0.
Report MAE, RMSE, R-squared, log-space MAE, training-derived Q90 tail error, and
the complete K curve. Average seed losses within each partition, then weight
the three partitions equally. Resample station IDs jointly across partitions
for paired uncertainty intervals, preserving repeated-station dependence.

Show partition consistency and station-level responses, including environmental
novelty and visible upstream support. A context-only fusion selection must be
reported as such. Calibration gains alone do not establish a neural contribution.
The comparison with calibrated ExtraTrees measures the benefit of the complete
hybrid workflow. The supplementary context-only affine and
context-plus-local-forest controls described in `context_affine_control.md`
assess how much benefit remains beyond global recalibration and combining two
environmental predictors.

The three-epoch, 20-tree smoke run checks execution and serialization only. The
nine-fit confirmation batch supplies the scientific results and manuscript
figures. Historical temporal and spatial experiments remain available separately.

## Reproduction

Training uses `scripts/run_ladder.py --experiment unified-doc-spatial`.
`scripts/analyze_unified_doc_spatial.py` produces the matched comparisons and
figures. The saved model exposes both component predictions and a single
query/support inference interface.
