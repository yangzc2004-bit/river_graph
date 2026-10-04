# Fresh station-role DOC confirmation: execution

Production training started on 2026-10-04 at 10:06 UTC. The fixed study plan
is implemented as a compact retained-branch runner, with nine fresh packages:
station-partition seeds242/243/244 and training seeds42/43/44.

The question is whether the current chemistry decoder and chemistry-based
support coordinates retain their gains after the source, validation and target
stations are reassigned. This is a repeatability study on ST357. DOC remains
the prediction target; measured pH and specific conductance are auxiliary
inputs. The fitted models from the earlier development partitions are not used.

## Completed before production

- Short complete-grid run: split242, seed42, one epoch per neural stage and
 20 trees per forest. It produced233,478 full-grid rows and333,040 query rows
  covering20 model curves at K0/1/3/5.
- A second short-run invocation verified and reused the completed package
  without fitting.
- Independent saved-state replay reconstructed the source/native features,
  three nonlinear heads, two daily/chemistry trees, chemical principal
  components, validation choices and all80 query panels. Neural/query products
  matched exactly; tree arithmetic used1e-12 tolerance.
- Full suite after the analysis additions:847 passed,2 skipped. Ruff and
  historical artifact audit passed.

The short run tests execution; its metrics are not confirmation results.

## Commands

```bash
uv run python scripts/run_ladder.py --experiment doc-chemistry-confirmation-v1
uv run python scripts/verify_doc_chemistry_confirmation_v1.py
uv run python scripts/analyze_doc_chemistry_confirmation_v1.py
uv run python scripts/plot_doc_chemistry_confirmation_v1.py
```

The first command supports restarting after an interruption. Completed
submodels are reused after their inputs and artifacts are checked; unfinished
chemistry stages are preserved before being rebuilt. During a live fit, do not
launch a second training runner in this root. Execution sources are preserved
in `code_snapshot/`; subsequent analysis scripts do not change the training
recipe.

## Interpretation

The reported estimator averages training seeds within each station partition
and gives the three partitions equal weight. Paired intervals jointly resample
station identities, including stations shared by partitions. Target outcomes
do not choose a decoder, support representation or support count.

The retained neural model uses a spatial self path without river messages.
Any new chemistry gain is attributed to observed chemistry and station
adaptation, rather than to a newly established river-transport effect. Support
calibration is retrospective. Availability on observed DOC query cells and on
genuinely missing DOC cells is reported separately.

Production results will be written to `analysis/` after all nine packages
finish, followed by the compact figure and a research decision note.

The first full-budget package (split242, seed42) completed in456.4 seconds.
Its independent saved-state replay also passed. The remaining packages are
running serially; a thread follow-up checks completion and performs final
replay, analysis and rendering. Training does not depend on a further user
confirmation.
