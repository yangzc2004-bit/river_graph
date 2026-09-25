# Stage 2C cross-basin controls specification (v1.1 pilot)

Status: **versioned pilot freeze 2026-09-25; execution pending identity review**.
This is the next bounded arm under `spec_v2_cross_basin.md` and the frozen
`cross_basin_tasks_v1` task manifest. It does not alter the task/query masks,
the K ladder, endpoints, or any Phase 0--3 artifact. No task-conditioned or
shared multi-analyte model is included.

## Unit and visibility

Each unit is one `(arm, target analyte, target HUC6, seed)` combination. The
target analyte is trained on source HUC6 `source_fit` labels only, with the
frozen source validation labels used only for a source-only diagnostic score.
All outcome labels in the target HUC6 remain hidden. At prediction time the
GNN may see exactly the source-fit cells plus the declared target-analyte
support cells for K in `{0, 1, 3, 5}`. `extra_values` must be the support
values supplied by the task; query values are never passed to the model.

The runner validates the task manifest file hash, exact `roles_hash` and
`tasks_hash`, dataset/node/edge hashes, and every role mask hash before an
execution can start. Plan-only mode is the default. A prediction product is
label-free; the final evaluator opens hidden query labels once for scoring.

## Fixed arms

| arm | model | ecology input | graph input |
|---|---|---|---|
| `ecorf` | deterministic source-only Random Forest | hydro, month, static, 13 regime columns | none |
| `h2x_full` | `GCNDocModel(architecture="transport_enc")` | all regime groups through the frozen encoder | river edges |
| `h2x_no_graph` | same H2X model | all regime groups through the frozen encoder | empty edge set |
| `h2_no_ecology` | `GCNDocModel(architecture="transport")` | hydro regime group only | river edges |

The no-graph arm is a matched topology control. The no-ecology arm keeps the
transport trunk and river edges while retaining only the hydro regime group.
An arm name does not establish an ecological or topology contribution.

## EcoRF features

EcoRF fits `log1p(y)` on source-fit cells and predicts native units. Its fixed
features are temperature, temperature availability, discharge, discharge
availability, month sine/cosine, the two static coordinates, and all 13
regime covariates. No target-label aggregate is a feature. Missing hydro
values are already represented by the dataset value plus availability mask.
The forest has 200 trees, `random_state=seed`, and `n_jobs=1` for reproducible
execution. No target support is used by EcoRF in this control arm; its output
is repeated over K for the support comparison.

## GNN training and support prediction

Each GNN is fit once per unit with the frozen source `train`/`val` split,
`max_epochs=50`, `patience=10` for the bounded feasibility pilot, `lr=1e-3`, hidden width 64, two layers,
dropout 0.1, and edge direction `both`. The base prediction for K=0 is

```text
model.predict(only_visible=source_train)
```

For K>0, the same fitted model predicts with

```text
model.predict(
    only_visible=source_train,
    extra_visible=support_cells,
    extra_values=target_y[support_cells],
)
```

The source validation labels are never visible in either prediction call.
All K values use the same frozen query cells.

## Provenance and completion boundary

Every unit records the arm config hash, target analyte/basin/seed, role mask
hashes, task manifest hashes, dataset/node/edge/spec hashes, runtime snapshot,
and training budget. Every product row records `model_name`, `analyte`, basin,
month, station, K, support count, `y_pred`, `uncertainty=NaN` (no interval is
estimated in Stage 2C), `uncertainty_status=not_applicable_no_interval`, and
`query_labels_used_for_prediction=false`.

This stage is an arm-completion diagnostic. It cannot unlock Stage 3 until
EcoRF, H2X, no-graph, no-ecology, and the missingness information-decomposition
controls are evaluated under the frozen endpoints. No result may select a
new model, mask, target basin, analyte, or endpoint after query scoring.

## Versioned compute revision

The initial v1 implementation was started but stopped after 15 partial units because a full 200-epoch GNN unit took 8–15 minutes under the available runtime. No query labels were opened and no aggregate result was evaluated. Those partial files are invalidated. v1.1 keeps the task, arms, seeds, visibility rules, endpoints, and model architecture unchanged and reduces only the training budget to 50 epochs with patience 10 for a bounded feasibility pilot. A future confirmation run may restore the longer budget only under a new version.
