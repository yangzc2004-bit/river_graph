# Historical H2X under the current DOC input protocol

The user requested a comparison of historical H2X with the current three-part
fusion, random forest and XGBoost on identical data, splits and input conditions.
This is a new development comparison, not a change to frozen paper endpoints.

## Cohort, roles and endpoints

Reuse ST357 and every monthly-only mask from doc_hybrid_fusion_v1 without
regeneration: five spatial HUC4 regions times three optimization seeds and
three temporal gap lengths (3, 6, 12 months) times three mask/optimization
seeds. Spatial metrics pool the five held regions within seed before averaging
seeds. Temporal metrics average seeds within gap; the overview weights gap
lengths equally. Query cores are reused across temporal gap lengths.
Primary endpoints remain native MAE and log1p MAE. RMSE, predictive R-squared
and existing centered/change diagnostics remain supplementary.

## Fixed model and input adaptation

Reuse the released TransportGCNImputer implementation without editing it:
environmental MLP (32-dimensional embedding), two directed transport-gated
GCN layers (width 64), ReLU, dropout 0.1, and scalar linear head. There is no
GRU, Transformer, residual extension or station-specific parameter.

Every model has the same 15 normalized environmental inputs, the same twelve
months of eight driver/missingness/age/seasonal channels and current seasonal
coordinates. H2X flattens the 12 x 8 history before the original graph trunk.
This gives it the same 113 local raw feature values as the tree controls.
The added eight daily channels used in an earlier study remain exactly zero.
DOC is supervision only; no observed-DOC value or visibility channel is fed.

H2X's original extra NHD physical edge attributes would violate input parity.
Its static up/down gates instead receive the concatenated existing normalized
environmental features of their two endpoints (30 values); these add no new
source information. Both directions preserve their separate gate and linear
weights. Only direct upstream edges from the common role-specific relation
matrix are used, as in the original local graph architecture. Global/path
attention remains an architectural difference, not an extra data source.
Graphs contain the same eligible role nodes as fusion; test nodes and labels
do not enter training. The comparison is explicitly named **H2X matched-input
adaptation**, not a literal reproduction of the old DOC-context/edge-physics
experiment, nor an isolated architecture-only ablation.

## Fitting and controls

One architecture is declared before new test scoring; no hyperparameter search.
Match original fusion's standardized log1p squared loss, AdamW lr 0.001,
weight decay 0.0001, gradient clipping 1, shuffled monthly batches of 32,
maximum 60 epochs, minimum 15 epochs and patience 10. Choose checkpoints
solely using native validation MAE. Compute the same previous-month union
used by the parent schedule even though the log loss uses current anchors.
Retain H2X's original hidden/embedding widths; report parameter counts rather
than claiming matched capacity. The original historical 200-epoch schedule
and DOC-context remasking are not this experiment's schedule.

Keep both original RiverDOCNet and its last explicit-history/direct-readout
update in the table, since both retain three-part fusion. Reuse their audited
immutable fit packages and both tree controls; no refitting is necessary for
identical frozen conditions. There are 24 new H2X fits and 96 reused packages.
Seal all 24 validation selections before generating any new test prediction.

## Provenance and verification

Freeze protocol, source snapshot, dataset/nodes/masks and reused-package hashes
before fitting; launch through an isolated run_ladder runtime. Store each
prediction parquet and provenance sidecar. Replay all 120 selected models on
validation and test queries, independently recompute metrics, and verify exact
query/truth/scaler parity. Report paired HUC4-block bootstrap differences
using the established sufficient-statistics method, with development scope
and the five-cluster spatial limitation stated. Preserve all prior results.
