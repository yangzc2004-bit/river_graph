# River structure: monthly hydrologic-memory follow-up

Experiment: `experiments/phase4_transfer/doc_river_flow_memory_v1`.

Completed current/preceding-month DOC-flow diagnostics using permitted ST357
source-role cells. No neural model training or recurring automation was started.
The main population has 206 stations, 49 HUC4 regions and 13,846 paired months.
Chronological tests have 72 stations and 4,722 later query months.

The recent-period sensitivity shows +9.80% log1p MAE gain from previous flow
relative to current flow, CI [+2.01%, +18.90%]. Its native gain is +3.01% with an
interval spanning zero. Full-period and temperature-adjusted contrasts do not
establish the same temporal gain. These are small response-model diagnostics,
not gains to the complete DOC neural predictor.

Actual static structural blocks do not reliably predict current/previous DOC
response differences across held-out HUC4 regions. This does not alter the
earlier conditional branching information for DOC level. It means those level
results cannot assign validated morphology-specific lag weights.

The bounded research round closes with a concrete model destination: dynamic
current/history upstream-state messages, source-only development, matched local
structure inputs, and true-upstream/no-message/non-upstream comparisons. The
previous frozen graph readout already failed to improve the full predictor and
had sparse observed upstream support; a new operator must address that issue.

Independent NEON flow requires authorized raw-data download. A single documented
endpoint request returned 403; no credentials were used. That replication remains
pending while available source-role model development can proceed.
