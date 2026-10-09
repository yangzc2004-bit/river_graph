# Portable current-source DOC reconstruction

## Purpose and fixed architecture

Package the geographically confirmed 20-candidate current-availability model
for arbitrary new stations. Its equal-region K0 error improves 1.478% over the
retained complete model and 4.109% over strong station-hidden trees. Preserve
the earlier release and its scores. Wider source pools and source-state keys
did not establish additional value and are excluded.

Retain the environment tree, ecological/self encoder, observation-aware
12-month GRU, three matched source-innovation readout features, two 32-dimension
attention heads, native tail-aware loss and ecological memory fusion. Individual
values are source OOF log1p residuals including permitted seasonal means;
current donors do not require an earlier same-season observation. The query
uses ecology and daily hydro. Use 30 epochs, patience 5 and the existing scale
selection. No new architecture or model selection from external outcomes.

## Source refit and inference

Reuse the completed source-only deployment roles and saved base stages from
`doc_portable_source_fit_v1`: 303 training stations and 54 validation stations,
seeds 42–46. Retain their base tree, source-blocked OOF prediction, backbone
initial states and preprocessing. Fit ten complementary pair references per
seed to build query-fold-excluded source banks, then one fixed attention model
per seed. The receiving validation chemistry remains hidden in inputs. Its
labels select early stopping, the existing ecological-memory family, support
shrinkage and empirical interval scale only.

Persist a native-unit aggregate innovation library, a relative log1p individual
library and source daily hydro, all aligned by station names and actual calendar
months. New receivers supply ecology, hydro, visibility and daily features;
DOC/pH/conductance fields are ignored at K0. Absent source months have no donor
observations and a zero-prior fallback. No positional ST357 node IDs are
required. K1/3/5 accept only explicit support and source-selected adapters.

Expose `predict`, `predict_components`, `save` and `load`. Components distinguish
environment, local temporal readout, dynamic source observations, static source
memory and final DOC. Attention is ecological/hydrological donor allocation;
it is not a physical upstream transport coefficient. River context features
retain the old inference path as a separately identifiable information source.

## Evidence and delivery

Verify hidden labels, future inputs with fixed source libraries, new receiver
counts/order/calendar, component closure and exact saved-state replay. Reproduce
the existing geographical fitted products through the portable facade before
launching the source refit. Use the fixed external 02040104 inputs and queries
for an updated comparative replication; preserve the previous results and
explicitly state that the external case was already inspected for the previous
release. No external label contributes to fitting, K0 calibration or choice
of this architecture. A new prospective external test requires a new case.

Report source validation and updated external MAE, station-equal error, Q90,
bias, fixed-query K curves and source availability. Compare complete procedures
and native-only ablations, preserving the same tree and source roles. Empirical
coverage and width use only saved source-validation residuals. Update manuscript
and versioned figures after the fixed comparison. Keep large fitting caches
local and the relevant small results in the scoped pending-submission list.

## R1 execution repair

V2 completed all ten seed42 complementary pair forests, then stopped before
its first neural epoch: unobserved source-grid OOF cells contain NaN, and their
attention reference was not zero-filled. R1 keeps observed OOF values exactly
and fills non-loss cells with zero, matching the confirmed source architecture.
No architecture, budget, label visibility or scientific comparison changes.
The V2 execution sources/log remain intact. Verify and reuse its ten complete
references unchanged; only40 new pair forests remain for seeds43–46.
The portable facade and25-fit geographical replay already passed under V2.
