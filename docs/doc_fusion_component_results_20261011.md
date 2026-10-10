# DOC reconstruction: environmental encoding, time modelling and fusion

Date: 2026-10-11. Status: completed internal development experiment; suitable for design justification with the scope below.

## Decision

Retain the environmental MLP + GRU + concatenation configuration, with the existing graph Transformer spatial module, as the working reference. The experiment does **not** establish that these components are optimal. Linear environmental projection and a flattened-history MLP remain strong, inexpensive alternatives. More elaborate temporal and fusion recipes did not show a reliable reduction in the predefined primary endpoint.

The source-validation-selected pipeline obtained a test MAE of **2.19033 mg/L**, compared with **2.18480 mg/L** for the reference. Its relative improvement was **−0.253%**, with an exploratory 95% paired region-bootstrap interval of **[−3.487%, +2.192%]**. The final selection improved strictly in only one of five regions. This does not justify adopting the more elaborate pipeline on accuracy grounds.

This memo supplements [the design rationale](doc_fusion_design_rationale_20261010.md) and [the experimental plan](doc_fusion_component_experiment_plan_20261010.md). It records results rather than rewriting the previous rationale as if these outcomes had been known beforehand.

## What was tested

The study used the ST357 Mississippi dataset, holding out whole HUC4 regions 1013, 1019, 0708, 1030 and 1101, with seeds 42, 43 and 44. There were **165 unique fitted models**, **7,262 observed station-month query cells**, and **104 query stations**. Every candidate covered exactly the same queries. The primary endpoint was native-unit, cell-weighted DOC MAE; seed errors were averaged rather than predictions ensembled.

The common spatial module was the existing directed graph Transformer. Inputs were 15 environmental/regime and coordinate features, plus a 12-month window containing temperature, discharge, observation masks, observation ages and seasonal encodings. Target DOC values were absent from model inputs and training. Training used source-station graphs; inference retained source stations as covariate-only context alongside the target graph. This is not yet an independent-river, target-only inference test.

The three stages followed a frozen order. Environmental encoding was tested with GRU and concatenation fixed. Time candidates used the environmental module selected by source-validation MAE in each outer fold. Fusion candidates additionally used the selected time module. All stage choices were frozen before test scoring; a diagnostic without environmental inputs and a current-only time diagnostic were excluded from full-input selection by the original protocol. Checkpoint reuse avoided duplicate fits. Consequently, comparisons are valid **within each stage**, but raw numbers across stages have different parents. This was not a full factorial search.

All candidates shared the optimizer, early-stopping rule, maximum 60 epochs, masks and normalization procedure. No selected checkpoint reached the 60-epoch ceiling. Parameter counts differ, so the results compare these concrete recipes, rather than all possible implementations of an architecture family.

## Environmental information matters; nonlinear encoding is not settled

| Environmental branch | Test MAE, mg/L |
|---|---:|
| Learned constant; environmental inputs removed | 3.67460 |
| Linear projection | 2.17487 |
| MLP reference | 2.18480 |
| Residual MLP | 2.23325 |

Replacing the constant branch with the MLP reduced MAE by **40.54%**, with improvement in all five regions. This supports retaining the environmental information package; it does not isolate ecological descriptors from hydraulic regime or coordinates, nor identify a causal environmental mechanism.

Linear projection had a slightly lower point estimate than the MLP: **0.455%** relative improvement, interval **[−1.498%, +4.825%]**, with three of five regions improved. The evidence therefore supports environmental conditioning more clearly than it supports nonlinear MLP superiority. Residual MLP was worse on aggregate despite having the same total parameter count as the MLP. A defensible writing claim is: environmental encoding supplies geographically varying background information; a compact MLP remains a practical reference, while the need for nonlinearity remains unresolved.

## Temporal memory is plausible, but GRU superiority is unproven

| Time branch; environmental parent selected by validation | Test MAE, mg/L |
|---|---:|
| Current-month MLP diagnostic | 2.23569 |
| Flattened 12-month history MLP | 2.20276 |
| GRU reference within this stage | 2.20830 |
| LSTM | 2.24821 |
| Compact causal temporal Transformer | 2.22544 |

History MLP improved on GRU by **0.251%**, interval **[−3.631%, +3.101%]**. The temporal Transformer was **0.776% worse** on aggregate, interval **[−4.857%, +2.121%]** for improvement, despite winning in three regions. LSTM also lacked a stable advantage. GRU beat the current-only point estimate, but uncertainty does not establish a transferable memory benefit. Current-only and history branches use different operators, so this is not a same-operator causal isolation of memory.

GRU can be retained for its compact recurrent representation of antecedent conditions; it should not be described as experimentally optimal. The spatial graph Transformer and the temporal Transformer answer different modelling questions: this experiment tests temporal attention while holding spatial attention fixed.

## Elaborate fusion did not reliably beat concatenation

| Fusion; environmental and time parents selected by validation | Test MAE, mg/L |
|---|---:|
| Concatenation + projection | 2.19432 |
| Residual concatenation | 2.20338 |
| Environment-conditioned time representation | 2.19843 |
| Learned gating | 2.19088 |

Gating had the lowest point estimate, but its gain over concatenation was only **0.157%**, interval **[−0.631%, +1.008%]**. Environmental conditioning and residual fusion were worse on aggregate. Conditioning modifies the final temporal representation, not recurrent gates; conditioning and residual fusion each add 600 parameters. These results support simple concatenation as the current practical choice, without proving that interactions between environmental and temporal information are absent.

Source-validation choices were heterogeneous: environment MLP/linear/MLP/residual/residual; time history-MLP/GRU/Transformer/GRU/GRU; fusion concatenation/conditioning/gating/conditioning/conditioning, in region order 1013/1019/0708/1030/1101. Frequent selection by source validation did not guarantee improvement in the held-out region.

## Input availability is a more consequential unresolved issue

Observed DOC query months had current temperature available in **95.6%** of cases and discharge in **85.0%**. The **11,076 naturally unobserved DOC cells within query stations' first-to-last query spans** had temperature available in only **32.6%** and discharge in **71.4%**. No hydrological observations in the preceding 12-month window occurred in **11.9%** of these gaps, versus **0.29%** of observed queries. These are covariate-mask diagnostics; DOC values in the gaps are unknown.

Two separately frozen, fixed-checkpoint stress tests therefore accompanied the primary comparison. Neither retrained or selected models. Erasing temperature at target query months raised GRU MAE to **2.31698** and time Transformer MAE to **2.42638**. Erasing both temperature and discharge at query months gave GRU **2.38837**, history MLP **2.31197**, LSTM **2.30939**, and current-only MLP **2.32208 mg/L**. Historical non-query measurements were retained, erased months disappeared from subsequent windows, and ages were recomputed. Memory did not automatically rescue missing current inputs under these interventions.

The separate pipeline stress test erased target temperature throughout all months: reference MAE became **2.30164**, versus **2.33378** for the selected pipeline. Erasing both target dynamic channels throughout all months produced **2.36451** and **2.36496**, respectively. Source covariate context remained available. These interventions are stress tests on observed DOC queries, not accuracy estimates for naturally missing DOC cells, and they do not isolate individual component effects in the selected pipeline.

The next design priority is validation under realistic covariate missingness and genuinely independent rivers. Improving the availability of dynamic environmental inputs may matter more than adding model complexity. The present evidence cannot yet support the intended continuous, arbitrary-location river DOC field: outputs were assessed on observed monthly station cells, not arbitrary reaches or true missing-cell ground truth.

## Computation, verification and interpretation limits

A serial, one-thread benchmark on a representative 357-station × 654-month grid gave median model-only inference times of **0.827 s** for history MLP, **1.072 s** for GRU and **1.313 s** for temporal Transformer, over three passes after warmup. Feature assembly, graph construction and I/O were excluded. This is a station-grid cost comparison for one region's parent choices, not a universal deployment latency estimate.

All **165 checkpoints** replayed validation and test predictions with **zero maximum difference**. An independent calculation against the original DOC grid reproduced primary MAE exactly; both input diagnostics also underwent independent arithmetic checks. The final suite passed **1,528 tests**, with three explicit skips and eight warnings. Ruff passed. Dataset, masks, configurations, code snapshots, predictions and selection receipts are hashed. Existing paper freezes and primary endpoints remain unchanged.

The five regions belong to one major basin and had been inspected in earlier development work. Bootstrap intervals use only five region clusters, are exploratory and have no multiplicity correction. Results apply to the declared inputs, compact recipes and optimization budget. Secondary improvements in RMSE and upper-tail error cannot replace the predefined MAE endpoint. Global transfer, out-of-domain climate or land-use regimes, target-only graphs and arbitrary river locations require separate evaluation.

Full evidence: [technical report](../experiments/phase4_transfer/doc_fusion_component_comparison_v1/analysis/research_decision.md), [primary table](../experiments/phase4_transfer/doc_fusion_component_comparison_v1/analysis/summary.csv), [paired uncertainty](../experiments/phase4_transfer/doc_fusion_component_comparison_v1/analysis/paired_comparisons.json), [audit](../experiments/phase4_transfer/doc_fusion_component_comparison_v1/analysis/audit.json), and [source receipts](../experiments/phase4_transfer/doc_fusion_component_comparison_v1/analysis/sources.json).
