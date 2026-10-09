# Research decision: antecedent hydro-climate trees

All9 source-role packages, matched forest replay and5,000 paired station draws
are complete. The four-panel figure was inspected.947 tests passed, two skipped;
Ruff and historical artifact verification passed. No old query or product changed.

| Procedure | Source K0 MAE, mg/L | Q90 MAE, mg/L |
|---|---:|---:|
| Retained complete model |1.769053|9.075420|
| Retained strong trees |1.866362|9.394458|
| Matched hydro-availability trees |1.890018|9.465110|
| Antecedent-state trees |1.874402|9.409167|

Physical descriptor values reduce MAE0.826% [0.423%,1.230%] versus their
availability-only control,3/3 positive partitions. But the complete55-input
tree is0.431% worse than the retained strong tree [-1.150%,0.163% gain],
and5.955% worse than the retained complete model. Do not call the control-only
gain a performance upgrade. Do not replace the environmental reference or
rebuild its OOF predictions merely to compensate for this tree regression.

The new information has a small signal relative to its explicit control.
The next separate source experiment feeds these bounded values directly into
the retained neural residual's current-month feature head, keeping the old
environment tree/OOF and existing ecology/GRU fixed in definition. A matched
availability head is included. This asks whether the existing native-MAE
residual optimizer can use the information better than changing the tree.
It adds8 inputs to the existing linear residual head, not another backbone,
gate or larger readout. Its independent specification is in
`doc_antecedent_residual_v1/study_plan.md`. The current portable release remains.
