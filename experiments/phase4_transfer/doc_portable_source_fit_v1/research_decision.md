# Source deployment fitting: completion

All five fixed source packages (seeds42–46) and their portable exports are
complete. The fixed roles contain303 training stations with20,288 DOC cells
and54 validation stations with2,283 cells; there is no source test role. All
five procedures replay saved source-validation predictions to floating-point
precision. The maximum error over25 replays is2.14e−14 mg/L.

The recipe and arithmetic native-scale ensemble were fixed before any external
DOC outcomes. Source validation selected gamma=0 for ecological memory in all
five upgraded packages. Preserve this selection: the complete upgrade and its
native-only ablation coincide in this deployment. K1 alpha is0 for the upgraded
ensemble; K3/K5 alpha is0.25/0.5. These are source-validation choices.

`exports/seed42` through `exports/seed46` contain portable predictor artifacts
for arbitrary new station identities and consecutive calendar months. New
station DOC/pH/conductance inputs are ignored. Explicit K-support correction
requires only designated values. Source labels, preprocessing and station
experience are serialized; ST357 node numbering is not required at prediction.
Large fitted forest/input caches remain local.

The independent02040104 evaluation is complete in
`../doc_external_replication_v1/research_decision.md`. This deployment fit
produces no additional internal confirmation score: historical ST test roles
were legitimately absorbed as source data for the independent basin case.

Subsequent training-source changes require a new execution directory and saved
source snapshot. This completed version and its external predictions remain
unchanged. Next development will use source roles142/143/144 rather than
choosing adjustments from this deployment's external query results.
