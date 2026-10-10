# DOC confluence/storage operator: completed source-development round

The existing GNN river branch now has explicit nonredundant tributary mixing and
storage/confluence attenuation.45 fits, nine packages and all evaluations are
complete in `experiments/phase4_transfer/doc_confluence_storage_v1`.

Primary MAE 1.716517 mg/L versus ordinary attention 1.717226:0.041% improvement,
95% station interval[-0.058%,0.156%]. Mixing-only 1.716458 is virtually identical;
storage adds no established gain. The preceding environmental-state branch
remains better at 1.714729. Q90 also does not improve over either river comparator.
Retain the released model and preceding candidate, not this operator.

The new experiment contributes a controlled propagation comparison. Nested
catchment duplicates account for about 40% of candidate–lag slots. Only 22–52%
of query months have usable upstream DOC; multiple represented tributaries cover
7–29%. Flow-based mixture weights are possible in30–69% of retained slots;
other groups use drainage-area proxies. Matched real versus nonancestor gain
0.229% has a station interval crossing zero.

Exact cache reconstruction/checkpoint replay and independent analysis passed.
Full pytest 1474 passed,3 skipped; Ruff and historical artifact audit passed.
Two scientific figures were viewed and legend overlaps repaired. Training and
old products were unchanged. English `research_decision.md` records complete
comparisons, diagnostic limitations and the next observation-time alignment
question. No automation was restarted and no geographic/external tests were
used for development.
