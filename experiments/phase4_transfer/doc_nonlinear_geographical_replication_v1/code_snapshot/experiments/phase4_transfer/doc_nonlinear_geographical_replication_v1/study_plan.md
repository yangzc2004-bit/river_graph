# Fixed nonlinear-head geographical replication

## Candidate and task

The source-only32-unit nonlinear readout gives0.468% MAE reduction over the
actual retained complete model in partitions142/143/144 × seeds42/43/44, with
seven of nine positive directions. Its station interval crosses zero. Freeze
that single head before evaluating geographical transfer. The full release is
not changed by the development result.

Use the same five HUC4 tasks1013/1019/0708/1030/1101 and cyclic-next validation
region, with seeds42/43/44/45/46. The target region's DOC/pH/conductance and
chemical history are hidden at K0. The model receives environmental and causal
hydro/temporal information plus source station context. These ST357 tasks have
already been evaluated in the prior version. This is retrospective replication
of a source-fixed candidate, not a new independent test or external validation.

## Reuse and fitting

Reuse each matching parent region/seed's saved source-only environmental tree,
OOF reference and source backbone. Verify the parent stages before reuse. The
new neural fit starts from that parent's native candidate's initial encoder,
GRU and decay states, not its final candidate weights. Both output paths start
at zero. Its initial states match the saved linear candidate's starting states
and exclude the parent's target region from source fitting.

Keep source input views, native MAE/tail weight2,30 epochs, patience5, learning
rates, residual-scale grid and validation-selected ecological-memory method.
Only the fixed nonlinear readout differs. Cache the completed neural fit before
export. Save label-free point components and fitted memory before opening test
DOC for scoring or the designated support correction.

Preserve the parent's five primary procedures and all support curves unchanged.
Add nonlinear neural-only and nonlinear complete products, using the identical
all-observation K0 query and fixed-query K0/1/3/5 adaptation sets. The existing
simple support-adapter family selects strength on the held validation region;
only prescribed test support is opened. There is no per-region or per-K model
winner selection.

## Analysis

Compare nonlinear complete against the retained linear complete model, preceding
complete model and strong station-hidden trees. Also compare matched neural
heads. Average seeds within region and then weight five regions equally.
Use5,000 paired station-bootstrap draws, retaining all months per station.
Report MAE/RMSE, station-equal error, Q90 error/bias/recall, full K curves,
ecological novelty, hydro availability and source-similarity strata.

Report every region and failure. Geographical results belong to this fixed
version and do not change this head's dimensions, training objective or policy.
Further mechanisms return to source-development roles. Retain the delivered
portable/external/manuscript release until the complete evidence warrants an
upgrade. New outputs live only in this versioned directory.
