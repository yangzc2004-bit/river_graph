# Antecedent hydro-climate information for high-DOC reconstruction

Retained source-validation diagnostics show that approximately10% high-DOC
cells carry44% of absolute error and85% of squared error. About79% of MSE
remains within stations after removing station mean error. This motivates
testing dynamic exogenous states rather than another static offset or deeper
readout. The decomposition is descriptive; it does not identify event causes.

Keep the current47-input tree and complete neural procedure as fixed references.
Use source partitions142/143/144 × seeds42/43/44, station-hidden train inputs,
the same300-tree settings and exact query cells. Add four bounded descriptors
and their four validity flags:

1. Current flow relative to the same month12 and24 months earlier.
2. Preceding3-month flow deficit relative to preceding12-month flow.
3. Positive one-month flow recovery multiplied by the antecedent deficit.
4. Positive one-month warming multiplied by preceding3-month cool history
   relative to preceding12 months.

All use existing temperature/discharge and their visibility; no water-quality
labels, future months, target normalization or external queries are read.
Temperature differences use Celsius with10-degree bounded compression; negative
flow is unavailable for these new descriptors, while old signed input features
remain untouched. Missing values have explicit validity. These are hydro-climate
proxies, not measured storage, flushing rates or process/causal coefficients.

Compare the full state block with its identical-dimension availability control,
whose four value columns are zero. Both use55 tree inputs. Preserve current
models and all previous experiments. Report source MAE, Q90, bias, paired5,000
station intervals and seed/partition consistency. Match query/role identities
and replay saved forests. If physical values improve the retained strong tree,
rebuild a consistent station-blocked OOF reference before residual training;
otherwise do not claim that extra availability or a favorable control contrast
upgrades the complete model. Subsequent work remains source-only until a fixed
candidate is ready for geographical replication.
