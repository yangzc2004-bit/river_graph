# Source DOC state keys: research decision

## Decision

Close this mechanism without geographical training. Retain the preceding
20-candidate current-availability model. Adding source DOC state to attention
keys learned nonzero coefficients, but did not improve overall error or
establish an additional high-DOC benefit. Increasing the donor pool and
enriching its selection keys have both been tested; neither should replace
the geographically confirmed current-availability version.

Nine fixed source packages (142/143/144 × 42/43/44) completed: 27 neural fits,
90 reused double-held references and no new forest. Candidate arrays,
source-only RMS, initial weights, predictions, fusion and diagnostics replay
bitwise. The analysis uses 5,000 paired station draws, seed means within
partition and equal partition weights. All nine zero-key controls reproduce
the preceding native and complete predictions exactly. The final PNG was
inspected. Validation: 1,009 tests passed, two skipped; Ruff and the historical
artifact audit passed.

## Matched comparison

| Procedure | Complete MAE, mg/L | Complete Q90 MAE, mg/L | Native MAE, mg/L |
|---|---:|---:|---:|
| Retained complete | 1.769053 | 9.075420 | — |
| Preceding current-availability | **1.723002** | 8.895024 | **1.726375** |
| Zero additional state keys | **1.723002** | 8.895024 | **1.726375** |
| Seasonal state keys | 1.725698 | 8.884154 | 1.727364 |
| Current and seasonal state keys | 1.725912 | **8.879700** | 1.727058 |
| Strong station-hidden trees | 1.866362 | 9.394458 | — |

The isolated complete-model gain over preceding/zero keys is
**-0.169% [-0.525%, 0.155%]**, five of nine packages improving, one of three
partition means and one of three seed means positive. Native-only gain is
-0.040% [-0.502%, 0.417%]. Complete Q90 gain is 0.172%
[-0.172%, 0.563%]; native Q90 gain is 0.144% [-0.252%, 0.560%]. None
establishes an incremental improvement.

Current keys do not outperform seasonal-only keys: complete overall gain
-0.012% [-0.280%, 0.238%], Q90 gain 0.050% [-0.239%, 0.317%]. Native
differences are likewise unresolved. Accumulated gains over retained complete
(2.439%) and strong trees (7.525%) include earlier changes; these must not be
presented as the contribution of source-state keys.

## Implementation interpretation

Actual donor values, validity, receiving query, aggregate descriptors,
ecological prior and zero prior are identical across controls. Two channels
add 128 coefficients, yielding 38,028 trainable parameters. New coefficients
start at zero. Current-key coefficient norms range 0.287–0.840; seasonal-only
norms range 0.123–0.784; zero controls remain exactly zero. All fits select
residual scale 1. This is a completed learning experiment rather than a
disconnected or accidentally disabled branch. Increased representational
capacity alone did not make donor allocation more useful.

The source-only RMS is fitted from permitted OOF log1p residuals. Query fold
A is absent from its library, and donor reference forests exclude A and B.
Fixed training-library statistics can use the permitted spatial-task source
record; inference reads only contemporaneous source observations with those
statistics fixed. This does not turn the spatial task into prospective
forecasting. Receiving DOC, pH and conductance remain unavailable.

## Next work

Move the geographically confirmed current-availability architecture into a
new portable inference product, preserving the existing deployment release.
Save its source preprocessing, native and relative residual libraries and
source hydrology; support arbitrary receiver identifiers and calendars.
Refit this fixed architecture on the same source deployment roles as the
earlier portable model, then compare them using the existing external case
without changing the model from external outcomes. The external case has
already been inspected for the previous release, so an updated replication
must explicitly record that exposure. Do not call it a newly blinded test.

The internal geographical gain is already supported; complete the deployment
and comparative evidence rather than running another key-capacity scan. Keep
the pending-submission whitelist scoped and large fitting caches local.

Reproduce with `scripts/analyze_doc_source_value_key_attention_v1.py
--bootstrap-draws 5000` and `scripts/plot_doc_source_value_key_attention_v1.py`.
