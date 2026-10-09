# Research decision: conditional-median reference with the existing DOC residual

## Matched source results

All nine source packages are complete: partitions 142/143/144 and training seeds
42/43/44. The receiving stations have no water-quality inputs. The forest
partitions and the ecology encoder, observation GRU, native readout, tail loss
and memory method match the retained procedure. The changed reference is the
conditional median of source concentration leaf distributions; source residual
training uses station-held-out distributions.

| Procedure | K0 MAE (mg/L) | Q90 MAE (mg/L) | Signed bias (mg/L) |
|---|---:|---:|---:|
| Retained complete model | 1.769053 | 9.075420 | -0.638608 |
| Retained strong trees | 1.866362 | 9.394458 | -0.611014 |
| Conditional-median reference | 1.853846 | 9.541649 | -0.783988 |
| Median reference + neural residual | 1.779036 | 9.138417 | -0.656750 |
| Median reference + neural/memory integration | 1.778791 | 9.152245 | -0.663915 |

The new integrated procedure improves its own median reference by 4.05%
[2.03%, 6.11%], and reduces its Q90 error by 4.08% [2.24%, 6.80%]. This
demonstrates a useful neural correction to that reference. It does not improve
the actual retained complete procedure: overall gain is -0.55%
[-1.53%, 0.42%], with zero of three partition averages improving and only two
of nine package directions positive. Q90 gain is -0.85% [-1.95%, -0.014%],
also zero of three positive partitions. The larger negative bias is consistent
with the concentration underprediction seen in the bare median.

Its 4.69% gain over strong trees and 2.78% gain over the preceding full model
are real matched comparisons, but neither establishes a gain over the current
best complete model. Five thousand paired station draws retain all months of
each sampled station and average seeds within three equally weighted partitions.
These are source-development results, not geographical or external validation.

## Decision

Keep the current log-trained environmental reference and retained complete DOC
release. The median integration repairs much of its own tail error but does
not turn the small reference-only signal into a complete-model improvement.
Do not export this candidate, repeat the geographical/external matrix, scan
quantiles, or blend it with the retained model by choosing weights on these
results. The existing deployment and manuscript conclusions are unchanged.

The next source experiment will examine optimization of the retained effective
structure. Its nine 30-epoch fits all stopped after 7–24 epochs under patience
5. A single longer, more patient schedule can test whether this stopping rule
misses later useful adaptation of the slow ecological encoder and GRU. This is
an optimization hypothesis, not a diagnosis established by the median study.
Use the original reference/OOF inputs and the original backbone initialization;
only the maximum epoch budget and patience change. No new geographical or
external labels are used in developing that schedule.

## Verification

All nine prediction identities, five-fold source exclusion records, source-only
readout normalization, unchanged initialization/settings/raw information and
parent predictions were verified. Saved neural and memory states replay
bitwise. The conditional-median reference reproduces the prior distributional
study exactly. Analysis uses 5,000 paired station draws; PNG/PDF/SVG figures
were generated and the PNG inspected. The completed version has 963 passing
tests, two skips, Ruff success and historical audit success. Large fitted
caches remain local. Related files are recorded in the existing whitelist;
Git index writes remain unavailable under the current workspace policy.
