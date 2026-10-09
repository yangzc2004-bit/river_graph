# Research decision: detailed ecological composition in source DOC trees

## Result

All9 source-role packages (142/143/144 ×42/43/44) completed. Saved forest
replay, query/label alignment and unchanged preceding predictions were verified.
The11 predefined land-cover fractions cover356/357 stations. COMID alignment
and1,424 broad-total comparisons agree with the existing regime within3.7e-6
percentage points. These are2019 static proxies, not historical land-cover maps.

| Procedure | Source K0 MAE, mg/L | Q90 MAE, mg/L |
|---|---:|---:|
| Retained complete model |1.769053|9.075420|
| Retained strong trees |1.866362|9.394458|
| Matched expanded broad-total trees |1.893667|9.514768|
| Detailed-composition trees |1.868740|9.417979|

Detailed composition reduces MAE1.316% [0.020%,2.639%] versus the matched
69-input aggregate control, with3/3 positive partition directions and8/9
packages. Its Q90 reduction1.017% [-0.025%,2.312%] remains uncertain. These
are5,000 paired station draws, equal partitions after averaging training seeds;
repeated seeds do not add independent ecological samples.

The added broad-total columns themselves worsen the old strong tree by1.463%
[-2.642%,-0.375%]. Therefore the matched-control improvement is not a gain
over the existing best model: detailed trees are0.127% worse than retained
trees [-1.122%,0.836% gain] and5.635% worse than the retained full procedure.
Do not deploy this changed tree or report the1.316% as an overall model gain.

## Next experiment

The composition contains a small measurable signal relative to its same-size
control, but the tree implementation has not converted it into an upgrade.
Test it in the existing ecological encoder, keeping the retained environment
tree and source-OOF residual target fixed. This isolates the information change
without making the neural branch compensate for a worse environmental base.
Both expanded-total and detailed encoder arms use zero new input columns at
initialization, identical parameter counts and the current training budget.

This is a separate source-development study in `doc_composition_encoder_v1`;
the original tree plan and all results remain. No categories are picked from
DOC scores, and old geographical/external queries are not consulted for this
choice. The portable release stays unchanged. The retained native residual
remains the model to improve; unconfirmed nonlinear heads are not combined.

## Verification

`analysis/` contains the complete metric tables and paired effects;
`verification/replay.json` verifies all9 packages. The four-panel figure was
actually inspected.941 tests passed, two skipped; Ruff and historical artifact
audit passed for this tree-comparison version. Large forests/input caches remain
local, with their content bound to each saved product.
