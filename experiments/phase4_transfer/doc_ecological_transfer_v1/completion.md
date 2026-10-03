# Ecological residual transfer: completed DOC iteration

The existing interaction model now has a source-station ecological residual
memory. Nine watershed descriptors identify similar source stations; each donor
station contributes equal total weight. The memory learns a constant or
concentration-dependent correction from source station-blocked OOF context
errors and competes with the existing recurrent correction.

All 36 memories were fitted on nine frozen expert packages. The four controls
cross global versus ecological donors with bias versus affine correction.
Source-validation K0 MAE selects donor count, ridge and mixing proportion.

| Model, with the existing GRU support adapter | K0 MAE | K5 MAE |
|---|---:|---:|
| Previous interaction | 1.835240 | 1.581605 |
| Global bias | 1.834580 | 1.584386 |
| Ecological bias | 1.819789 | 1.583988 |
| Global affine | 1.835240 | 1.581605 |
| Ecological affine | **1.809217** | 1.587188 |

MAE is in mg/L. Seeds are averaged within each partition, then partitions are
weighted equally. The ecological affine model reduces K0 error by **1.42%**,
with a paired station-bootstrap interval of **0.52% to 2.45%**. All three
partition means and eight of nine fits improve; 104 distinct target stations
improve and 68 worsen. The gain is distributed across partitions, rather than
being confined to the previously difficult validation partition.

Ecological bias alone improves K0 by 0.84%. Affine ecology improves on that
control by a further 0.58%, with an interval of 0.03% to 1.20%. The global
affine procedure selects zero mixing in all nine packages and reproduces the
interaction model exactly. Its comparison therefore repeats the same reference
evidence, rather than providing another independent replication.

The new gain comes mainly from ordinary concentrations: non-tail MAE decreases
by 0.0278 mg/L (interval 0.0107 to 0.0482 improvement). High-DOC Q90 MAE changes
from 7.4371 to 7.4318 mg/L; its paired interval crosses zero. This iteration
improves zero-observation spatial reconstruction but does not resolve high-DOC
underprediction.

K5 does not benefit: affine ecology MAE is 1.587188 versus 1.581605 for the
interaction model, a 0.35% worsening with an interval spanning -1.00% to
0.28% relative improvement. The regional prior's mixing proportion is selected
at K0 and carried unchanged into all support levels. The next integration
therefore lets source-validation choose the mixture separately after K1/K3/K5
support adaptation, keeping this v1 result as its fixed-mixture control.

The memories and their parameters remain available. No candidate is promoted
using outer-query results. These previously examined station partitions remain
development data, and K-shot reconstruction is retrospective.

## Implementation and reproduction

The source-only ecological scaler handles -1/nonfinite missing fields; a target
with fewer than five ecological descriptors uses the global donor pool. All
receiving source stations are excluded from their own donor library. The one
station missing all ecological fields is handled explicitly. Residual fitting
uses station-balanced smooth absolute error with coefficient shrinkage, without
additional tail weighting.

An initial smoke exposed floating-point line-search termination at gradient
6.26e-9. The optimizer was repaired before production by extending the line
search and recording a deterministic retry, with failures still reported.
The original incomplete smoke is retained separately.

All nine packages and 36 memory states reproduce exactly from source and
validation data. Full-grid components and all 360 model-by-K query groups are
bitwise identical after reload. Recorded fit/product time totals **14.5 s**,
excluding earlier expert training, analysis and replay. Full suite: **617 passed,
2 skipped**; Ruff passes and the historical artifact audit exits zero.

```bash
uv run python scripts/run_ladder.py --experiment doc-ecological-transfer-v1
uv run python scripts/analyze_doc_ecological_transfer_v1.py
uv run python scripts/verify_doc_ecological_transfer_v1.py
```

[Detailed results](analysis/findings.md) · [Replay](verification/replay_checks.json)
