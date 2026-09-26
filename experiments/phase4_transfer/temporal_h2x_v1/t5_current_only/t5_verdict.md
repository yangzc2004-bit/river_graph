# T5 current-only diagnostic verdict

## Scope

T5 keeps the H2X-T GRU wrapper, spatial encoder, target transforms, masks,
three seeds (42--44), and the matched 10-epoch training budget, but sets
`lookback=1`. It contains 36 configurations: three analytes, four missingness
families, and three seeds. This is a mechanism diagnostic, not a new primary
endpoint.

## Integrity

- 36/36 products passed the V3 audit.
- Dataset, target-mask, query-cell, and training-protocol identities match the
  corresponding H2X-T runs.
- No old Phase 0--3 artifact was overwritten.

## Matched result

The value reported below is the percentage change in MAE for current-only
relative to full H2X-T (positive means current-only is worse):

| analyte | E2a/E2b mean change | E1 change |
|---|---:|---:|
| DOC | 0.54% | 5.28% |
| pH | 20.85% | 13.56% |
| specific conductance | 13.79% | 8.63% |

Current-only was worse in the family mean for all 12 analyte-by-mask
comparisons. The largest differences are for pH and specific conductance,
while DOC is nearly unchanged in E2.

## Interpretation

The multi-month input window is useful, especially for pH and specific
conductance. The T4 shuffle arm showed that reversing the preceding history
had little effect at this budget, so the current evidence supports a simpler
reading: H2X-T benefits from having a longer history representation, but a
strict chronological-order or target-analyte-memory contribution has not been
isolated. The hydro-only arm also remained close to full H2X-T, suggesting that
much of the useful history signal can be carried by hydro/ecology/seasonal
inputs.

The recommended model for subsequent analysis remains H2X-T with `lookback=12`.
No larger history architecture sweep is justified until the existing temporal
products are analyzed by channel and analyte.
