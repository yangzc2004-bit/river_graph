# Separate chemical calibration: development decision

Date: 2026-10-04. All nine calibration packages completed and replayed.

## Result

The separate two-coordinate chemical increment improves conditional
station-validation MAE relative to the frozen legacy chemistry calibration:

| Support K | Legacy MAE | Separate increment MAE | Relative gain | Improved partitions / packages |
|---|---:|---:|---:|---:|
| 3 | 1.943208 | 1.930864 | 0.635% | 2/3; 8/9 |
| 5 | 1.881774 | 1.858551 | 1.234% | 3/3; 9/9 |

Q90 error decreases by 1.209% at K3 and 1.452% at K5, with positive directions
in all three partitions. Chemical increments outperform their matched
availability-only control by 0.635% and 1.296%. Every fitted chemical procedure
and all 45 held-fold choices retain a nonzero increment. K0/K1 and absent
chemistry keep the legacy prediction exactly.

This is useful evidence for the chemical correction itself. The previous joint
procedure remains stronger on these validation curves: the separate increment
has 0.859% higher MAE at K3 and 0.309% higher MAE at K5. It has not established
an improvement over the complete current model. Full-validation fitted scores
look stronger than held-fold scores and remain tuning evidence.

## Decision and next experiment

Proceed to a fixed-recipe comparison under fresh station-role assignments
**342/343/344**, using three training seeds and complete refitting of every
source model and preprocessing stage. The previous target results motivated
the intervention; they are not used to choose among its new fitted variants.

The scientific test is whether preserving legacy temporal calibration and
ecological mixing while separately shrinking chemistry improves K5 stability
on held target stations, compared with jointly fitting the expanded support
representation. Retain joint, legacy, general, tree and availability-only
comparators. No population chemical prior or kernel extension is added to this
confirmation. Its protocol is recorded separately in
`../doc_nested_confirmation_v1/study_plan.md`.

## Scope and implementation

Only source-validation DOC is extracted in this development run. Target
query predictions and labels from the parent are not opened. The parent itself
used validation for checkpoint and calibration selection, so its station-fold
diagnostics are conditional adapter validation, not a fully OOF evaluation of
the complete predictor. Repeated seeds do not increase ecological sample size.
Reporting weights query cells within each run and partitions equally, while
selection deliberately weights K3/K5 and stations equally.

Nine saved-state replays reconstruct the full-validation and held-fold
chemical predictions exactly. The new operator's focused tests cover direct
ridge calculation, hidden query labels, missing-input and zero-strength
fallback, shared K selection, held-station selection and serialization.

An initial implementation check stopped before completion because a timestamp
required by the existing sidecar writer was omitted. Its files and source are
preserved in `_implementation_check_20261004/`; the corrected complete
calibration run has its own execution snapshot. No old study was changed.

Reproduction:

```bash
uv run python scripts/verify_doc_nested_chemistry_v1.py
uv run python scripts/analyze_doc_nested_chemistry_v1.py
```

See `analysis/findings.md` and adjacent tables for complete conditional-CV,
tuning, tail, correction and station-response results.
