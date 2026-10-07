# Verification record

This version completes mapped local-geometry analysis. Original public mixing
event files remain unavailable; it does not claim an observed mixing replay.

## Calculations and source alignment

- Replayed all 295 directed corridors from the saved routing arrays and actual
  cached flowline WKB. Recomputed 885 scale-specific rows, checked the unchanged
  whole-network labels against the frozen 297-network morphology/DOC panel, and
  checked each used routing/geometry content against the saved receipt.
- Independently calculated the 250 m line lengths, incoming chord angles and
  downstream sinuosity from the exported actual vertices, without using the
  analysis angle helper. The resulting values agree with the saved tables.
- Recomputed the original 22 elongated/broad comparisons and their 5,000 HUC4-
  group resamples. The seven other class-pair rows in the source pair table are
  explicitly excluded from this contrast. Class-pair validation is tested.
- Recorded physical junction reuse: 295 network instances, 251 different
  junctions, 29 reused junctions and five shared between forms. No 500 m route
  is extended where the common downstream corridor is shorter.
- All 250 m mapped junction endpoint gaps are zero. The 100/250/500 m differences
  remain in the scale-sensitivity results.

## Necessary repairs made during construction

Three implementation issues were found before delivery and corrected:

1. CSV integer-versus-float class dtypes were not a changed class label. The
   numeric label check now compares all selected stations against the original
   complete panel, rather than an expanded map catalogue missing three aliases.
2. Centreline length roundoff could append an effectively zero final sampling
   interval at an exact 25 m multiple. The turning sampler now avoids that
   duplicate endpoint; an almost-exact-length regression fixture covers it.
3. The first comparison construction included all 29 source pairs. It now uses
   exactly the 22 original elongated/broad pairs, matching the stated question
   and the preceding experiments. No results from the incorrect pair set are
   retained as final findings.

The saved replay keeps physical junction identifiers as strings consistently.
These repairs change construction/reading logic rather than selecting favourable
outcomes; the final tables and source snapshot were regenerated after them.

## Code, figures and historical checks

- New analytic tests: **8 passed**. The related junction/corridor/storage subset:
  **33 passed**. Known angles, reversed cached coordinate order, short routes,
  local versus distant map gaps, exact turning and fixed pair membership are
  covered.
- Full repository suite: **1,267 passed, 2 skipped**. Existing numerical and
  dependency warnings remain; there are no test failures.
- `ruff check .`: **all checks passed**.
- Full geometry replay with `--full-replay --skip-figures`: **passed**, including
  all 295 corridors and the paired numerical summaries.
- Final default replay: **passed**, including actual representative corridors
  and both languages' figure input/output identities.
- Actually viewed all four final PNGs: two figures in English and Chinese.
  Fixed an overlapping scale-bar annotation and crowded class labels, then
  regenerated and inspected the final images. The displayed numbers, geographic
  orientation, scale bars and class/pair denominators match their source tables.
- Historical `audit_artifacts.py --verify`: **exit 0**. It reports 83 verified
  parquet-only artifacts, one zero-coverage artifact and the existing excluded
  G0 conflict. The 85 historical predictions have no sidecars and the absent
  old datasets prevent complete mask-based verification; this check does not
  upgrade their historical evidence status.

No model was trained, and no prior result directory or frozen endpoint file was
overwritten. The unrelated pre-existing working-tree edits were left intact.
