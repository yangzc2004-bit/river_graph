# Verification record

- New scientific tests: 7 passed. Discharge unit conversion preserves slopes and
  predictions; perturbing later DOC preserves all chronological predictions;
  previous flow aligns to the preceding calendar month, not preceding DOC row;
  excluded DOC remains unread; collinear flow is unidentified; paired bootstrap
  operates at station/region grain.
- Full repository pytest: **1,428 passed, 2 skipped**, 49.44 seconds. Eight
  pre-existing warnings (torch deprecation, interval overflow fixtures, empty
  edge standard deviation and Shapely envelope fixtures) remain.
- Ruff full repository: passed after fixing the generated notebook import order
  in its generator and rebuilding/re-executing the notebook.
- Study verification: 21,459 source cells checked; all 18 temporal and 36
  descriptor gain point estimates independently recomputed; identical query
  cells across arms and strictly earlier fitting years verified. One station's
  response coefficients checked using direct full-design least squares.
- Historical `audit_artifacts.py --verify`: exit 0. The known overwritten G0
  conflict is excluded; one kriging zero-coverage case, 85 historical sidecars
  absent and missing old datasets remain reported. This run does not repair or
  upgrade the verification status of those historical artifacts.
- Both scientific figures viewed as actual images. Comparison legend placement
  corrected so it no longer covers the bottom-row estimates. The notebook is
  executed top-to-bottom and its exact output preview is retained locally for QA.

No existing training configuration, model source, mask or historical frozen
experiment was edited. Only this new source-only research analysis was run.
