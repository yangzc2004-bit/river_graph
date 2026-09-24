# Invalidated exploratory output

This directory was generated on 2026-09-24 before the Stage-2 isolation and cluster rules were fully implemented. It is retained as an audit trail and is not evidence for the transfer paper.

Reasons for invalidation:

- the climatology used target-analyte labels outside the target basin, so it was a privileged same-analyte cross-basin reference rather than a held-out analyte baseline;
- task rows used all stations in each HUC6 rather than the frozen largest connected component;
- bootstrap rows treated repeated task seeds as independent clusters;
- Q90 thresholds were computed from query truth;
- provenance sidecars and complete Stage-2 controls were absent.

The old gate result must not be read as a scientific result. A corrected Stage-2A diagnostic must use a new versioned output directory and protocol.
