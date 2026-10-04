# Fresh nested chemistry confirmation figure

**Separating chemical and temporal station calibration.** a, Native DOC MAE across all fixed queries
as retrospective support increases from K0 to K5. b, Separate chemical increment minus joint-selected
calibration MAE at K3/K5; negative differences favor the separate increment. Filled squares show the
equal-partition estimate; bars show 95% percentile intervals from 5,000 joint whole-station bootstrap
draws. Open symbols show partition means over three training seeds, not additional confidence intervals.
c, MAE for DOC at or above each source-training Q90 threshold. Curves average seeds within partitions,
then weight partitions equally. All six predeclared procedures appear in a fixed semantic order.

Panels a/c use focused absolute-error axes with 13% padding; panel b retains zero. K0/K1 chemical
increments are exactly zero by construction. These are fresh roles on the same ST357 cohort, not
an independent external basin. Auxiliary pH/conductance are allowed covariates; support may postdate
queries. Five reserved support candidates are excluded from every query set. Repeated seeds do not
increase ecological sample size. Tail groups below 20 cells are marked unstable in analysis tables.

The figure does not select a model or splice procedures by K. See the complete findings and availability
tables for ordinary-DOC errors, bias, classification, and the difference between covariate availability
on observed test queries and at genuinely missing DOC cells.

| Legend | Fixed model |
|---|---|
| Separate chemical increment | `neural_chemistry_integrated_nested` |
| Joint-selected calibration | `neural_chemistry_integrated_selected` |
| Legacy calibration | `neural_chemistry_integrated_legacy` |
| Availability-only increment | `neural_chemistry_integrated_nested_masks` |
| General model | `point_integrated_legacy` |
| Chemical trees | `tree_chemistry_selected` |

PDF and SVG are vector outputs; PNG is 300 dpi. Input and output hashes are recorded in
sources_manifest.json. Reproduce with:

```bash
uv run python scripts/plot_doc_nested_confirmation_v1.py
```
