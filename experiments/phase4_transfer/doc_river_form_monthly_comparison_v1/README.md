# DOC properties of real river forms on shared calendar months

Read `study_plan.md` for the pre-fit comparison design and
`research_decision.md` for the scientific result and next morphology question.
This is a source-role exploratory extension of the earlier real-form study.

## Reproduce

From the repository root, with its uv-managed environment:

```bash
uv run python scripts/analyze_doc_river_form_monthly_comparison_v1.py --bootstrap-draws 5000
uv run python scripts/plot_doc_river_form_monthly_comparison_v1.py
uv run python scripts/plot_doc_river_form_monthly_comparison_v1.py --chinese
uv run python scripts/verify_doc_river_form_monthly_comparison_v1.py
```

Local prerequisites are the ST357 dataset, existing permitted source-cell
array, and the preceding fixed morphology panel and covariate-selected pair
list. Required paths and source hashes are in `analysis_sources.json`.
No model-training entry point or neural fitting is invoked.

## Main tables and products

- `station_month_inputs.parquet`: one permitted source-role observed
  station/month with fixed ecology/morphology and observed current hydro.
- `monthly_predictions.parquet`: six fixed information arms on identical
  HUC4-held observation rows, with a prediction provenance sidecar.
- `fitted_states.json`: all 40 source-fitted regression/probability states,
  including medians, scales, coefficients and held/training HUC4s.
- `paired_month_records.parquet`: same-calendar observations and source-fitted
  adjustment backgrounds for the unchanged eligible matched pairs.
- `pair_inclusion_ledger.csv`: all 29 original candidate pairs, including
  shared-month, measured-hydro and longer-record eligibility.
- `paired_response_evidence.csv`: both member values, equal-pair differences,
  common-month count and high-DOC numerator/denominator for every response.
- `paired_response_contrasts.csv`: all three class contrasts and all three
  populations, with paired HUC4 bootstrap intervals.
- `hydro_balance.csv` and `omitted_huc4_sensitivity.csv`: measured hydro
  comparability and influence of each region.
- `station_model_errors.csv`, `information_scores.csv`,
  `information_gains.csv`: station-equal geographic information comparisons.
- `figures/`: three scientific figures, PNG/PDF in English and Chinese,
  with data/generator manifests.

DOC threshold is fixed at 10 mg/L; SD is sample SD (ddof1). Primary response
contrasts average months within each matched river, then give pairs equal
weight. Information scores average months within each station, then give
stations equal weight. Bootstrap retains entire HUC4s; a one-region class
contrast has no geographic interval. These are observed monthly associations,
not physical buffering rates, event peaks, or released-model performance gains.
