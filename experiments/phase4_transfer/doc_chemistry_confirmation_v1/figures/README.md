# Fresh station-assignment DOC confirmation figure

## Figure caption

**DOC reconstruction using hydrological history, auxiliary chemistry and sparse station calibration.** **a,** Native-scale mean absolute error (MAE) over the fixed held-station query observations, as the retrospective DOC support set increases from K = 0 to 5. **b,** Paired MAE difference between the chemistry-aware neural procedure and the retained general model at K = 0, 3 and 5; negative values favor the chemistry-aware model. Filled squares and horizontal bars show the aggregate difference and its paired 95% percentile interval from 5,000 joint whole-station bootstrap draws. Open markers show individual partition estimates, each averaged over three training seeds. They describe partition variation, not uncertainty intervals. **c,** MAE conditional on observed DOC being at or above the source-training 90th percentile. Tail thresholds are defined separately by source partition. Panels a and c show seed-averaged metrics within each partition and equal weights across the three partitions; these curves carry no confidence bands. Their y axes focus on the displayed range, with 13% range padding, rather than starting at zero; tick labels give the absolute error values. Panel b retains its zero-difference reference.

The three partitions use new station-role assignments on the existing ST357 Mississippi cohort. They are not external-basin validation or new independent measurements. K counts retrospective support observations per held station. All five reserved support candidates are excluded from the query set at every K, so its population remains fixed. Training seeds do not increase ecological sample size.

## Curves and information scope

| Legend | Frozen analysis model |
|---|---|
| Chemistry-aware neural | `neural_chemistry_integrated_selected` |
| General model | `point_integrated_legacy` |
| Chemistry-aware trees | `tree_chemistry_selected` |

The chemistry-aware neural and tree procedures select their support representation using source validation only, among legacy temporal, availability-augmented and chemistry-augmented coordinates. Each uses the same held-station support/query identities. Auxiliary pH and specific conductance are observed calendar-month means, not necessarily measurements taken at the same sampling instant. When both are absent, predictions retain the corresponding general neural/tree fallback after adaptation.

## Evaluation denominators

| Partition | Partition seed | Held stations | Fixed query cells | Auxiliary observed | Q90 cells |
|---|---:|---:|---:|---:|---:|
| 1 | 242 | 71 | 4,163 | 4,043/4,163 (97.1%) | 288 |
| 2 | 243 | 71 | 3,064 | 3,062/3,064 (99.9%) | 441 |
| 3 | 244 | 71 | 4,324 | 4,286/4,324 (99.1%) | 415 |

The evaluation contains **9,705 unique station-months** and **11,551 partition-cell occurrences**. Of the unique query cells, 9,547/9,705 (98.4%) have at least one auxiliary chemistry observation. The occurrence count retains cells that appear in multiple partitions; the bootstrap samples such stations jointly. It does not treat seed repeats as new observations.

| Unique query availability | Station-months | Stations (may overlap groups) | Partition-cell occurrences |
|---|---:|---:|---:|
| both | 9,415 | 174 | 11,253 |
| ph only | 63 | 3 | 64 |
| ec only | 69 | 21 | 74 |
| neither | 158 | 5 | 160 |

## Availability beyond observed DOC queries

This table describes covariate coverage, not predictive accuracy at missing DOC cells.

| Full-grid population | Auxiliary observed | Population denominator | Fraction |
|---|---:|---:|---:|
| all station months | 63,824 | 233,478 | 27.3% |
| doc observed | 22,201 | 22,571 | 98.4% |
| doc genuinely missing | 41,623 | 210,907 | 19.7% |

## Files and reproduction

The PDF and SVG are vector figures; the PNG is 300 dpi. White backgrounds, DejaVu Sans typography and the manuscript's blue/orange/grey palette are used. Shape and line style provide redundant model encodings. Panels a and c use focused, explicitly labeled absolute-error axes; panel b has a visible zero-difference reference.

Run from the repository root after the complete analysis:

```bash
uv run python scripts/plot_doc_chemistry_confirmation_v1.py
```

The source manifest records every input CSV, the complete-analysis manifest and the plotting script. All numbers are read from those analysis products. This command does not fit models, read target prediction parquet files, recalculate intervals or choose models from plotted outcomes.
