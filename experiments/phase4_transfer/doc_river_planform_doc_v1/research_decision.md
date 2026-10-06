# River planform and DOC: research decision

## Decision
Proceed with **continuous planform-conditioned DOC modelling**. Retain the three
real form groups for communication and stratification, and encode measured
elongation, spatial axis ratio, drainage density, mainstem share and mainstem
sinuosity as continuous information. A type label alone discards useful variation.
The historical uniform river-message residual realizes almost none of that
increment. Next distinguish a morphology-informed local representation from a
morphology-conditioned upstream operator.

## Population and analysis
The fixed geometry typology supplies 353 complete station networks, with sparse
networks and acquisition failures separately retained. The deduplicated union of
142/143/144 source-training roles contains 21,459 observed DOC cells. Among
classified networks, 312 source stations contribute records; 305 satisfy the
existing response criteria (at least 12 observations, six calendar months).
Their three group sizes are **106 / 31 / 168**, across **63 HUC4s**.

This is an exploratory source-role study. A station can be source train in one
split and validation in another; the union is not independent confirmation.
No geographical confirmation or external-basin test results were read, and no
neural model was retrained. Station records, not seeds or repeated months, are
the ecological observation units. Regional intervals additionally resample
HUC4; nested basins can extend beyond those blocks. The geometry grouping is
held fixed rather than re-estimated in the DOC resamples.

## DOC dynamics by actual river form

| Real form | Eligible stations | Median of station median DOC, mg/L | Median annual log1p DOC amplitude | Median observed high-DOC frequency |
|---|---:|---:|---:|---:|
| Elongated, tributary-rich | 106 | 4.195 | 0.306 | 3.68% |
| Mainstem-dominated, sparse | 31 | 3.650 | 0.338 | 0.00% |
| Broad, tributary-rich | 168 | 4.200 | 0.261 | 2.63% |

The amplitude is harmonic peak-to-trough change, not a native concentration
percentage. High DOC is a fixed source Q90 of **10.0 mg/L**; frequencies describe
observed samples. A zero median frequency does not mean a whole class never has
high DOC. Broad networks have a lower descriptive seasonal amplitude, while
the concentration distributions overlap substantially. Calendar curves average
station monthly medians and their station populations vary by calendar month.

After measured polygon area, ecology, climate, local hydro, geography and sampling
adjustment, the class contrasts for concentration, amplitude, variation,
high-value frequency and concentration-flow response all have HUC4-block
intervals spanning zero. Keep the unadjusted patterns as descriptions rather than
attributing them entirely to river form.

Continuous mainstem sinuosity has a positive adjusted association with log1p
station-median DOC: **0.079 per station SD**, regional interval **[0.028, 0.115]**.
Its high-value-frequency coefficient is **0.050** (fraction units, about five
percentage points per SD), interval **[0.014, 0.074]**. These are exploratory
conditional associations among several examined outcomes and correlated shape
features. They motivate a longitudinal-path hypothesis; sinuosity is not a
measurement of residence time, processing or carbon retention. All coefficients
and both station/HUC4 intervals are saved, including those spanning zero.

## Additional information beyond environment and basin area
Five HUC4-blocked folds predict **station median DOC** with fixed Ridge alpha=10.
Each fold fits its own preprocessing, imputation and model on training regions.
The baseline already includes log measured polygon area, ecology, climate,
local hydro, coordinates, HUC2 and sampling covariates. Fold MAEs are equally
weighted. Bootstrap samples preserve station identity; region-block resampling
is reported as a sensitivity.

| Inputs | Native MAE, mg/L | Native improvement | HUC4-block 95% interval | log1p improvement | Positive folds (native / log1p) |
|---|---:|---:|---|---:|---|
| Environment + area | 1.915 | reference | — | reference | — |
| + Three form labels | 1.880 | 1.81% | [-0.12%, 4.44%] | 1.48% | 3/5 / 3/5 |
| + Continuous form measurements | 1.775 | **7.31%** | **[2.82%, 12.56%]** | **7.25%** | 4/5 / 5/5 |

The continuous log1p improvement has regional interval **[2.67%, 11.19%]**.
Station-resampled intervals give the same positive direction for continuous
shape. This is evidence of joint shape information in this source diagnostic,
not a 7.3% improvement of the released monthly DOC model. It also does not assign
the gain uniquely to any single shape feature or isolate a physical effect.
Sampling covariates and hydro summaries here use the source DOC observation
calendar; this diagnostic is not the no-water-quality-input K0 deployment task.
The next reconstruction experiment must construct all inputs from its permitted
prediction-time information instead of carrying those descriptive covariates.

## Do existing graph messages realize that information?
Reloaded matched historical KGML validation predictions compare upstream-only
or both-direction graph residual with the same no-message residual model.
Models have identical query cells and three seeds; seed losses are averaged
before paired station resampling. Shape classification does not select runs.

- Strict temporal extrapolation: upstream gain **-0.0008%**, interval
  **[-0.227%, 0.207%]**. No class has an established gain; same-month upstream
  observed support is zero in these validation cells. Historical upstream
  context and environmental states may still be used by the model.
- Spatial station holdout: upstream gain **0.0083%** and both-direction gain
  **0.0349%**. These can have intervals above zero, but their effect sizes are
  negligible for practical reconstruction.
- The broad class shows the clearest positive spatial difference:
  upstream **0.0157%**, both directions **0.0583%**. Do not interpret this as
  a material performance improvement or a benefit of the current release.
- Support-stratified results retain every subgroup. The sparse class has only
  two supported receivers and ten query cells, so it cannot sustain a strong
  support-specific comparison. Cells without current visible upstream DOC may
  still receive environmental, historical or downstream information.

Same-month upstream/downstream source seasonal-anomaly correlations have median
**0.313 / 0.265 / 0.335** by receiving form, based on **38 / 5 / 89** connected
edges (22 / 4 / 46 receivers). All lag buckets retain this fixed edge population.
One-month medians are **0.102 / 0.104 / 0.166**, and 12-month medians are negative.
Shared forcing, endpoints and monitoring distances limit interpretation; monthly
lag correlations are not measured transit times. The sparse class is poorly
represented among connected monitored pairs.

## Next model experiment
Use the current model as the common backbone and change one information source
at a time, on source train/validation roles:

1. **Morphology-informed local state:** append the five continuous form features
   and measured polygon area to the ecological representation. Compare with the
   unchanged full model and a tree baseline receiving the same added features.
2. **Morphology-conditioned upstream information:** condition existing real-edge
   residual weights on receiver form, branch organization, path distance,
   current hydro and visible upstream support. Retain a matched no-message arm
   with all the same local shape features. This isolates upstream information
   from the improvement of the local representation.
3. Analyze whether the difference concentrates in broad branch-rich networks,
   long longitudinal paths or well-supported receivers. Use continuous curves
   alongside the three summary groups; do not fit a separate model per class.
4. Only after source-role benefit is established, perform a new fixed geographic
   confirmation. Existing geographic/external versions remain historical results.

The scientific question is now sharper: **river geometry contains DOC information;
which representation and observation conditions allow a model to use it?**

## Reproduction
```bash
uv run python scripts/analyze_doc_river_planform_doc_v1.py --bootstrap-draws 5000
uv run python scripts/verify_doc_river_planform_doc_v1.py
uv run python scripts/plot_doc_river_planform_doc_v1.py
uv run python scripts/plot_doc_river_planform_doc_v1.py --chinese
```

Four figure families are saved in English and Chinese. Checks cover source
visibility, region isolation, identical paired populations and metric
recalculation. Historical files and model products are unchanged.
The full suite passed **1,058 tests, 2 skipped**; Ruff passed; the historical
artifact audit exited 0. The response and paired-error verifier passed all
54 source identities and reconstructed the stored point estimates.
