# Integrating station adaptation into the DOC manuscript

Prepared 2026-10-03 against the running nine-fit study. These notes contain method text and an integration map, not experimental results. The scientific question is whether sparse station calibration extends the environmental–temporal hybrid beyond an equally calibrated environmental predictor.

## Ready-to-insert Methods text

### An integrated predictor with station adaptation

We integrate environmental prediction, temporal residual learning, and sparse local calibration in one fitted reconstruction model. The environmental expert, \(C_{it}\), predicts DOC from environmental covariates, explicit local lags, and visible river-context summaries. A second expert, \(R_{it}\), combines a local environmental prediction with an observation-aware recurrent correction. Its local reference is an ExtraTrees model with 300 trees and minimum leaf size four. Five station folds provide out-of-fold reference predictions for training the neural residual, with all DOC observations at the withheld training stations hidden from the corresponding input view. The residual expert retains the ecological encoder, 64-unit observation-aware GRU, and 12-month history of the temporal model. Its selected configuration uses a self-node encoder and explicit upstream-support features, without learned river-edge messages or attention. The recurrent model is trained for at most 20 epochs with early-stopping patience five.

For the new spatial experiment, source-validation MAE selects the environmental expert from four 300-tree ExtraTrees configurations: minimum leaf sizes four, two, or one with all features available at each split, and leaf size one with square-root feature selection. The selected environmental expert and the complete temporal expert are combined using the same validation stations. Candidate combinations comprise either expert alone, convex mixtures of their log-transformed predictions with a weight grid of 0.005, and an affine combination,

\[
 H_{it}=\max\left\{0,
 \exp\left[\beta_0+\beta_C\log(1+C_{it})+
                  \beta_R\log(1+R_{it})\right]-1\right\}.
\]

The affine coefficients are fitted by ordinary least squares to validation log-DOC, and raw-scale validation MAE selects the final candidate. The context-only candidate is retained and its selection reported. This allows the integrated model to retain environmental prediction where temporal residual learning supplies no useful validation improvement. The candidate selection is an output-level combination of complete experts; the neural branch itself remains trained relative to the separate local reference.

For a target station \(i\), let \(\mathcal S_i^{(K)}\) contain \(K\) support observations. Given either uncalibrated predictor \(P\in\{C,H\}\), station adaptation applies the mean log-space residual at the support months:

\[
 \bar r_{i,P}^{(K)}=\frac{1}{K}
 \sum_{s\in\mathcal S_i^{(K)}}
 \left[\log(1+y_{is})-\log(1+P_{is})\right],
\]
\[
 \widehat y_{it,P}^{(K)}=
 \max\left\{0,
 \exp\left[\log(1+P_{it})+\alpha_{P,K}\bar r_{i,P}^{(K)}\right]-1\right\}.
\]

The correction compares observations with model predictions at the same support dates, preserving the base prediction's temporal pattern in transformed space. For \(K=0\), the correction is zero and the original prediction is returned exactly. Shrinkage \(\alpha_{P,K}\) is selected separately for the environmental and hybrid arms at each support budget from \(\{0,0.25,0.5,0.75,1\}\), using the same source-validation station tasks and raw-scale MAE. The selected experts, combination, and shrinkage are fixed before target query evaluation. Target support observations enter only this explicit output correction; they neither enter the experts' features nor update the forest or neural weights. Predictions are bounded below by zero after inverse transformation, without upper clipping.

### New station-partition experiment

We evaluate the integrated model on three new random station partitions of the existing 357-station Mississippi cohort. Each partition contains 232 training, 54 validation, and 71 test stations. Partition seeds are 142–144; training seeds 42–44 provide three fits within each partition. Station allocation depends on observation availability and the random permutation, not observed DOC values. Stations require at least six observations to be eligible for validation or testing, allowing five potential support observations and at least one query. The complete station covariates and river graph remain available. These partitions test transfer to withheld stations within the development cohort; they do not introduce new observational data, a geographically contiguous holdout, or an independent river basin.

Both experts use only training-station DOC as input during validation and test prediction. Validation labels guide early stopping, environmental-model selection, fusion, and station-calibration selection, but remain hidden from expert input features. The same validation station set supports these selection steps; validation scores are consequently tuning scores, rather than a separate performance estimate. Current hydroclimatic covariates are assumed available. At each held-out station, five observed dates are reserved in the order first, middle, last, first quartile, and third quartile of its observed record. Nested prefixes define \(K=0,1,3,5\), and all five reserved observations are excluded from the query at every budget. Thus each partition uses identical query cells across support budgets, models, and training seeds. Support dates can occur after query dates, so the experiment assesses retrospective record reconstruction.

The four comparison arms are the environmental predictor, its calibrated version, the hybrid, and its calibrated version. The primary spatial comparison is calibrated hybrid versus calibrated environmental prediction at \(K=5\). The uncalibrated hybrid-versus-environmental comparison at \(K=0\), and each predictor's change from \(K=0\) to \(K=5\), distinguish expert complementarity from the benefit of local observations. MAE is accompanied by RMSE, \(R^2\), log-space MAE, and tail MAE above the training partition's DOC 90th percentile. We average the three training-seed metrics within each partition and then weight the three partitions equally. Paired uncertainty intervals use 5,000 bootstrap resamples of station identities. A station appearing in multiple test partitions receives the same resampling multiplicity in each, preserving this repeated-station dependence; within each partition, all query cells of a sampled station contribute to the cell-weighted error. These intervals summarize uncertainty in the evaluated fixed predictions rather than variation from refitting the entire model.

Station-level responses and descriptive strata summarize variation in benefit. Environmental novelty is the nearest training-station distance in training-standardized regime descriptors, normalized by the square root of the descriptor count. Upstream support is the fraction of directly connected upstream stations with visible DOC in the current month. These analyses characterize where the correction helps and where it does not, rather than identify physical transport effects.

## Changes needed in the current manuscript when results arrive

| Location in `latex/spatial_transfer_draft_v1.tex` | Integration action |
|---|---|
| Abstract, lines 22–24; Introduction contributions, lines 35–37 | Introduce station calibration as part of the evaluated integrated predictor. Replace the separate regional 17.9% result with the matched nine-fit finding if used in the main abstract. Retain the established temporal results as a distinct experiment. |
| River cohort and reconstruction tasks, lines 41–60 | Add the three random station partitions. Scope the existing statement that terminal inference opens validation DOC to the historical experiments only: the new experiment uses training DOC at both validation and test inference. |
| Context predictor, lines 63–65 | Keep the six-candidate historical selection description attached to the old temporal/scenario product. The new spatial study has four ExtraTrees candidates and no RF candidate in its selection set. |
| Residual training, line 100 | Retain the shared architecture and 20-epoch budget. The statement that completed runs stopped after 15–16 epochs describes the historical temporal runs only; summarize the nine new traces separately. |
| Expert combination and architecture figure, lines 102–115 | Retain the historical hard-coded context route for the original spatial experiment. Add the new validation-selected spatial fusion and support correction; the new study does not hard-code the spatial route to context. Update the schematic to show the output adapter and its separate support-label input. |
| Separate spatial-adaptation extension, lines 119–131 | Replace in the main Methods with the integrated predictor and new experiment above. Move the 120-tree, 40-neighbor regional product and its original 43-station query to a clearly identified historical/supplementary experiment if retained. The new model does not use regional neighbor selection. |
| Evaluation, lines 133–140 | Add seed-then-partition averaging and the joint station bootstrap. Replace “independent station partitions remain the next tests” with the completed new-partition experiment once available; external-basin assessment remains future work. |
| Spatial results, table, and figures, lines 195–223 | Replace the main spatial section with the four matched arms, K curves, calibrated-hybrid versus calibrated-tree effect and interval, partition consistency, and station heterogeneity. Do not reuse the old 2,316-query count, 17.9% gain, or shuffle diagnostic for the new study. Report new per-partition query counts. |
| Discussion, lines 236–238 | Replace “the current experiments establish the two pieces separately” with an assessment of the jointly evaluated prediction workflow. Its components are still fitted in stages; avoid “jointly trained end-to-end.” |
| Future research, lines 245–252 | Advance the third direction from integrating the existing components to adaptive updates of temporal states/weights and prospective support limited to earlier dates. Current support changes the output level only. Independent basins, contiguous regional withholding, higher-frequency events, and uncertainty-aware loads remain natural next studies. |
| Conclusion and code availability, lines 254–260 | Report the new matched spatial evidence and cite `analyze_unified_doc_spatial.py`. Separate historical temporal results from new station-partition results. |

## How to read the eventual results

- A calibrated-hybrid improvement over its own \(K=0\) output establishes the value of local calibration. The stronger comparison for neural complementarity is calibrated hybrid versus equally calibrated ExtraTrees on the same queries.
- Report how often validation selects context-only, temporal-only, a geometric mixture, or an affine mixture. If context-only is selected, that fit contributes no neural prediction effect; with the identical calibration search, its two calibrated arms should also coincide.
- Even a positive hybrid-versus-tree result evaluates the complete expert-combination procedure, including validation fusion. It does not by itself isolate each observation-age feature or GRU mechanism, establish river-message value, or separate neural information from every possible output-recalibration effect. The matched temporal component experiment remains the clearest existing direct residual-branch evidence.
- Three new partitions probe partition sensitivity on the same developed cohort. Training seeds and repeated station appearances do not constitute independent ecological samples. Report the number of unique test stations as well as partition counts.
- Use the nine confirmation fits for scientific reporting. The 20-tree, three-epoch smoke run is solely an execution and serialization check.

## Implementation anchors

- `experiments/phase4_transfer/unified_doc_spatial_v1/study_plan.md`: research question and nine-fit design.
- `src/river_graph/models/unified_doc.py`, `UnifiedDOCReconstructor._new_residual`, `.fit`, `.predict_components`, `.predict`: selected architecture, four context candidates, training-only expert visibility, integrated inference.
- `src/river_graph/models/kgml_local_transport.py`, `fit_rf_artifacts`, `LocalTransportKGML.fit`: five-fold residual targets, station-hidden views, scaling, recurrent optimization and checkpoint selection.
- `src/river_graph/models/station_adapted_hybrid.py`, `fit_fusion`, `fit_calibration`, `station_residual_correction`: exact fusion and calibration equations, selection grids and context fallback.
- `src/river_graph/experiments/unified_spatial_protocol.py`, `build_unified_spatial_split`, `support_query_cells`: value-blind station partitions, support eligibility and fixed queries.
- `scripts/run_unified_doc_spatial.py`, `covariate_diagnostics`, `export_products`: novelty/upstream definitions and exported component predictions.
- `scripts/analyze_unified_doc_spatial.py`, `joint_station_bootstrap`, `comparison_products`, `fallback_diagnostics`: paired effect summaries, partition averaging and fallback accounting.

The production runtime snapshot is saved under `experiments/phase4_transfer/unified_doc_spatial_v1/confirmation/`; final numerical claims should cite its completed prediction products and generated analysis, not these integration notes.
