# Common-query DOC comparison with SSN and PhyST-HC architecture adaptations

Date: 2026-10-11. Written before the new fits and target scoring.

The experiment asks whether the current environmental encoder + GRU + graph Transformer remains competitive against environmental regression, hydrological GAM, stream-network covariance and a published temporal Transformer–GCN architecture. It does not assume a win, prove universal information-fusion superiority, or reproduce native SWAT-C forcing.

## Common task

Reuse the frozen ST357 Mississippi cohort and whole-station HUC4 development masks: 1013, 1019, 0708, 1030, 1101; seeds 42/43/44; zero target DOC support. Every method predicts the same 7,262 observed station-month queries. Source stations, validation stations and target stations are disjoint. These tasks were inspected in earlier work, so this is development evidence, not fresh external confirmation.

DOC supervision is available only at source training cells and source validation cells used for selection. Features do not include DOC history/context, pH, conductance or station identity embeddings. The task is retrospective monthly concentration reconstruction: contemporaneous temperature/discharge are permitted. Environmental and hydrological normalization is fitted on source training stations. Network and location geometry are known without DOC labels.

Native concentration MAE and log1p MAE remain joint primary endpoints; RMSE, predictive R², bias, station/basin macro errors and coverage are secondary. No existing paper endpoint is changed. Report paired exploratory five-basin uncertainty and regional variation. Input sets differ transparently, so the main table compares whole common-data recipes rather than isolating architecture alone.

## Methods and budgets

Reuse hashed, audited predictions for the current fusion model, contemporary environmental RF, hydrological GAM, full-history RF and XGBoost. Those use the same dataset, labels and query masks; do not reselect them after seeing new results. RF/GAM are literature-motivated adaptations, not original-input reproductions.

Fit SSN as a linear mean on E15 + current T8, with spatial Gaussian station effects and independent measurement residuals. Exponential tail-up, tail-down and Euclidean kernels follow the public USEPA SSN2 formulas. Compute mainstem distances and first downstream meeting points from cached NHDPlus VAA and station COMID/linear measures. Tail-up weights use drainage-area branching fractions; when a junction has missing or nonpositive area, use equal branch shares and report the count. The Euclidean term uses positive-definite Earth-centered chord distance. Full physical mainstem geometry is additional declared SSN information. Minor diversion pathways are outside this geometry.

Fit three SSN recipes: log1p network kernels, log1p network+Euclidean kernels, native network+Euclidean kernels. Covariance parameters use exact grouped REML over all training records, not a DOC subsample; two source-likelihood starts and maximum 100 optimizer iterations. Station grouping is algebraically exact and tested against dense GLS. SSN fits are deterministic: fit each recipe once per region and reuse predictions across nominal seed rows without treating those rows as independent replicates.

For PhyST-HC, load only the reviewed, hash-frozen `PositionalEncoding` and `ImprovedTransformerGCN` definitions from the author's repository. Preserve their forward architecture. Repeat E15 alongside each T8 time step, adding one constant zero column for the original two attention heads: 24 channels × 12 months. Disjoint monthly graph panels prevent edges across months. Train on the source-induced station graph; predict on source+validation or source+target covariate graphs, matching the fusion model's information regime.

Two compact PhyST architecture recipes use (feed-forward width, GCN width) (96,24) and (128,48), both with two temporal layers, two heads and 0.1 dropout. Use the current reference's source-standardized log1p-MSE training, AdamW, learning rate 0.001, weight decay 0.0001, 32-month batches, maximum 60 epochs, minimum 15 epochs and patience 10 selected by source-validation native MAE. The common concentration response, covariates, time window, loss and widths differ from the complete native event-scale SWAT-C/flux workflow. Label this an architecture adaptation throughout.

There are 15 deterministic SSN fits and 30 seeded neural fits: 45 new fits. Family budgets differ and do not establish optimized architecture-family rankings. Freeze all regional selections before any new target scoring. Existing results remain dependencies rather than new fits.

## Verification and provenance

Use a new versioned workspace and its `scripts/run_ladder.py` dispatcher. Keep original training scripts, historical freezes, source code and predictions unchanged. Each prediction receives dataset/mask/config/code hashes and a provenance sidecar. Record public source commits, geometry hashes, full preparation audit, training traces and selection receipts before evaluation.

Tests cover physical distance/position/branch-weight examples, positive covariance, exact grouped-vs-dense GLS, redundant mean features, neural batch isolation and frozen contracts. After training, replay all 45 candidate validation checkpoints and selected query predictions, independently recompute primary and secondary metrics against original DOC truth, and verify equal coverage. Re-run historical K-shot contract tests and the full repository suite. Preserve adverse results without target-driven retuning.

The controlled source-package ablations from earlier component work remain separate evidence; this literature comparison alone cannot attribute any performance difference to adding environmental, temporal or graph information.

## Execution revision before new target scoring

The serial v1 run was interrupted after a source-only runtime benchmark; it never generated new target predictions. Its frozen files and partial source fits are retained. A fresh v2 freezes identical model recipes, masks, epoch/selection rules and endpoints with five independent region workers and one CPU thread per worker. The timing benchmark used source covariates and no DOC labels; measured train+evaluation batch times were approximately 0.416, 0.422 and 0.454 seconds at 1/2/4 threads. This revision addresses runtime, not target performance. All 45 v2 fits restart under their own protocol and provenance; v1 partial fits are not mixed into final results.
