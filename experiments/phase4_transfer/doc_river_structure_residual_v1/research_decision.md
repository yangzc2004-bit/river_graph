# River structure residual research decision

The structure-aware river extension is implemented and all 36 neural fits are complete: three source-role station partitions, three seeds and four learning arms. Receiving stations have no DOC, pH or conductance inputs. The nine prediction packages replay exactly, including their donor banks, integrated predictions and fitted zero-message diagnostics.

## Scientific result

Real upstream information has its clearest predictive benefit along short source paths. Joint fitting also changes the existing local and similarity paths, offsetting the message benefit at stations without usable upstream observations. The next model should preserve the strong local prediction while learning selective river corrections.

| Model | Equal-partition K0 MAE mg/L | Q90 MAE mg/L |
|---|---:|---:|
| Complete current reference | 1.723002 | 8.895024 |
| Matched current refit | 1.723029 | 8.894860 |
| Simple upstream messages | 1.723066 | 8.883750 |
| Structure-conditioned river messages | 1.724014 | 8.879511 |
| Source-rewired structure control | 1.722055 | 8.903378 |
| Strong matched environmental trees | 1.866362 | 9.394458 |

The structure model has 0.0587% higher overall error than the fixed complete reference, with a 95% interval for MAE reduction of [-0.5476%,0.4052%]. Its Q90 error is 0.1744% lower, with an interval crossing zero. The complete current release remains the deployed predictor.

## Where the additional information appears

| Source condition or structure | MAE reduction versus matched current refit | 95% paired station interval | Unique stations |
|---|---:|---:|---:|
| Nearest eligible upstream path at most 50 km | 2.649% | [0.617%, 4.306%] | 23 |
| Nearest path 50–200 km | 0.076% | [-0.703%, 0.642%] | 30 |
| Nearest path over 200 km | -0.660% | [-3.463%, 3.004%] | 6 |
| Integrated mainstem | 1.193% | [0.245%, 2.560%] | 28 |
| Major confluence vicinity | 0.357% | [-0.124%, 0.937%] | 64 |
| Lake/reservoir path | -0.023% | [-0.715%, 0.684%] | 73 |
| No observed upstream source | -0.496% | [-0.966%, -0.045%] | 122 |

The at-most 50 km group also improves 2.696% over rewired sources [0.177%,4.399%]. Rewiring preserves receiving path slots and matches source drainage area; real banks have 32.32% supported query cells versus 28.67% for rewired banks. This comparison combines physical pairing with differences in observation support. Rewired donors are outside the eligible upstream graph within the 3000 km search limit; upstream connectivity beyond that search limit is not excluded.

An additional common-support diagnostic was defined after the first full readout, using only predictor availability. In nearby paths where both banks have an observed source, the structure model improves 4.848% versus the matched refit [1.052%,7.210%], based on 14 unique stations and 570 unique station-months. Its 4.274% improvement over rewiring has a wider interval [-0.891%,7.885%]. This is an exploratory source-development signal for nearby observed upstream information. Connectivity alone and the extra path-conditioning transform are not established as the cause of the improvement. The simple directed model remains competitive in this subset.

Structure profiles overlap. Observed-support and unsupported-month groups can include the same station. Seeds repeat the same ecological observations. These results use ST357 source training/validation roles, not geographical confirmation or independent external-basin validation.

## What the message contributes within the fitted model

Zeroing only the fitted structure message gives MAE 1.728806 mg/L. Restoring it reduces MAE to 1.724014 mg/L: a 0.2772% improvement [−0.0375%,0.5768%], positive in all 9 split-seed packages. This comparison measures the direct prediction contribution in the same fitted network. It does not compare two independently trained models.

Relative to the fixed complete reference, refitted nonriver paths and integration raise point-estimate MAE by 0.3368%; the direct river correction removes 0.2781%, leaving 0.0587% higher error. This is an arithmetic error decomposition, not a causal attribution. Stations without observed upstream sources degrade 0.4956% [0.0450%,0.9656%]; their direct river correction is exactly zero. The degradation therefore occurs through jointly refitted nonriver paths/integration.

## Architecture and training refinement

The existing ecological self encoder, observation-aware GRU and ecological-source attention remain. A sparse directed operator gathers observed upstream departures over mapped paths and lags 0/1/3, with explicitly aged retained observations. Continuous length, major-junction, drainage-area and lake/reservoir attributes condition conveyance, mixing and storage-memory readouts. The added projection starts at zero. Actual river messages now exist in this experimental model; the fixed released model continues to use similarity retrieval.

The atlas proposal requested complete-model OOF targets for a frozen post-hoc river branch. Those complete-model OOF products were unavailable. Before fitting, this version explicitly changed to joint neural training with environmental tree OOF bases and double-held donor references. Tree OOF predictions are not described as complete-model OOF predictions. A matched no-river training arm controls the new training budget and initialization.

The no-river refit uses the same initial weights, data and budget. The float32 refit has small aggregate differences from the historical prediction: the largest package MAE difference from the fixed current reference is 0.000241 mg/L; one package has larger individual prediction changes. These differences are saved in refit_reproducibility.csv. Consequently fitted refit predictions are not claimed to be bitwise copies of historical fits. Saved prediction replay within each new fit is bitwise exact.

## Next iteration

1. Preserve current local and similarity predictions, concentrating new learning on the river correction. Compare a warm-start, tightly anchored joint variant with the existing cold joint fit. A frozen complete-base residual variant requires complete-model cross-fitting, including its DOC-supervised initialization, before describing its targets as OOF.
2. Give source age and short path distance an explicit role in information reliability. Start with the existing small 0/1/3-month operator; add conditional updates and support-matched controls before increasing model depth or the number of lag parameters.
3. Include a protected no-source prediction path that reproduces the current reference exactly when no eligible upstream observation is available. This targets the observed 0.5% degradation without changing the task or dropping unsupported stations.
4. Confirm the nearby-source pattern across source-role episodes with availability matched between physical and rewired pairs. Continue to separate simple upstream information from added structure conditioning. Mainstem and confluence groups remain useful diagnostics.
5. Develop the next version on the same source roles. A later fixed candidate can receive new geographical/external confirmation; completed historical tests and manuscript results stay results of their existing versions.

## Reproducible products

- Nine completed source packages, 36 fitted checkpoints, prediction parquet/sidecars, donor roles and real/rewired candidate arrays.
- analysis/: paired overall/tail effects, station-equal error, physical profiles, fixed distance bands, common-support diagnostics, source availability and refit reproducibility.
- figures/: inspected comparison and distance/error-component figures, each in PNG/PDF/SVG.
- verification/replay.json: nine successful source-bank, prediction and diagnostic replays.
- pytest: 1039 passed, 2 skipped. Ruff and the historical artifact audit pass.

Large fitted checkpoints and candidate caches remain local. Relevant code and small reproducible results are recorded in the scoped pending submission list because the Git directory is read-only. No recurring automation was created or restarted.
