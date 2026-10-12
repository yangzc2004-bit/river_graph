# River DOC literature and comparative experiment

River DOC concentration has dedicated statistical, machine learning, satellite
retrieval and process models. The numerical comparison below selects four
representative model families that can use the project's existing permitted
covariates. It evaluates the fixed environmental encoder, hydrological GRU and
graph Transformer against alternatives motivated by DOC research. The numerical
arms are algorithm or model-family adaptations, not complete paper reproductions.

## Representative DOC studies

| Study | DOC task and method | Relevance and numerical comparison |
| --- | --- | --- |
| [Tian et al. 2025, Aqua-OC](https://authors.library.caltech.edu/records/b74cg-58e81) | Reach-scale satellite retrieval. Four optical water types select among RF, XGBoost, SVR and DNN, using Landsat bands, spectral indices and ERA5-Land environmental variables. | Closest framework to large-scale river mapping. XGBoost and RBF-SVR are evaluated with our covariates. The optical classification and satellite retrieval framework are not reproduced. |
| [Lu et al. 2026, Yangtze DOC reconstruction](https://xblk.ecnu.edu.cn/EN/10.3969/j.issn.1000-5641.2026.03.002) | Monthly DOC at Xuliujing, reconstructed for 2001–2020. RF is selected from five ML methods. | Represents environmental predictors for long-term DOC reconstruction. Our ecological RF retains the algorithm family with our available environment, current hydrological drivers and season. |
| [Casas-Ruiz et al. 2020, temperate river networks](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2019GB006495) | DOC GAM with catchment area, discharge condition and their interaction. A global environmental GAM and RF assess station-median DOC patterns. | Directly addresses spatial and hydrological DOC variation. Our pooled pyGAM uses ecological smooths and an area-by-local-discharge tensor interaction. |
| [Futter et al. 2007, INCA-C](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2006WR004960) | Daily soil and stream DOC with land-cover effects, soil transformations and in-stream processing. | Relevant process-model alternative. It needs daily air temperature, precipitation, effective rainfall and soil moisture deficit, and the paper calibrates against streamflow and DOC. Those inputs and calibration conditions are not available in this comparison. No numerical proxy is presented as INCA-C. |
| [Li et al. 2026, ELM-MOSART-DOC](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2025MS005307) | E3SM river DOC module with land-model runoff and soil carbon, ML parameterization, river routing and reservoir effects. The study reports predictions without DOC calibration. | A current dedicated process model, with [open source code](https://doi.org/10.5281/zenodo.17730412). Its GSWP3 forcing, NLDAS surface parameters and ELM equilibrium restart are absent from the ST357 feature artifact; its native analysis covers 1984–2012. It is discussed, not numerically evaluated in this frozen experiment. |

Aqua-OC's main text and author-hosted supplement were inspected. Its DOC experts
are DNN for optical type 1, XGBoost for types 2 and 3, and RF for type 4. It is
therefore insufficient to fit one regressor and call that a complete Aqua-OC
reproduction. Its satellite matching uses daily samples within one day of a
Landsat overpass; our monthly station aggregates cannot reproduce that pairing.
The authors' random sample split also differs from our held-out basin task.

The Yangtze study uses monthly temperature, vegetation, soil moisture, PET, GPP,
precipitation and discharge, plus annual land cover and population attributes.
Most of those dynamic environmental variables are absent here. The Casas-Ruiz
local GAM uses river-mouth discharge as a network condition and R mgcv cubic
regression splines; its global model also uses terrestrial primary productivity.
Our available local monthly discharge and pyGAM penalized splines change both
the input and fitting details. These adaptations are documented in
`configs/doc_domain_work_v1.json` before target scoring.

These published studies cannot supply comparable accuracy numbers without a
common dataset, query cells, information regime and metric definition. The
comparison does not mix original-paper R² or RMSE with local benchmark scores.

ELM-MOSART-DOC also means that predicting DOC without local calibration is not
unique to our proposed approach. A defensible distinction must specify the
permitted inputs, temporal and spatial representation, inference requirements,
and verified transfer performance. Its reported long-term average performance
does not itself answer our monthly reconstruction question, while our monthly
station metrics do not establish superiority over its process representation.

## Numerical arms and information

| Arm | Features | Fixed candidate budget |
| --- | --- | --- |
| Ecological RF | Environment 15 plus current temperature/discharge, masks, ages and season 8; 23 columns | 300 trees. Native DOC with minimum leaf 3; log1p DOC with leaf 3 or 10. Three seeds each. |
| Hydrological ecological GAM | Same 23 columns. Six cubic penalized spline bases per environmental or driver smooth; linear headwater, masks, ages and sin/cos season; a 5×5 area-by-discharge tensor | Native target with smoothing penalty 10; log1p with penalty 1 or 10. Maximum 100 iterations, tolerance 0.0001. One deterministic fit per recipe. |
| XGBoost | Full 113 columns: environment 15, twelve months of eight-channel drivers, current season 2 | 200 trees, learning rate 0.05, histogram trees, row subsampling 0.8. Native absolute loss at depth 3; log1p squared loss at depth 3 or 6. Three seeds each. |
| RBF SVR | Same full 113 columns; StandardScaler fitted on source training cells | Native target C=10, epsilon=0.2; log1p C=1 or 10, epsilon=0.1. Gamma follows source feature variance. Exact libsvm optimization without an iteration cap. One deterministic fit per recipe. |
| RF with history | Full 113 columns | Reuse the prior audited, source-selected random forest predictions. This separates the ecological feature restriction from the RF algorithm itself. |
| Fixed fusion reference | Environment encoder, twelve-month driver GRU and graph Transformer | Reuse the same audited fixed model predictions without further tuning. River connectivity allows permitted other-node covariate aggregation. |

The ecological RF and GAM intentionally represent contemporaneous environmental
approaches. They have less temporal information than the fixed fusion model.
The full-history RF, XGBoost and SVR comparisons are necessary before attributing
any gain to the learned temporal or spatial architecture. No numerical arm uses
DOC as an input, including source DOC history. Missing hydrological covariates
remain identifiable through their masks and observation ages.

Covariate scaling uses the existing source-station-only normalization. Additional
SVR scaling uses source training cells only. Windows end at the query month,
including available current-month drivers, so this is retrospective prediction
or reconstruction rather than a forecasting experiment.

## Evaluation contract

The ST357 dataset and existing geographical masks hold out entire station sets
in HUC4 tasks 1013, 1019, 0708, 1030 and 1101. These are previously inspected
development tasks within the Mississippi system. They do not constitute five
independent unseen global river systems.

Each new family has three declared recipes. Native source-validation MAE chooses
the recipe by averaging stochastic training seeds, with log1p MAE and lexical
recipe ID as exact tie breakers. All five regional selections are bound before
any new target prediction is scored. There are 120 unique candidate fits.
Deterministic methods' repeated evaluation seed labels do not create independent
fits or additional river replications.

Native MAE and log1p MAE remain joint primary endpoints. Secondary endpoints are
RMSE, predictive R², signed bias, station and basin macro MAE, source-training-Q90
tail MAE, and paired observed temporal and river-edge structure diagnostics.
Identical hidden observed cells and full coverage are required for each arm.
Metrics are computed per seed, then averaged; predictions are not ensembled.
The bootstrap resamples five whole basin clusters and is exploratory and
unadjusted for multiple comparisons.

Naturally missing station-months and virtual river locations have no truth here.
These experiments cannot verify a continuous river field, arbitrary virtual
stations, or global zero-shot performance. Original paper endpoints and
historical protocol versions remain unchanged.

## Reproducibility and audit

Output: `experiments/phase4_transfer/doc_domain_work_v1/`.
The isolated runtime dispatches training through `scripts/run_ladder.py`; the
original repository dispatcher remains unchanged. The manifest records commit,
dirty-workspace status, source hashes, data/mask hashes, dependency files and
reference prediction receipts. Each new prediction parquet has a sidecar.
Completed fits reject changed configurations or artifacts on resumption.

pyGAM 0.10.1 and XGBoost 3.1.0 use cache-local pinned packages. On macOS only the
XGBoost binary's OpenMP loader path is linked to the existing torch runtime and
the binary is signed locally. The algorithm implementation is unchanged; the
patched binary and OpenMP runtime hashes are recorded. The main project venv and
dependency lock are unchanged.

```bash
uv run python scripts/prepare_doc_domain_dependencies_v1.py
uv run pytest tests/test_doc_domain_work_v1.py
uv run ruff check .
uv run python scripts/start_doc_domain_work_v1.py --threads 2
uv run pytest tests/test_doc_domain_work_v1.py tests/test_kshot_protocol_v2.py
uv run python scripts/verify_doc_domain_work_v1.py
```

The independent audit recomputes MAE, log1p MAE, RMSE, predictive R², bias,
station macro and tail MAE, and basin macro errors from saved predictions and
original truth. It verifies identical query cells, source-only station roles,
selection receipts, deterministic repeats and source-Q90 thresholds. Numerical
results are deliverable only after this audit succeeds.
