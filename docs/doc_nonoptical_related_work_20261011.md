# Environmental and network-based river DOC comparators

Date: 2026-10-11. Status: literature screening and proposed comparison design; no new model fits in this review. Aqua-OC replication is outside the current scope.

## Research question and interpretation

The target is DOC **concentration** reconstruction from accessible environmental descriptors, dynamic drivers and river-network information, including inference without target DOC observations. Comparisons with published methods assess predictive competitiveness under a shared task. They do not, on their own, attribute a gain to information fusion: architectures, objectives, tuning and input availability can also differ. Source contribution needs controlled ablations within the same model and evaluation protocol.

Environmental background, temporal information and network structure are information packages, not necessarily independent data sources. Published methods frequently already combine two or three packages. A model with dynamic environmental predictors is not automatically a pure environmental-background control, and a spatial stream network model with environmental covariates is not a network-only control.

## Primary-source shortlist

### Environmental machine learning: Yangtze monthly RF, 2026

*Long-term variations and driving factors of dissolved organic carbon transport in the Yangtze River using machine learning*, DOI [10.3969/j.issn.1000-5641.2026.03.002](https://xblk.ecnu.edu.cn/EN/10.3969/j.issn.1000-5641.2026.03.002).

The study compares machine-learning methods for monthly DOC and selects random forest. Predictors include vegetation, soil moisture, evapotranspiration, precipitation and human/land-cover descriptors. It represents environmental-driver regression without an explicit river-network operator or recurrent history encoder. Some inputs vary in time, so it should not be labelled static-environment-only. Its original inputs differ from the current workspace feature set.

The existing `doc_ecological_rf` result is an adaptation, not a complete reproduction. It includes current hydrological and seasonal features in addition to environmental descriptors; it is not an E-only experiment. Retain the strong history-aware RF/XGBoost controls as well, rather than selecting only the weaker ecological RF.

### Hydrological longitudinal GAM: Casas-Ruiz et al., 2020

*Delineating the Continuum of Dissolved Organic Matter in Temperate River Networks*, DOI [10.1029/2019GB006495](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2019GB006495).

The Fluvià DOC-concentration model uses cubic regression splines for upstream catchment area, river-mouth discharge at each sampling date and their interaction. Thus it captures longitudinal position and changing hydrological conditions without explicit network message passing or learned long-term memory. The paper also fits environmental GAM/RF models to median DOC across temperate rivers; those have a different response grain and should not be mixed with the campaign-level model.

A common-task GAM is a useful comparator. Substituting local discharge for river-mouth discharge, changing spline complexity or using the workspace ecological feature set makes it an adaptation. The original campaign data are linked through the paper's [Figshare project](https://figshare.com/projects/Bending_DOM_concept/80270); this review did not verify a successful download.

### Spatial Stream Network: Wampler et al., 2024 — priority new comparator

*The influence of burn severity on dissolved organic carbon concentrations across a stream network differs based on seasonal wetness conditions*, DOI [10.5194/bg-21-3093-2024](https://bg.copernicus.org/articles/21/3093/2024/).

The study sampled 129 sites in Oregon during four seasonal wetness campaigns. Its SSN models combine an environmental regression mean with network-aware spatial covariance. Tail-up and tail-down constructions account for along-stream distance and flow connectivity; Euclidean covariance is another candidate. The study predicts DOC at points spaced 1 km across the network and 100 m along two selected streams. This directly supports treating arbitrary prediction locations as a relevant comparison task, although four campaigns do not establish continuous temporal reconstruction.

The paper states that data and scripts are public at [Oregon State Scholars Archive, DOI 10.7267/zc77sz60m](https://doi.org/10.7267/zc77sz60m). The archive landing page could not be retrieved during this review; successful retrieval and script execution remain unverified. The original software uses R SSN and ArcMap STARS. [SSN2, maintained by USEPA and collaborators](https://usepa.github.io/SSN2/), provides a current implementation of the same statistical family; using SSN2 rather than the original SSN version must be disclosed.

Native SSN prediction can condition on training DOC residuals. This is legitimate training-label use, but access to target support observations must match the evaluation task. Within-network interpolation and zero-target-observation basin transfer should be reported separately: network covariance may supply little residual information between disconnected source and target networks. Do not replace SSN with ordinary Euclidean kriging and call it a reproduction.

A relevant predecessor is Rodríguez-Jeangros et al. (2018), *Analysis of Anthropogenic, Climatological, and Morphological Influences on Dissolved Organic Matter in Rocky Mountain Streams*, DOI [10.3390/w10040534](https://www.mdpi.com/2073-4441/10/4/534). It combines environmental variable selection using LASSO with SSN modelling of baseflow DOC. This is another environmental-plus-network method, with a baseflow target that needs alignment before comparison.

### Physics-guided GNN–Transformer: PhyST-HC, Budamala et al., 2026

*Hierarchical prediction of streamflow and organic carbon at event scale using a physics-guided AI hybrid framework*, DOI [10.1016/j.jhydrol.2026.136248](https://www.sciencedirect.com/science/article/pii/S0022169426013454).

The framework combines uncalibrated SWAT-C process states, a GNN encoding watershed connectivity and a temporal Transformer in hierarchical streamflow/organic-carbon prediction. This is a close multi-information framework and must be discussed when assessing novelty; simply combining environmental, temporal and graph components cannot be claimed as an unprecedented DOC architecture on the evidence of this search.

The [authors' public repository](https://github.com/VenkateshBudamala/PhyST-HC) contains model, input and simulation scripts. Inspection of [Functions_DOC.py](https://github.com/VenkateshBudamala/PhyST-HC/blob/main/scripts/Functions_DOC.py) confirms a temporal Transformer followed by GCN layers, SWAT subbasin/reach inputs, and predicted discharge from the preceding Q module. Plotting/output labels use **kg/day**, indicating a flux-oriented implementation that needs explicit target/unit verification before any concentration benchmark. No complete native data bundle or working native reproduction was established in this review.

Its native event-scale performance cannot be compared directly with our monthly concentration errors. Replacing SWAT-C states with our accessible inputs and changing the response to concentration produces an architecture adaptation. Keep that distinct from original-framework reproduction and from single-source controls.

## Proposed comparison set

Prioritize a strong environmental/history regression baseline (RF and XGBoost), the hydrological GAM family, and a faithful network-covariance SSN implementation. PhyST-HC is an essential close related work; numerical inclusion depends on confirming native target semantics, inputs and executable code. Original-input reproduction and common-input adaptation answer different questions and should have separate labels.

All common-task models need identical held-out queries, DOC units, source-only preprocessing/selection and explicit coverage. Report the frozen native and log1p MAE endpoints, with RMSE and predictive R² as secondary outcomes. Published adjusted R², KGE or errors on another dataset cannot be inserted into the workspace results table as if evaluated here.

## Controlled source-contribution design

Define E as the declared static environmental/regime/coordinate package; T as the dynamic temperature/discharge history with its masks, ages and declared seasonal encoding; and S as the river-network relations. Structural node attributes and common date encodings must be assigned explicitly so that a nominally S-only model does not silently receive E or T. Topology alone is generally insufficient to identify changing DOC levels; a network-only arm needs a clearly specified minimal predictor and should not be an intentionally uninformative straw man.

The principal contrasts are the full E+T+S model against T+S, E+S and E+T. Additional E-only, T-only and S-only arms can complete the information-package analysis. Preserve the encoder/operator structure where possible: replace removed features with a constant branch and use the existing no-message control, which retains the self path. To test temporal memory separately from current dynamic drivers, retain the same temporal operator and restrict historical values; GRU versus current-month MLP confounds history with operator choice. Use comparable tuning budgets and document capacity differences. Select on source validation only and report paired basin-level uncertainty.

No new experiment protocol is frozen by this memo. Any run-affecting implementation requires a new versioned protocol/runtime and the repository's provenance and contract checks; current freezes and primary endpoints must remain unchanged.

## Existing evidence and its limits

The completed [component comparison](doc_fusion_component_results_20261011.md) already shows that removing the environmental package raises MAE from 2.18480 to 3.67460 mg/L, with improvement from environmental inputs in all five held-out regions. It supports the environmental package's contribution, without isolating individual ecological descriptors.

The separate [development benchmark table](../experiments/phase4_transfer/doc_fusion_benchmark_development_v1/analysis/summary.csv) has MAE 2.231835 mg/L for the E+T no-message control and 2.183878 for full fusion. This is a favourable graph-contribution point estimate, not by itself a robust general transfer claim. The two studies have different comparison parents and must not be presented as one completed factorial experiment.

The component study does not establish a reliable temporal-memory advantage or GRU optimality. The [DOC-family comparison](doc_domain_work_results_20261011.md) also has a GAM adaptation with better RMSE/R² than full fusion and an XGBoost MAE difference whose paired basin interval crosses zero. The present evidence does not support superiority on every metric, universal multi-source superiority, or global SOTA.

Current validation covers observed monthly station cells in five regions of the Mississippi system. Independent rivers, naturally missing DOC cells, target-only graphs and arbitrary river locations require separate evidence. These are the more meaningful eventual distinctions from existing within-network studies.
