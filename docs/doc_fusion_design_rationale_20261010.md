# DOC fusion model research rationale

Date: 10 October 2026. Status: research synthesis for the current architecture candidate.

The research goal is to reconstruct river DOC variation by combining environmental background, temporal history and river-network relationships. The current candidate uses a small environmental encoder, a GRU for hydrological history and a graph Transformer for spatial interaction. The spatial comparison supports prioritizing this candidate among the tested implementations. The environmental encoder and GRU remain motivated starting choices whose superiority has not been tested in that comparison.

This note consolidates the research argument, relevant literature, implementation facts and evidence needed for subsequent decisions. It concerns the new standalone architecture branch. The existing manuscript evaluates a different tree/recurrent/source-attention procedure; its results cannot be assigned to this candidate. The existing manuscript, frozen claims and fitted release remain separate.

## Research question and intended output

River DOC varies with catchment properties, hydrological mobilization and conditions along connected channels. Uneven sampling leaves both missing periods at monitored sites and locations without local DOC history. A record assembled from stations sampled in different months cannot directly describe a river's concentration distribution at a common time. Reconstruction would provide comparable location–time estimates, helping characterize seasonal variation, concentration gradients and unusual conditions. Carbon-load calculations would additionally require compatible discharge estimates and uncertainty.

The central question is: **Can shared environmental, temporal and spatial representations reconstruct DOC at missing locations and times, including in rivers that contributed no DOC training observations?** The rationale does not require a global-climate framing. [CAMELS-Chem](https://hess.copernicus.org/articles/28/611/2024/) documents variation in chemistry sampling frequency and record length. In the local ST357 dataset, DOC occupies 22,571 of 233,478 station–month cells, or 9.67%. That percentage describes this cohort and grid; it is not the fraction of the world's rivers monitored for DOC.

The intended output is a queryable DOC field over river positions and times. A virtual station would extract a trajectory from that field. The present experiment evaluates monthly predictions on a sampled-station graph, using observed withheld cells as truth. It has not established reconstruction at every physical reach or at continuous times. New locations will need environmental attributes, temporal drivers and a consistent connection to the river representation. A continuous query decoder and predictive uncertainty are future implementation questions.

## Related research and the remaining question

| Research family | What it contributes | What must be established for this project |
| --- | --- | --- |
| Environmental analysis and tabular prediction | Catchment attributes explain geographical differences; trees provide a strong nonlinear reference. [Yang et al. 2017](https://doi.org/10.3390/w9060383) and [Gorishniy et al. 2021](https://arxiv.org/abs/2106.11959). | Whether a learned environment representation contributes beyond raw attributes and strong trees under identical information conditions. |
| Process-based carbon models | Explicit carbon production, leaching, transport, transformation and aquatic exchanges. [ORCHILEAK, Lauerwald et al. 2017](https://gmd.copernicus.org/articles/10/3821/2017/gmd-10-3821-2017.html). | Assess its input and parameter requirements for the reconstruction setting; process constraints are potential complements, whereas the current neural candidate does not implement these carbon budgets. |
| Stream-network spatial statistics | Flow connectivity and stream distance can support prediction at unsampled locations. [Ver Hoef et al. 2006](https://link.springer.com/article/10.1007/s10651-006-0022-8). | Whether neural interactions improve reconstruction beyond an appropriately specified network-statistical comparator, with matched query coverage. |
| Recurrent hydrological models | Shared static attributes and dynamic sequences can support prediction outside training basins. [Kratzert et al. 2019](https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/2019WR026065). | Streamflow success motivates testing shared representations; it does not establish DOC transfer or select GRU over other temporal models. |
| Missing-data recurrent and graph models | Explicit masks, elapsed time and relational representations address incomplete multivariate records. [Che et al.](https://arxiv.org/abs/1606.01865) and [Cini et al. 2022](https://arxiv.org/abs/2108.00298). | Reconstruction without any local DOC history is a distinct information condition from filling gaps in partially observed target series. |
| River recurrent graphs and graph Transformers | River graphs encode segment relationships; attention can combine graph structure with broad interactions. [Jia et al. 2021](https://www.usgs.gov/publications/physics-guided-recurrent-graph-model-predicting-flow-and-temperature-river-networks) and [Ying et al. 2021](https://arxiv.org/abs/2106.05234). | Whether river relations supply additional DOC information beyond environmental and temporal similarity. Evidence from temperature or generic graph benchmarks does not answer this DOC question. |
| Satellite DOC retrieval | [Aqua-OC, Tian et al. 2025](https://authors.library.caltech.edu/records/b74cg-58e81), already estimates long-term, reach-scale organic carbon using Landsat, optical water types and multiple input sources. | Position the proposed model through its missing-data conditions, input requirements and transfer tests. Broad claims that previous work cannot reconstruct river DOC over space and time are unsupported. |

These approaches already address important parts of the problem. The proposed contribution is an evaluated way to combine complementary information under sparse DOC supervision, with explicit tests of component contributions and geographical transfer. Combining familiar modules alone is insufficient to establish novelty. Published errors from different datasets, targets and validation designs must not be ranked against our MAE table.

## Current candidate and information flow

The implementation is [RiverArchitectureModel](../src/river_graph/models/river_architecture_comparison.py), with the frozen experiment in [architecture comparison v1](../experiments/phase4_transfer/doc_river_architecture_comparison_v1/protocol.json).

| Information | Actual inputs in this branch | Representation |
| --- | --- | --- |
| Environmental background | 13 river/ecological attributes and latitude/longitude | Two linear layers with GELU, producing 24 features; 984 parameters |
| Temporal history | Water temperature, log discharge, their availability masks and elapsed observation times, and seasonal sine/cosine | Causal rolling 12-month GRU with hidden width 12; 792 parameters |
| Spatial relationships | Known directed station graph; self, direct upstream, other ancestor, direct downstream, other descendant and unconnected relation categories | Two graph-attention blocks with three heads and learned relation biases |

Environmental and temporal representations, plus current season, are concatenated and projected to a common state. Spatial attention then updates these states, and a readout predicts standardized log DOC. The training-source normalization and inverse log transform produce concentrations in mg/L. This is a shared model fitted jointly, rather than three separately predicted DOC series averaged together. The environmental vector does not currently condition the GRU gates directly.

No DOC values or history, pH, conductivity, station identity embedding or station-specific fitted parameter enters this branch. DOC is the supervision target. Historical memory refers to hydrological drivers and their availability, making inference possible without local DOC observations. This deliberately stringent input condition differs from the existing source-attention manuscript, which permits monitored source DOC.

## Environmental encoding

**Why this information is needed.** Catchments with similar flow and season can have different carbon sources and concentration backgrounds. The input attributes describe stream order, drainage area, slope, sampled-graph headwater status, forest/crop/urban/wetland cover, climate normals, soil organic matter, elevation and baseflow context. Yang et al. identify environmental associations that vary by region and stream size. This motivates representing environmental differences; it does not prescribe a particular encoder.

**Why start with a small learned encoder.** A shared nonlinear representation can express combinations of heterogeneous attributes and give temporal and spatial states a common feature space. It also avoids fitting a separate identity vector for every training station. These are design reasons for a compact MLP, with modest capacity relative to available supervision. They do not establish that two layers or 24 features are optimal, or that an embedding is intrinsically transferable outside the training environment range.

First compare the complete candidate with a constant environmental branch to measure the branch's incremental value while retaining the same graph and temporal inputs. Then compare a linear projection, a modest residual MLP and raw-attribute fusion. Feature-attention encoders are additional candidates if those controls reveal a representation limitation. Gorishniy et al. found strong residual-network and feature-Transformer baselines in tabular learning, with no universally superior solution relative to boosted trees. Consequently, keep a strong environmental tree predictor as a separate methodological comparator, not as another required deployment model. An equal-information neural–tree comparison must give both methods the same permitted history and context.

**Project evidence.** The four spatial arms all used the same encoder, so their ranking cannot validate the encoder. In a different, earlier source-role study, adding detailed ecological categories to the retained neural encoder did not establish an overall improvement: complete MAE was 1.769340 versus 1.769053 mg/L. [Detailed composition study](../experiments/phase4_transfer/doc_composition_encoder_v1/research_decision.md). This argues against assuming that more categories automatically improve prediction; it does not rank encoders for the new branch.

**Decision.** Retain the compact MLP provisionally. First compare it with simpler encoding under fixed temporal and spatial modules. Environmental conditioning of the recurrent state is a promising separate question: EA-LSTM explicitly uses catchment attributes to represent different hydrological behavior ([Kratzert et al.](https://arxiv.org/abs/1907.08456)). Its DOC value remains untested.

Static land cover in the present data is a 2019 proxy, and climate normals summarize a fixed period. They cannot describe historical land-use changes across the entire DOC record. Audit static missing-value conventions before interpreting embeddings; legacy dataset assembly uses finite sentinels for some missing attributes. Water temperature and discharge are also not guaranteed globally available inputs: their raw monthly coverage is approximately 27.8% and 51.6% in ST357. Air-temperature reanalysis would be a different input, requiring a separately evaluated substitution.

## Temporal modelling

**Why history is needed.** Current environmental conditions alone may not distinguish how a catchment reached its present state. High-frequency DOC studies observe storm hysteresis and dependence on season and land cover ([Vaughan et al. 2017](https://doi.org/10.1002/2017WR020491)). Those observations motivate testing antecedent information. A monthly model can represent monthly dependencies; it cannot resolve the original subdaily event mechanisms.

**Why start with GRU.** Its gated recurrence explicitly processes the ordered driver history while maintaining a compact hidden state. Compared with LSTM at the same input and hidden widths, its parameterization is smaller. With a short monthly window, this provides a practical starting tradeoff between sequential representation and model size. [Chung et al. 2014](https://arxiv.org/abs/1412.3555) found GRU competitive with LSTM on their sequence tasks; those were not DOC or hydrological transfer experiments.

Missingness must be represented explicitly. Current inputs distinguish a missing standardized value from an observed value near zero and provide time since the last driver observation. Che et al.'s GRU-D supports the broader idea of using masks and elapsed time. The present code is an ordinary GRU receiving these channels, not GRU-D with learned input/hidden-state decay. Sparse observations do not themselves prove that GRU is the best temporal model, and GRU does not automatically resolve irregular sampling or informative missingness.

Twelve months spans one annual cycle and is a reasonable starting horizon alongside explicit seasonal encoding. It is not a measured DOC residence time or a demonstrated optimal memory length. In the earlier retained ensemble, training with 24 rather than 12 months did not establish an upgrade: 1.772521 versus 1.769053 mg/L. [Window comparison](../experiments/phase4_transfer/doc_longer_history_v1/research_decision.md). Another earlier experiment found a small historical-driver benefit over its matched current-input control, while the gain over the retained complete procedure was uncertain. [Hydrological memory study](../experiments/phase4_transfer/doc_daily_hydro_memory_v1/completion.md). These results favor testing information value before expanding the temporal backbone; they do not select GRU for this standalone architecture.

**Decision.** Retain GRU provisionally. With environment and spatial modules fixed, compare current-month inputs, a lag-vector/summary MLP using the same history, GRU, and compact LSTM or temporal attention. Use separate 6/12/24-month comparisons if history contributes. A same-weight history truncation is a sensitivity diagnostic, not a substitute for separately trained matched competitors. Temporal-Transformer superiority cannot be inferred from the spatial-Transformer result.

Each window ends in the target month and includes that month's available inputs. This is retrospective reconstruction, not a forecast issued before the month. Future observations are excluded. A retrospective bidirectional smoother would be a separate legitimate task if future driver information were explicitly allowed.

## Spatial modelling

**Why river relationships are needed.** Close geographical positions can lie on different tributaries, while distant stations can be linked along a channel. Stream-network statistics show why connectivity and flow distance deserve explicit treatment. River relationships can supply useful predictive context, but their contribution beyond local environment and history must be measured.

**Why graph Transformer is the current candidate.** Its attention can directly compare states at distant nodes without requiring every useful relationship to traverse a short stack of adjacent station edges. Structural relation biases identify the direction and connectivity of those pairs. Graphormer provides general evidence that encoding graph structure is important to graph attention. Applying that idea to a sparse river observation network is a project hypothesis, supported provisionally by the following controlled comparison.

All four arms shared environmental inputs, the 12-month GRU, initialization, loss, nominal parameter tensors, optimizer budget and validation rule. Self-only local attention has inactive inter-node query/key comparisons, so nominal parameter matching does not establish identical effective capacity. Five whole-HUC4 holdouts and three seeds produced 60 fits. The primary endpoint is equal-cell DOC MAE, averaging seed errors rather than ensemble predictions, over 7,262 observed cells at 104 withheld stations.

| Spatial operator | MAE in mg/L | Change in error relative to local |
| --- | ---: | ---: |
| Local self path only | 2.231835 | Reference |
| Directed upstream-neighbour GNN | 2.326068 | 4.22% higher |
| Stream-order-scanned hierarchical GNN | 2.296068 | 2.88% higher |
| Graph Transformer | 2.183878 | 2.15% lower |

Graph Transformer improves over the two GNN implementations by 6.11% and 4.89%, respectively, in all five regions. Its improvement over the local control has a 95% basin-cluster bootstrap interval of −0.92% to 9.83%, crossing zero. Hierarchy improves over the ordinary directed GNN by 1.29%, with an interval of −3.70% to 5.29%. With only five development regions, these intervals are exploratory. [Full results and audit](../experiments/phase4_transfer/doc_river_architecture_comparison_v1/analysis/research_decision.txt).

**Decision.** Prioritize graph Transformer among the tested spatial implementations. Do not claim universal Transformer superiority or a stable benefit of adding space to the local predictor. Stream-order hierarchy is an exploration rather than an established main contribution. River order is a structural attribute, not a neural-layer count: same-order nodes need not be adjacent, and relevant edges can cross order groups. The tested hierarchy preserves actual same-order and cross-order edges.

Three limits affect the explanatory argument. First, dense attention also permits downstream and unconnected keys, whereas the directed GNN uses upstream neighbours. The comparison changes an architectural bundle; it does not isolate global reach or graph bias. Second, relation biases provide direction information but do not enforce one-way transport, travel-time delay or carbon conservation. Attention weights are not carbon fluxes. Third, withholding graph nodes changes induced connectivity. Dense attention may benefit from predictive similarity where sampled edges are absent; that possible explanation has not been isolated experimentally.

The next structural controls should therefore include an identically trained Transformer without pairwise graph bias, flow-compatible topology randomization with matched availability, and an upstream-restricted attention variant. These distinguish general cross-node information from the contribution of correct river relationships. Dense attention scales quadratically in node count. [GraphGPS](https://arxiv.org/abs/2205.12454) and [Exphormer](https://proceedings.mlr.press/v202/shirzad23a.html) motivate sparse or local/global alternatives for a full-reach river representation; they have not been tested here and are not established DOC upgrades.

## Fusion and transfer argument

The decomposition assigns distinct roles: environmental attributes describe persistent differences, driver history describes changing local state, and graph interaction supplies relational context. Concatenation and projection followed by spatial interaction is a simple way to let the prediction depend jointly on these roles. It does not guarantee statistically identifiable environmental, temporal and spatial effects. Modularity helps design controls; it does not make neural weights causal explanations.

The fusion rule is also provisional. Compare the current concatenation with environment-conditioned temporal representations or a modest reliability-aware fusion only after establishing the value of the individual information branches. Changing several modules together would obscure which change helped. Keep the existing local arm to quantify the spatial increment; it is an experimental control within the three-information research question.

Shared parameters, feature-based node states and variable-node-count processing make the candidate compatible with new rivers without learned station IDs. Successful transfer additionally depends on the training environments, dynamic-input quality and graph representation. Training every global river is neither an established requirement nor a guarantee of generalization. Diverse monitored catchments and deliberately withheld river systems are more informative tests of the objective. The current five internal regions have already been inspected during development and cannot serve as fresh blind external confirmation.

## Sequential investigation and decision rules

| Step | Focused question | Comparison and fixed conditions |
| --- | --- | --- |
| 1 Environmental representation | Does the environmental branch help, and does nonlinear encoding add value? | Constant environmental branch control, then linear/raw fusion versus compact MLP; retain the same drivers, GRU, graph Transformer, missingness treatment and supervision. Include strong trees as a separate equal-information comparator. |
| 2 Temporal representation | Does history help, and does recurrence add value beyond access to history? | Current-input control, equal-history lag MLP, GRU and compact temporal alternatives; retain environmental and spatial choices. Compare horizons separately. |
| 3 River relationships | Does the correct graph add value beyond unrestricted node attention? | Local control, no-pairwise-bias Transformer, real graph bias, and matched structural controls. Distinguish connectivity, permissible directions and receptive field. |
| 4 Fusion and transfer | Does the combined model generalize beyond development geography? | Freeze the selected candidate using source/validation evidence, then evaluate a reserved river and matched observed-query population. Report input availability and high-DOC behavior alongside overall error. |

These are research questions for separate, versioned experiments, not newly executed results. Specify endpoints, allowed information, masks and selection rules before fitting each comparison. Keep native MAE in mg/L, paired query cells and complete observed-query coverage consistent; use basin-level uncertainty for transfer questions. Repeated seeds do not create additional independent rivers. Do not promote a variant from a favorable subgroup or from improvement only over a weakened control. Consider consistency across regions, high-DOC errors and computational cost alongside the predefined primary endpoint.

## Wording that the current evidence supports

The core motivation can be expressed as: “Sparse and asynchronous river DOC observations motivate reconstruction methods that combine catchment context, antecedent hydrological information and river-network relationships, so that missing location–time concentrations can be estimated under explicit input-availability conditions.”

The architecture rationale can be expressed as: “We use a compact environmental encoder to represent catchment heterogeneity, a gated recurrent module to encode ordered hydrological history, and graph-biased attention to combine local states across the river network. Controlled development comparisons prioritize the graph-attention implementation over the tested directed and hierarchical neighbour operators. The environmental and temporal choices remain subject to matched component evaluation.”

The strongest defensible architecture statement is **a motivated fusion design with a provisionally selected spatial implementation**. The present evidence does not establish an optimal three-module architecture, independent unseen-river performance, full continuous river-field accuracy or physically identified transport coefficients.

## Evidence records

[Evidence matrix](doc_fusion_design_evidence_20261010.csv) records supported claims, inferential limits and needed comparisons. [Source record](doc_fusion_design_sources_20261010.json) binds the local implementation and experiment documents to file hashes and records the checked literature links. No model training or architecture change was performed while producing this synthesis.
