# Ecology- and hydro-aware temporal graph reconstruction of sparse water-quality observations

## Abstract

Sparse water-quality records limit both ecological interpretation and the
ability to compare monitoring conditions across space and time. We develop
H2X-T, a causal temporal extension of an ecology-aware river-network graph
model, and evaluate it under controlled spatial, temporal, and random
missingness. The model applies a shared spatial graph encoder to monthly
snapshots and uses a 12-month gated recurrent unit to reconstruct one target
analyte at a time. We study dissolved organic carbon (DOC), pH, and specific
conductance on an audited 357-station, 654-month river-network cohort. In a
five-seed matched comparison, H2X-T reduced DOC mean absolute error by
37.5--47.2% and specific-conductance error by 38.8--57.2% across the tested
missingness regimes. pH gains were smaller (0.5--3.8%) and depended on the
missingness regime. A window diagnostic showed that most pH and conductance
improvement appeared by six months, whereas DOC changed little with window
length. History ablations indicate that the value of a longer input history
cannot be assigned to one channel or to strict chronological order alone. The
results show that temporal reconstruction gains are analyte- and
missingness-dependent. The model is an instrument for quantifying this
conditional information value, rather than a claim that one architecture is
universally optimal.

Here, causal describes the information-flow constraint that a prediction at
month *t* uses no future month. It is not a causal-inference claim.

## 1. Introduction

Long-term river monitoring networks rarely observe every analyte at every
station and month. Missing observations arise from changes in sampling
programs, seasonal access, instrument failure, and uneven allocation of
monitoring effort. The resulting gaps are ecological as well as statistical:
missingness can align with river position, watershed land cover, hydrologic
regime, season, and the historical availability of particular analytes. A
single overall error score can therefore hide the conditions under which a
reconstruction is useful.

Machine-learning models can combine heterogeneous information sources that
are difficult to integrate with independent interpolation. A river network
provides a spatial relation between monitoring stations. Watershed ecological
features describe the context in which a station is embedded. Temperature and
discharge provide dynamic hydro-ecological signals, while the history of the
target analyte can provide local persistence. The scientific question is not
whether a more complex neural network can reduce average error. It is which
information sources remain useful under a particular missingness regime and
for a particular analyte.

We address this question with H2X-T, a temporal extension of an ecology-aware
transport graph model. The spatial trunk is applied independently to each
monthly graph snapshot. A causal GRU then combines the current representation
with a fixed history window. We train separate models for DOC, pH, and
specific conductance so that the reported comparisons do not depend on a new
multi-task coupling assumption. H2X-T is compared with the same spatial model
without temporal memory.

Our analysis has four goals:

1. quantify the value of adding temporal memory under random, temporal, and
   spatial missingness;
2. determine whether the value differs across DOC, pH, and specific
   conductance;
3. identify how much history is useful using a 1/3/6/12-month diagnostic; and
4. distinguish a longer useful history from a specific target-history or
   chronological-order mechanism.

The study is intentionally conditional. It concerns the audited ST357 cohort,
frozen masks, preprocessing, and matched training budget. It does not claim
that H2X-T is causal, mechanistic, or universally superior, and it does not
turn ensemble spread into a calibrated monitoring-risk score.

## 2. Materials and methods

### 2.1 River-network cohort and analytes

The primary cohort contains 357 stream stations in the audited ST357 river
network, 324 graph edges, and 654 monthly time points. The same station-month
layout is used for DOC, pH, and specific conductance, with target-specific
observed-label intersections materialized before training. Each analyte is
trained separately. DOC and specific conductance use a log1p target transform;
pH uses raw-scale standardization based only on training cells.

The monthly input includes temperature and discharge, their visibility flags,
seasonal sine/cosine terms, static station coordinates, ecological regime
features, and the target-value and target-visibility channels. The graph input
uses the frozen river edge set and edge attributes. All feature standardization
statistics are computed from the training role.

### 2.2 Missingness regimes and visibility

We use four frozen missingness families. E1 masks random observed cells. E2a
and E2b represent temporal extrapolation settings. E3 holds out spatial
stations. For each analyte, the frozen role cells are intersected with that
analyte's observed-label mask. Training and context labels can enter the input
according to the role definition. Validation labels remain hidden during
training and early stopping. Test labels are used only for the final metric
calculation.

The same target masks and query cells are used for the snapshot and temporal
models. This makes the model difference a paired comparison rather than a
comparison of different missing-data samples.

### 2.3 H2X snapshot and H2X-T

The snapshot baseline is the ecology-aware transport graph model H2X. It uses
a two-layer spatial encoder with hidden width 64, dropout 0.1, transport edge
gates, and the ecological encoder. H2X-T reuses this spatial representation
at each month and applies a one-layer GRU with hidden width 64. The current
month is the final recurrent step; the history window contains only current
or earlier months. The primary H2X-T configuration has a 12-month lookback.

The model produces one prediction for every station-month in the full grid.
The primary reported prediction is the five-seed median. The seed standard
deviation and interquartile range are stored as ensemble-dispersion diagnostics.
They are not prediction intervals.

### 2.4 Training and reproducibility

All primary comparisons use five training seeds (42--46) and the matched
10-epoch, patience-3 budget. This budget is shared by H2X and H2X-T. Each run
writes a prediction product and a provenance sidecar binding the dataset,
target mask, configuration, runtime snapshot, and prediction hash.

The historical T4 and T5 diagnostics use three seeds (42--44). T4 reverses the
preceding history steps while retaining the current step. T4 hydro-only sets
the target-value and target-visibility channels to zero at all months. T5
current-only retains the GRU wrapper but uses a one-month lookback. These are
mechanism diagnostics and are not additional primary endpoint tests.

### 2.5 Metrics and statistical comparison

The primary metric is mean absolute error on the hidden query cells. We also
report RMSE, R2, log-space error, and a train-derived Q90 tail MAE. Tail cells
with fewer than 20 observations are flagged unstable rather than interpreted
as a reliable tail estimate.

For the five-seed comparison, errors are averaged across seeds at each query
cell before bootstrap resampling. We use 2,000 paired bootstrap replicates and
report station-clustered and month-clustered percentile intervals for the MAE
difference. Training seeds are not treated as independent ecological clusters.
The relative reduction is `100 x (1 - MAE_H2X-T / MAE_H2X)`.

### 2.6 Analysis provenance

T8 and T9 are synthesis stages over already audited products. They do not
select a model or change a primary endpoint using hidden test labels. The
complete input ledger, source hashes, tables, figures, and analysis scripts
are stored under `experiments/phase4_transfer/temporal_h2x_v1/t8_synthesis/`
and `t9_products/`.

## 3. Results

### 3.1 Five-seed matched comparison

H2X-T reduced MAE in every analyte-by-mask family. DOC reductions ranged from
37.5% in E3 to 47.2% in E2b. Specific-conductance reductions ranged from
38.8% in E3 to 57.2% in E2b. Both analytes improved for all five seeds in all
four families.

The pH effect was smaller. Reductions ranged from 0.5% in E3 to 3.8% in E2b.
E1, E2a, and E2b improved for four of five seeds, while E3 improved for three
of five. The station-clustered E3 interval included zero, so we report this as
a regime-dependent small effect rather than a general pH improvement.

The largest DOC and conductance gains occurred in E2a and E2b, the temporal
missingness settings. The result is consistent with a temporal representation
being most useful when the query month is separated from the observed target
history. The same model also improved E3, showing that the gain is not limited
to temporal masking, although the magnitude is smaller for conductance in E3
than in E2.

### 3.2 Temporal window length

The E2 window diagnostic compared 1, 3, 6, and 12 months. Mean MAE across E2a
and E2b decreased from 1.420 to 1.415 for DOC, from 0.362 to 0.301 for pH,
and from 279.12 to 245.75 for specific conductance. The pH and conductance
curves showed most of their improvement by six months. DOC was nearly flat.
We retained 12 months because it produced the lowest observed error and gives
a single configuration across analytes; six months is a credible lower-cost
alternative for future operational testing.

### 3.3 History diagnostics

The reverse-history and hydro-only arms remained within 0.6% of full H2X-T in
MAE across the three-analyte and four-mask diagnostic grid. In contrast, the
current-only control was 20.9% worse for pH and 13.8% worse for conductance in
E2a/E2b, while DOC changed by 0.5%.

This pattern indicates that a longer history representation is useful for pH
and conductance, but the available ablations do not identify a single causal
channel for the gain. The reverse-history arm changes the order while keeping
the history amount fixed. The hydro-only arm removes both current and past
target channels. These experiments therefore constrain the interpretation
without supporting a claim that chronological order or target persistence is
the sole mechanism.

### 3.4 Spatial and station-level products

The five-seed ensemble products provide a full station-month reconstruction
for all three analytes and four masks. The E2a station maps show heterogeneous
error across the river network. Descriptive station traces show the model's
seasonal reconstruction and local departures from observations. These products
are intended to support ecological interpretation and downstream monitoring
analysis; they are not themselves a new sampling-optimization endpoint.

## 4. Discussion

### 4.1 What temporal memory adds

The central result is conditional information value. A causal temporal wrapper
substantially improves reconstruction when the target's missingness removes
or separates temporal information, especially for DOC and specific
conductance. The effect is not a uniform property of the model: pH responds
less strongly, and its spatially held-out result is close to parity.

The window curve gives a practical interpretation. A small history is not
enough for pH and conductance, while extending the history beyond six months
has diminishing returns. This suggests that monitoring systems can consider a
shorter operational window when compute or latency matters, but the current
study does not replace the 12-month model as the main configuration.

### 4.2 Why the graph remains useful as an instrument

The study does not claim that the transport graph architecture itself is the
scientific discovery. Its role is to place ecological and hydro-temporal
information in the river-network context, so that the same temporal wrapper
can be tested across analytes and missingness regimes. The meaningful result is
the pattern of conditional gains, not a model name or an architectural novelty
in isolation.

### 4.3 Implications for ecological monitoring

A reconstruction model can make sparse monitoring records more comparable
across months and analytes, but only if its performance is reported by the
missingness process. The large DOC and conductance gains indicate that temporal
and ecological inputs can recover useful information in extrapolation settings.
The smaller pH effect is equally informative: it shows where the same
information sources do not transfer strongly. This heterogeneity can guide
which analytes receive model-assisted gap filling and which remain dependent
on direct observation.

### 4.4 Uncertainty and operational use

The model products include seed dispersion, but dispersion is not a calibrated
uncertainty interval. Earlier calibration experiments demonstrated a
coverage--width trade-off, and uncertainty ranking was not consistently
established across tools. We therefore treat uncertainty diagnostics as
supporting information and do not claim stable ecological blind-spot detection
or active-sampling optimization from the present results.

### 4.5 Limitations and next tests

The comparison is within one audited river-network cohort and uses a matched
10-epoch budget. A longer convergence study could change the absolute margin,
although it would not change the paired design. The window experiment changes
both history length and recurrent computation. The history ablations do not
separate hydro, ecology, seasonality, and target persistence into independent
causal components. External-basin replication is also outside the current
results package.

The next useful test is an external or held-out-basin replication with the
same feature processing, masks, and evaluation code. A second priority is a
channel-factorial experiment that keeps the 12-month window fixed while
separately retaining or permuting hydro, ecological, seasonal, and target
channels. Such an experiment should be designed as a new protocol rather than
added retrospectively to the current primary endpoints.

## 5. Conclusion

H2X-T turns the existing ecology-aware river graph model into a causal temporal
reconstruction system. On ST357, it substantially reduces DOC and specific
conductance error across random, temporal, and spatial missingness, while pH
shows smaller and regime-dependent gains. The value of temporal context is
therefore real but conditional: analyte and missingness regime determine how
much information can be recovered. This framing keeps the GNN as a useful
scientific instrument while placing the ecological result—the conditional
value of hydro-temporal information for sparse monitoring—at the center.
