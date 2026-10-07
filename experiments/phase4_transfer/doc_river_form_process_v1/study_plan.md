# River form, tributary integration and longitudinal DOC evolution

## Scientific question

How do the organization of tributaries and channel paths change the transmission
of DOC fluctuations to downstream stations? Whole-network form is the subject;
vegetation and upstream sampling coverage are explanatory background.

This exploratory follow-up reuses the source-role observations and location-
screened connections in `doc_river_mechanisms_v1`. Its analysis choices follow
the earlier morphology results: branch organization supplied more station-median
DOC information than footprint alone. This is not an independent confirmation
of that earlier result. No geographic/external prediction results are used and
no reconstruction models are trained.

## Observational population

- Monthly tributary pairs: independent upstream drainage, >=12 common months,
  existing gross gauge-area mismatch and drainage-order exclusions retained.
- Primary mixing population: area-weighted monthly records. Positive-flow,
  common-month records are a separate sensitivity, not a direct area-vs-flow
  competition. Strict same-day observations remain an availability diagnostic.
- Connected source-to-receiver paths: >=12 common permitted DOC months.
- Join the fixed real-network classes and source-screened morphology panel;
  unmatched/unclassified receivers stay in an inclusion ledger.
- Several pairs or incoming paths at a receiver are averaged first. Every
  receiver then gets one vote. Shared source/receiver connected components are
  the primary bootstrap unit; HUC4 resampling is a sensitivity. Neither creates
  independence between nested catchments without shared monitored stations.

## Measurements

Calendar sine/cosine and linear year are removed from `log1p(DOC)` on the exact
common monthly population of each connection. Define

`log_anomaly_sd_ratio = log(sd(downstream anomaly) / source RMS anomaly sd)`.

For two branches the denominator is `sqrt((var(a)+var(b))/2)`; for a path it is
the upstream standard deviation. Negative values mean reduced anomaly
variability, positive values amplified variability. This is not a mass-loss
estimate. Constant residual series give missing ratios, never artificial zeros.

Mixing diagnostics also include source Pearson synchrony, the mixture-to-source
anomaly SD ratio, downstream-to-mixture anomaly SD ratio, mixture/downstream
rank correlation, and mean log departure from the area/flow mixture. Path
diagnostics include source/downstream anomaly rank correlation and mean signed
log concentration change. Monthly samples do not measure event travel time.

## Morphology-centred comparisons

Fixed classes: elongated tributary-rich, mainstem dominated sparse, broad
tributary-rich. Report all class sizes. The broad-minus-elongated comparison is
descriptive; it is not a matched causal estimate. A class represented by one
independent component has no estimable population interval.

Continuous focal terms are evaluated separately:

1. `log_drainage_density` (tributary/channel richness);
2. `mainstem_share` (mainstem dominance);
3. `log_route_mean_scaled` (area-normalized channel path);
4. `route_distance_cv` (dispersion of paths to the outlet);
5. `mainstem_sinuosity` (channel curvature).

`log_basin_aspect` and `log_network_axis_ratio` are footprint comparators.
Separate-term coefficients do not establish independent effects among correlated
morphology terms. All terms/outcomes remain in the tables, without choosing
successful combinations.

Mixing controls: log basin area, monitored source drainage coverage, log distance
from confluence to receiver, and log common-month count. An additional model
controls source anomaly synchrony. Path controls: log basin area, log source
area, log source-to-receiver path length, added drainage fraction, storage path
fraction, wetland/forest cover contrasts, and log common-month count. These
background contrasts do not become the research subject. Missing nuisance
variables use median plus missingness indicator; focal morphology is not imputed.

Fit ordinary least squares, coefficient per observed receiver SD of morphology;
5,000 whole-component bootstrap draws, seed 42, plus HUC4 sensitivity. Reject
rank-deficient bootstrap fits and report their count. Intervals are exploratory
pointwise intervals across several correlated tests, not family-wise discovery
certificates. Leave-one-component-out estimates show domination by one monitored
system. Hydro-adjusted path correlation is recomputed on the exact positive-
flow and temperature-observed months (>=24) as a separate sensitivity.
The calendar-only fit on those SAME months is also retained so sample selection
is not confused with removing hydro variation. Design leverage is reported to
identify limited morphology overlap. These diagnostics complete the first
exploratory analysis; they do not choose a new successful morphology endpoint.

## Deliverables and interpretation

Save receiver and connection panels, class contrasts, conditional associations,
leave-one-component-out sensitivity, inclusion ledger and EN/CN figures. Verify
the DOC populations and variance formulas against source records. Distinguish:
geometric routing under fixed assumptions; observed downstream buffering;
conditional structure/process associations; and neural-model performance.

The outcome should identify which aspect of form deserves a process-informed
model or denser confluence sampling, without treating incomplete tributary
budgets as measured retention, or monthly covariance as a transport lag.
