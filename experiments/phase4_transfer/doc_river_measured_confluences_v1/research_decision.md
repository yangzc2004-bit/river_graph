# Measured channel geometry and DOC at river confluences

## Main result

Real confluences have DOC departures that cannot be summarized by a single
rule such as wider channels producing lower DOC. In the same five-confluence
network, two downstream channels are approximately 1.6 times as wide as their
upstream mainstems, but their fall DOC departures have opposite signs:
**−13.7% and +19.5%** relative to the measured input-flow-weighted mixture.
Channel geometry must be connected to its hydraulic effect before being used
to explain DOC.

The field lead is the relative water residence time. Across the five
confluences, averaging the two campaigns first, greater downstream/mainstem
residence time per 100 m accompanies more negative DOC departure
(Spearman **−0.90**, omitted-confluence range **−1.00 to −0.80**).
The relation is descriptive of five configurations within one network.
It is a testable connection between structure and processing, not a measured
causal effect or a whole-network form ranking.

## Observations and accounting

The public [Plont data archive](https://www.hydroshare.org/resource/c5e687fa040e4707ba922002bafd18fd/)
provides five real Tom's Creek confluences in summer and fall 2021. This
reanalysis uses all 30 fully mixed downstream transect-position rows, carrying
the published mainstem and tributary measurements. Each position is an
archived mean of laboratory DOC measurements; analytical replicate values
are not separate available observations. There are ten confluence-campaign
units and one river network.

The [source study](https://doi.org/10.1029/2022WR034224) collected chemistry
during two baseflow campaigns and measured NaCl tracer discharge and velocity
within three days of sampling. These are paired spatial snapshots, not
synchronized storm hydrographs. They resolve input/output mixing, not storm
peak lag or annual carbon export. This is a new analysis of published
measurements and does not independently replicate the source investigators'
field work.

For each archived position:

```text
incoming water Q_in = Q_main + Q_tributary
incoming carbon J_in = Q_main C_main + Q_tributary C_tributary
input-normalized mixture C_mix = J_in / Q_in
mixture using measured receiving flow C_mix,Q = J_in / Q_downstream
observed receiving carbon J_out = Q_downstream C_downstream
```

DOC mg C/L multiplied by Q L/s gives mg C/s. The resulting carbon discrepancy
is an instantaneous spatial difference; it is not an integrated event load or
DOC removal rate. Published Q is constant across the three positions of each
receiving transect. Our mean downstream DOC is an equally weighted transect
estimate, not a measured velocity-weighted cross-section concentration.

## All ten field comparisons

DOC departure below uses the source-flow-normalized concentration mixture.
The carbon column compares the measured receiving flux with the incoming flux.
Each campaign percentage is the equal mean of its three position-level
percentages, preserving the source measurements paired with each position.

| Confluence | Campaign | DOC departure | Carbon-flux discrepancy | Receiving/mainstem residence time ratio |
|---|---|---:|---:|---:|
| 1 | Summer | −4.9% | −1.0% | 1.44 |
| 1 | Fall | −13.7% | −13.7% | 2.53 |
| 2 | Summer | −4.7% | −3.4% | 1.12 |
| 2 | Fall | +19.5% | +25.1% | 1.13 |
| 3 | Summer | +12.4% | +16.1% | 0.63 |
| 3 | Fall | +5.4% | +8.1% | 0.91 |
| 4 | Summer | −10.8% | −9.4% | 1.10 |
| 4 | Fall | −3.1% | −0.9% | 1.17 |
| 5 | Summer | −2.2% | +0.3% | 1.09 |
| 5 | Fall | +11.0% | +14.3% | 1.06 |

Six campaigns have negative concentration departure and four have positive
departure. Confluences 1 and 4 remain negative in both campaigns, and
confluence 3 remains positive. Confluences 2 and 5 change direction with season.
Thus the fixed confluence arrangement alone does not determine the sign.
For example, confluence 2 changes from −4.7% to +19.5% while its relative
residence-time ratio stays close to 1.12. Incoming conditions and other
seasonal processes remain necessary alongside geometry.

The source-flow-normalized specific conductivity prediction differs from the
mean measured receiving conductivity by at most **1.26%** across these ten
comparisons. The archived water surplus is **0–4.69%**, up to floating-point
roundoff. These reference calculations show that DOC's larger differences do
not have the same magnitude as the conductivity mismatch. They do not eliminate
flow uncertainty, unmeasured lateral sources or reaction as alternative causes.

In two individual positions, measured conductivity falls outside the two
incoming endpoints. Their conductivity-derived mixture is left undefined;
the fraction is not clipped. The valid tracer-derived DOC results broadly
retain the stronger departures, including fall confluence 2 at +19.8% and
fall confluence 1 at −16.1% (only two in-bound positions in the latter).
Unequal tracer denominators stay explicit in the result table.

## Which aspects of structure are promising?

The fall channel survey has 552 raw rows but only **69 width transects**.
Repeated width entries accompany individual depth positions, so each transect
receives one width weight. Thirty-one depth measurements are missing; they
are not filled. Mean depth is calculated within transects before averaging
transects, while maximum depth retains all measured positions.

All five downstream reaches are deeper than their upstream mainstems under
these equal-transect means (ratios **1.43–1.65**). Three are clearly wider,
one is approximately unchanged, and the fifth is narrower. Confluence 5 has
only two upstream and three downstream width transects.

| Structural/hydraulic measure | Population | Rank association with DOC departure | Omitted-confluence range |
|---|---|---:|---:|
| Relative residence time per 100 m | Five two-campaign means | −0.90 | −1.00 to −0.80 |
| Tributary water contribution | Five two-campaign means | +0.60 | +0.40 to +0.80 |
| Source DOC concentration contrast | Five two-campaign means | −0.10 | −0.40 to +0.80 |
| Relative channel width | Five fall snapshots | +0.10 | −0.80 to +0.40 |
| Relative mean depth | Five fall snapshots | +0.90 | +0.80 to +1.00 |
| Change in width variability | Five fall snapshots | +0.30 | −0.40 to +0.80 |

No p values or population confidence intervals are assigned to these five
connected configurations. The relative-residence association is negative
within each campaign as well (summer −0.70, fall −0.60), but these are the same
five locations. Fall depth has a positive association with DOC departure,
which does not support a simple deeper-channel–lower-DOC story. Wider and
deeper reaches also differ in flow, velocity and inputs, so a dimension alone
does not identify buffering or carbon processing.

These measurements refine the mechanism proposed by the path experiments:

```text
river paths and local junction geometry
                  ↓
arrival-time dispersion, water mixing and residence time
                  ↓
DOC concentration waveform and possible process-related departure
                  ↓
carbon export, conditional on incoming and lateral loads
```

The fixed-input scenarios establish how arrival distributions can lower a peak
without reducing total carbon. The new field snapshots add measured input and
output evidence for DOC departures associated with local hydraulics. They do
not establish that the original elongated or broad network classes differ in
storm buffering.

## Calculation notes and next empirical test

Two summer confluence-1 rows have provided mainstem/tributary fractions summing
to 0.960665 rather than one. We preserve them and recalculate fractions from
the archived discharges for every row. The downstream width CV at confluence
1 is 46.30% under one-width-per-transect weighting, versus 45.21% in the supplied
summary; the other four agree to numerical precision. Both versions remain
in the audit. This changes neither the raw observations nor the selection.

The next event test should measure branch A, branch B and receiver DOC/Q on
the same clock across rises and recessions, alongside rooted path geometry
and tracer travel distributions. Predict the receiver first from its measured
incoming waves, then test whether path dispersion and relative residence
time explain additional response changes. The most useful contrasts include
similar-width confluences with different hydraulic delays, and repeated
conditions at the same geometry. Account for lateral inputs before interpreting
a residual carbon discrepancy as net consumption.

This is a field-supported next hypothesis for a structure-aware model:
geometry should influence an arrival/residence operator, rather than merely
assigning a fixed DOC offset to a river form. No prediction model is trained
or changed in this version.

## Reproduction

```bash
uv run python scripts/fetch_doc_river_measured_confluences_v1.py
uv run python scripts/analyze_doc_river_measured_confluences_v1.py
uv run python scripts/verify_doc_river_measured_confluences_v1.py
uv run python scripts/plot_doc_river_measured_confluences_v1.py
uv run python scripts/plot_doc_river_measured_confluences_v1.py --chinese
```

The archive is CC BY 4.0. Attribution: Plont (2022),
*Plont_WRR_BGCNonconservativemixingConfluences_Data*, HydroShare; source
interpretation and collection methods: Plont, Scott and Hotchkiss (2023),
*Biogeochemical Processes Are Altered by Non-Conservative Mixing at Stream
Confluences*, Water Resources Research, doi:10.1029/2022WR034224.
