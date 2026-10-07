# Real-network routing experiments: path spread, confluence and branch balance

## Question

Which part of river organization changes the downstream DOC signal when input
concentration and total flow are held fixed? This follow-up treats real channel
paths as the experiment's substrate. Vegetation is not an input weight.

## Two complementary experiments

### Whole catchment paths

Use all 297 source-screened real networks from `doc_river_morphology_effect_v1`.
Each positive-area, reachable incremental catchment contributes the same input
concentration pulse; fixed flow is proportional to its unique incremental area,
and total outlet flow equals one. Use the already measured shortest directed
midpoint-to-outlet channel distances, including secondary downstream links.
This is a single-path approximation for divergent networks, not a hydraulic
flow split solution. Retain unreachable-area fractions and midpoint resolution.

Distance/time is scaled by `sqrt(measured basin area)`, using the SAME normalized
velocity and pulse width across networks. Unlike the previous demonstration,
each network is not divided by its own maximum path length. Time is a scenario
unit, not months, days or a fitted travel time.

Compare actual paths with dispersion contracted to 50% and to zero around each
network's area-weighted mean path. Weights and mean delay remain unchanged.
The two contractions are virtual controls, not alternate observed rivers.

Measure outlet pulse peak, centroid, width, mass, and frequency response at fixed
periods 1/4/12 scenario units. A Gaussian input has SD 0.15 and peak one. Save
class summaries and the existing 22 environment-selected, nonnested elongated-
versus-broad pairs. Repeat numerical peak calculation at half time-step on those
pairs. Pure delay can change synchrony without changing the signal amplitude;
path dispersion can change amplitude, with frequency-specific cancellation.

### Real monitored tributary paths

Use the location-screened independent tributary-pair inventory with >=12 common
permitted DOC months, joined to the fixed morphology cohort. Inject TWO fixed
source pulses at the monitored upstream stations. The total source flow is one;
source weights use unique drainage areas or a balanced 0.5/0.5 control. These
two sources do not represent a complete field tributary budget.

Real paths are split into upstream branch segments and the shared post-confluence
segment. Compare the observed split with early/late virtual junctions at 20/80%
of the shorter total path, while retaining the two source-to-outlet distances.
The virtual junction experiment reallocates those distances; it does not claim
that a real confluence was physically moved or preserve the full channel budget.

Forcing is synchronous, or branch B leads/lags by 0.30 scenario units. Each pulse
has the same integrated concentration anomaly. Thus source synchrony changes
timing, while total flow and integrated input anomaly remain fixed. An additional
arrival-aligned diagnostic cancels the actual branch-delay difference.

Process cases:

- Uniform velocity, conservative routing: `v_a=v_b=v_common=1`.
- Flow-sensitive velocity: `v_branch=flow_share**(1/3)`, `v_common=1`.
  This is an unfitted illustrative speed scenario, not an empirical hydraulic
  law for these rivers. Exponents 0 and 0.5 are sensitivity controls.
- Uniform first-order processing: rate 0.10 per scenario time everywhere.
- Distinct common-segment processing: branch rate 0.10, common rate 0.80.
  These imposed rates demonstrate a mechanism; they are not estimated DOC rates.

Conservative output must preserve integrated input load. Under uniform speed,
changing only where two fixed paths merge must leave the outlet unchanged.
Uniform processing must also depend on total exposure, not the arbitrary split.
Distinct speeds or common-segment processing can create an independent junction
response. Report steady concentration separately from pulse attenuation.

## Reporting

All combinations remain in the tables. Average confluence combinations first at
each receiver, then use shared monitored-station system bootstrap (5,000 draws,
seed 42) plus HUC4 sensitivity. Whole-network contrasts reuse the fixed covariate
pairs and paired HUC4 bootstrap. No new pairing is selected by routing response.
These intervals describe sampled network geometry under imposed scenarios,
not uncertainty in a fitted physical DOC mechanism.

After the initial routing readout, add a spatial-resolution sensitivity for all
297 networks: distribute each incremental catchment's contribution uniformly
along its reach with five midpoint-quadrature sources. Preserve total weight and
mean delay. This tests whether sparse-network peaks depend on representing a
reach by one source location; it does not replace the primary midpoint result.
The distribution within a reach is an assumption, not a measured DOC source map.

Save representative pulse traces with fixed selection: the real network nearest
each existing class centroid, and the pair with median source path imbalance.
Plot all class comparisons. Report when increased path spread or a balanced
branch mix changes signal peak, and when source timing cancels/reverses the effect.

This experiment advances the morphology/process question. It does not retrain
the DOC model or turn monthly observational associations into causal effects.
