# Incoming timing and post-junction response in real river forms

## Question

When the same carbon pulses enter a real river junction, how do their arrival
alignment and the shared downstream response jointly change the outlet peak and
duration? Keep the original elongated, sparse and broad-branching form labels.
This follows the field-confluence study without fitting its five DOC deviations.

## Real structures

Reuse the geometry-selected junctions from `doc_river_storage_placement_v1`:
295 whole-network junction corridors and 59 monitored footprints. Retain their
source weights, path lengths, coverage and original representatives. The 295
corridors are selected parts of whole river networks, not all source paths in
each basin. Carry the original two junction exclusions forward. Report the
monitored set separately and average its connections within receiver before
descriptive summaries. Keep the original 22 covariate-selected elongated/broad
pairs; do not rematch on this response.

## Controlled operations

Each branch receives a unit-height Gaussian DOC anomaly with the same duration
and total integrated anomaly. Baseline constant positive branch flows follow
the saved incremental-area shares; equal flows are a sensitivity. Normalize the
weighted complete path mean to one once, using the unchanged real geometry.

Let b be the two independent branch delays, c the common downstream delay and
bbar the flow-weighted branch mean. Source pulse phases are lambda*(b-bbar):

- lambda=-1: source timing compensates branch delays; arrivals at the junction
  coincide;
- lambda=0: simultaneous source inputs; real path differences set arrival spread;
- lambda=1: source timing reinforces branch delays; junction arrival separation
  doubles.

The flow-weighted source phase stays zero. Negative phases are earlier input
centres on an arbitrary event clock, not negative routing delays. This changes
the timing of otherwise identical inputs, while keeping river geometry fixed.

Cross each timing condition with relative shared-channel volume V={1,2},
discharge Q={1,2}, and response fraction kappa={0,0.5}. The controlled shared
mean time is T=c*V/Q. Its routing kernel consists of deterministic translation
(1-kappa)*T plus a conservative exponential residence distribution with mean
kappa*T. This is a constant-flow concentration-response scenario, not a hydraulic
simulation of an evolving flood wave. Q scales both branch flows equally, so
their shares stay fixed. Branch routing delays are held fixed to isolate the
shared-channel intervention; this is not a recomputation of upstream hydraulics
under higher discharge. The integral preserved across settings is concentration
times time. At doubled flow the imposed carbon load doubles: equal concentration
forcing does not mean equal total carbon mass between discharge scenarios.
Within each constant-flow setting the routed carbon anomaly is conserved.

Unlike earlier fixed-mean storage placement, this experiment lets shared mean
time change with volume/discharge. kappa=0 isolates a longer/shorter transit with
no broadening. kappa=0.5 isolates a specified distributed response at the same
mean. Doubling both volume and discharge must leave the kernel unchanged. No
channel depth ratio is converted into a fitted storage coefficient.

Repeat SD={0.075,0.15,0.30} with area shares, and SD=0.15 with equal flows.
Preserve all 33,984 scenario rows, including exact parameter-equivalent controls.

## Field geometry check

For each of the 69 archived fall transects, compute a rectangular area proxy
width*mean observed depth, then average transects equally within reach. Compare
receiver/mainstem area-proxy ratio and discharge ratio with the published
100-m tracer transit ratio. These are different measurements: missing obstructed
depths, cross-section shape and survey support preclude treating the proxy as an
exact wetted area. Salt-tracer flow was collected within three days after chemistry.
Keep all five junctions and all previously reported DOC deviations visible.

## Outputs and interpretation

- Peak, centroid, central 80% duration and integrated anomaly for every scenario.
- Exact timing variance: input_SD^2 + (1+lambda)^2*branch_variance + (kappa*T)^2.
- Source-alignment, distributed-response and joint contrasts with explicit
  within-case denominators. Separate earlier/later peaks from attenuation.
- Equal-junction, within-form and original-pair summaries; HUC4 block bootstrap
  with 5,000 draws for the geometry cohort. These intervals describe the sampled
  geometry ensemble, not measured DOC treatment effects.
- Geometry-selected real examples and bilingual scientific figures. Repeat the
  examples at half the numerical time step; keep ambiguous double maxima.

The conservative calculation preserves integrated anomaly and steady gain.
It addresses carbon-signal timing and peak shape, not positive/negative chemical
production. The field DOC mixture deviations remain separate observations.

## Method references

The [USGS velocity-area definition](https://www.usgs.gov/mission-areas/water-resources/science/streamgaging-basics)
motivates separating channel area from discharge. The
[USACE Clark description](https://www.hec.usace.army.mil/confluence/hmsdocs/hmstrm/transform/clark-unit-hydrograph-model)
separates translation from distributed attenuation for runoff. Here the reused
concentration kernel follows the constant-volume, constant-flow conservative
mixing equation T*dC/dt + C = C_in. These references do not calibrate any of the
real rivers in this experiment.

No prediction model is trained; previous classes, analyses and endpoints remain
unchanged. New code, figures and a short English research decision live in this
versioned directory.
