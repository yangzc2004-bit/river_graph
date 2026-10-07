# Field responses connect river transport to DOC processing

## Main result

Three public co-releases of salt and labelled DOC supply an observed link in the
river-form mechanism. The conservative pulse reaches the downstream sampling
location later and has a longer central duration in every addition. Labelled
carbon has a further response change beyond the salt reference: it declines
strongly in the two glucose additions, while the leaf-leachate addition has no
corresponding downstream decline in its tracer-normalized core response.

The scientific direction is **river form → travel and mixing opportunities → DOC
response**. Structure and carbon processing need separate observables in this
chain. The original three whole-network forms remain the structural comparison;
this one-stream experiment supplies its transport/processing link.

## 1. All three original additions and both sampling locations

The Blaine Creek archive contains two glucose additions (8 and 15 August 2019)
and one leaf-leachate addition (9 August). The analysis retains all three dates,
433 original records, 210 scheduled laboratory rows and 205 rows with jointly
measured DOC and carbon isotope values. Release-referenced times are reconstructed
from the original date/time fields. Dense conductivity-only records do not count
as laboratory carbon samples.

Excess labelled DOC is derived from measured total DOC and isotope atom-fraction
excess. Salt conductivity provides the conservative carbon reference using the
authors' archived salt/carbon calibration. The primary core fraction is the
origin-constrained concentration slope at co-observed points where that reference
exceeds 25% of its local peak, within the recorded 0–300 minute window.

| Addition | Salt centroid delay (min) | Salt central duration change | Upstream DOC/reference core fraction | Downstream DOC/reference core fraction | Downstream/upstream fraction |
|---|---:|---:|---:|---:|---:|
|8 August, glucose|60.25|+9.26%|0.508|0.135|0.265|
|15 August, glucose|61.17|+16.34%|0.432|0.155|0.358|
|9 August, leaf leachate|67.42|+31.70%|0.567|0.684|1.206|

Timing and duration are bounded-window diagnostics, integrated only between
adjacent finite records no more than 30 minutes apart. Twenty- and 45-minute
limits are displayed as sensitivities. The leachate downstream curve has not
returned fully to background at the window boundary; its last anomaly remains
16.4% of the observed peak.

## 2. What the paired carbon response means

Glucose core responses downstream are about 27% and 36% of their upstream
tracer-normalized responses. This corresponds to approximately 73% and 64%
additional relative concentration-response decline. **These percentages do not
estimate total DOC mass removal.** They describe labelled-carbon concentrations
relative to a simultaneously released conservative reference at actual sampled
clocks.

The common conductivity–carbon coefficient cancels within each downstream/upstream
ratio. Absolute core fractions still inherit that coefficient and the background
policy. All six absolute fractions are below one under the archived calibration.
The extra relative glucose decline is consistent with carbon uptake/retention
beyond conservative transport; this reanalysis does not partition uptake into
respiration, storage, exchange or other fates.

The leaf-leachate ratio above one indicates no matching further decline in this
pair. It does not establish carbon production. The material comparison also
changes date and field conditions, so it does not isolate DOC composition as the
sole driver. The useful result is that the same stream segment does not impose
one fixed carbon attenuation for every release.

## 3. Original measurements and author processing agree on the main direction

Mean/median backgrounds and 10%, 25% and 50% core thresholds retain the paired
direction: both glucose downstream/upstream fractions are below one; the leachate
fraction is above one. At the primary threshold, author-processed ratios are
approximately 0.267, 0.303 and 1.144, respectively. Processing changes the magnitude,
especially for the second glucose release, without changing that pattern.

The original-point analysis keeps missing laboratory values missing. The author
intermediate contains three filled upstream leachate clocks and one downstream
second-glucose carbon value without an original joint laboratory measurement.
Manual conductivity substitutions, drift corrections and some filtered rows are
recorded separately. These processed values are a sensitivity reference rather
than additional measured replication.

One duplicate second-glucose conductivity clock contains conflicting readings
and remains unresolved. The catalogue identifies the locations at 41/61 m,
whereas the leachate spatial-survey labels use 50/75.5 m. We preserve that distance
conflict and do not calculate transport velocity or uptake per metre. Clock-based
and paired concentration-response quantities do not require that discrepancy to
be resolved.

## 4. Connection to the original river forms

The existing geometry experiments separate incoming-path arrival dispersion
from spreading in the shared downstream channel. The real tracer record now
shows why a longer mean travel time and a wider response must be measured
separately, and why conservative transport alone does not specify the observed
DOC response.

Keep three levels linked:

1. **Whole-network form:** the original elongated, sparse/mainstem and
   broad-branching classes, with their real mapped paths.
2. **Structural operations:** branch arrival separation, junction placement,
   shared-path length and local channel geometry.
3. **Observed outcomes:** tracer timing/spreading and DOC departure from the
   conservative reference.

The field segment does not compare the three classes. It informs how to measure
their proposed process link, while the earlier identical-input experiments
continue to isolate geometry. This keeps river structure at the centre of the
research instead of turning the project into a source-region comparison.

## Next empirical step: actual confluence geometry and mixing length

Return to multiple real junctions and evaluate geometry against an observed mixing
response before adding another response kernel. A newly located primary study,
[Meem et al. (2025)](https://doi.org/10.1029/2025GL114640), evaluates 150 aerial-image
mixing events at 43 confluences and points to Illinois Data Bank archive
[10.13012/B2IDB-5324086_V1](https://doi.org/10.13012/B2IDB-5324086_V1).
The immediate data task is to retrieve its event tables and coordinates, audit
the observational support, and recover real junction angles, tributary-width
balance and downstream curvature where imagery supports them. Event tables and
coordinates have not yet been replayed in this version.

That dataset measures lateral spatial mixing, whereas the Blaine curves measure
time responses and labelled carbon. Use each for its measured operation. Test
whether junction geometry explains mixing distance, keep repeated events at the
same junction grouped, and then connect the geometry result back to the original
whole-network forms. A co-measured conservative/DOC confluence dataset remains
the target for completing the chemistry link across structures.

The White Clay Creek DOC/bromide archive remains an access lead: its portal
redirects to login and its metadata API currently returns an authorization error.
No author contact or restricted-data retrieval was attempted.

## Outputs and reproduction

- [Original response curves](figures/conservative_and_doc_responses.png),
  [Chinese](figures/conservative_and_doc_responses_cn.png).
- [Transport/carbon decomposition](figures/transport_and_carbon_decomposition.png),
  [Chinese](figures/transport_and_carbon_decomposition_cn.png).
- Six series inventories, 18 integration-gap summaries, three paired contrasts,
  54 processing sensitivities and source-clock processing comparisons.
- Original records and derived response points retain source record IDs.

```bash
uv run python scripts/fetch_doc_river_tracer_process_v1.py
uv run python scripts/analyze_doc_river_tracer_process_v1.py
uv run python scripts/plot_doc_river_tracer_process_v1.py
uv run python scripts/plot_doc_river_tracer_process_v1.py --chinese
uv run python scripts/verify_doc_river_tracer_process_v1.py
```

Data: [Plont et al. (2025), CC BY 4.0](https://www.hydroshare.org/resource/988f0d0aa46249b2b654145cf5fbf895/).
Associated study: [Hall et al. (2026)](https://doi.org/10.1007/s10021-025-01030-2),
published online in December 2025. Author processing is pinned at commit
`37bd0ea10c3b328b5bc8c154e120d3651381429e` in the
[public repository](https://github.com/robohall/DOC_uptake). This version does not
reproduce the authors' respiration model. No prediction model was trained.
