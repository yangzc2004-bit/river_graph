# Independent river outlets: DOC pulse timing and spreading

## Research result

The investigation now includes three real catchments rather than one. Under a
common hourly-observation protocol, resolved positive DOC responses have median
half-excess widths **1.44–1.56 times their flow-pulse widths**. Peak delays are
much less uniform: **2 hours at Kervidy, 3 hours at Rappbode and 22 hours at
Bouleau**. Bouleau contributes only eight resolved responses and its lag interval
reaches zero. These observations motivate a study of **how geometric routing
modifies an event**, rather than a fixed concentration or delay label for each
river shape.

### Common-hourly primary results

| Outlet | Positive DOC responses | Complete width pairs | DOC minus flow peak, median [95% CI], h | DOC/flow half-excess width, median [95% CI] | Later DOC peaks | Wider DOC peaks |
|---|---:|---:|---:|---:|---:|---:|
| Kervidy, France | 58 | 55 | 2.00 [2.00, 2.00] | 1.556 [1.400, 1.714] | 54/58 | 50/55 |
| Rappbode, Germany | 62 | 51 | 3.00 [3.00, 4.00] | 1.444 [1.300, 1.625] | 53/62 | 46/51 |
| Bouleau, Canada | 8 | 6 | 22.00 [0.00, 34.00] | 1.533 [1.216, 1.818] | 6/8 | 5/6 |

Intervals use 5,000 calendar-month-block draws within each catchment, seed 42.
Lag samples occupy 24, 28 and 6 month blocks; width samples occupy 24, 27 and 5.
The degenerate hourly Kervidy median interval reflects quantized observations
and repeated bootstrap medians, not a physically exact two-hour transport time.
Catchments are not matched on climate, source composition, forcing or sensor
processing. No causal river-form effect is estimated from this table.

## What the real maps establish

The original maps show the geometric distinctions requested in the research
question. They are not AI-generated illustrations or land-cover-based classes.

- **Kervidy:** four mapped headwater routes merging towards the outlet. The
  independently acquired BD Topage vectors support measured paths: 1.960–2.207 km,
  length CV 4.2%, and a 569 m common terminal route.
- **Rappbode:** multiple lateral tributaries join an extended main channel in
  Werner et al.'s original catchment map. This is a visibly tributary-rich case.
- **Bouleau:** the published map depicts a dominant winding drainage channel
  through an elongated catchment. Isolated peatland pools are not counted as
  connected tributaries.

The latter two maps are qualitative evidence of mapped planform. The accessible
archives do not provide the comparable rooted vector networks needed for measured
headwater-path dispersion or shared-terminal lengths. They are not assigned to
the ST357 numerical clustering classes by visual judgement. A map depicting one
dominant channel does not certify the absence of every small tributary.

Published areas used for specific-discharge conversion are 4.8896 km² (Kervidy
provider polygon), 2.58 km² (Rappbode) and 2.22 km² (Bouleau).

## Same mapped form, different responses

Period summaries are retained for all years, including the zero-eligible-event
Rappbode year 2020. Two differences matter for the morphology question:

- Bouleau's median lag is **34 h in 2018 (five responses)** and **0 h in 2019
  (three responses)**. Rappbode's hourly median is **−1 h in 2019 (eight)** and
  **3–3.5 h in the other years with usable responses**.
- Among Bouleau's two resolved 3–24 h flow events, the lag median is −4 h;
  among its six >48–96 h flow events, it is 32.5 h. The middle duration band has
  no resolved positive responses. One long 2019 event has zero lag, so event
  duration does not by itself explain all the variation.

These small-subset duration comparisons were added as exploratory diagnostics
after observing the catchment summaries. They do not replace the primary table
or establish a duration effect. The same mapped river form accompanies different
timing outcomes. Therefore a between-river lag difference cannot simply be called
a difference caused by branching. The real DOC signals can also have several
local maxima within one single qualifying flow pulse; the protocol measures the
largest recorded DOC maximum and its local half-excess brackets, not a fitted
single-peak waveform.

## Cadence, thresholds and source processing

Native-resolution medians are:

| Outlet | Native interval | Resolved DOC / complete widths | Lag, h | Width ratio |
|---|---:|---:|---:|---:|
| Kervidy | 15 min | 60 / 56 | 1.75 | 1.642 |
| Rappbode | 15 min | 75 / 63 | 3.25 | 1.464 |
| Bouleau | 1 h | 8 / 6 | 22.00 | 1.533 |

Hourly signals use existing whole-hour rows, not averaged or interpolated
resamples. The new absolute flow prominence of 0.35 mm/day corresponds to
0.019807, 0.010451 and 0.008993 m³/s respectively. Relative prominence 15%, 20%
and 25% comparisons are all available in `analysis/pulse_comparison.csv`.
For Rappbode, adding more qualifying flow peaks can shorten follow-up windows,
so the lower threshold need not monotonically increase resolved DOC counts.

The previous Kervidy study remains intact. Its 0.020 m³/s absolute threshold
still gives **59 resolved responses and 55 width pairs**. The slight denominator
change here is from a new explicitly area-normalized protocol, not overwritten
history. A direct in-memory comparison confirmed the old operators reproduce
the previous event identities, eligibility flags, lags and width ratios.

Source distinctions:

1. **Kervidy:** optical DOC with the archived affine correction; raw discharge.
   No new temporal filtering. Original clocks are UTC.
2. **Rappbode:** laboratory-calibrated UV-VIS DOC; provider drift and outlier
   correction, spline interpolation of gaps below two hours, and a **2.5-hour
   moving average of flow and water quality**. Event outcomes describe these
   processed signals. Source row flags do not identify individual filled points
   or moving-average alignment. The archive's unspecified time offset is not
   invented. There are **299 extra timestamp rows and 292 conflicting DOC-clock
   fields**. Equal flow values are collapsed, conflicting DOC values are masked,
   and two negative DOC values are excluded. Raw rows remain preserved.
3. **Bouleau:** temperature-corrected fluorescence calibrated with laboratory
   DOC. Only `DOC [mg/l]` is used; **all 5,539 nonmissing entries in the separate
   RF-predicted DOC column are excluded from measurement input**. Three extra
   identical rows at one 2019 clock are collapsed. The archive has 4,056 valid
   canonical observed/calibrated DOC records; three reported zeros are retained.
   Flow in m³/h is converted to m³/s. The native Eastern-labelled clock is kept
   without assigning an undocumented offset. Growing-season hourly flow is
   distinguished from the paper's separate daily PHIM-modelled discharge.

The primary hourly inventories retain 101, 245 and 33 flow candidates, with
63, 82 and 17 dense uncensored assessment windows. Positive response counts
must be read alongside unresolved, boundary-peak and nonpositive flags; those
flags are not mutually exclusive. Recovery remains censored for 20, 35 and 7
resolved responses. No common uncensored recovery-time median is claimed.

## Scientific decision and next experiment

**Keep river geometry as the research object, but use event spreading as the
main bridge between geometry and DOC dynamics.** The independent cases support
the occurrence of broader DOC outlet responses. They do not establish that the
tributary-rich map has broader DOC than the long-channel map: width estimates
are close, and a comparison of three differently processed catchments cannot
identify a general geometric effect.

The next structural test has a concrete design:

1. Acquire comparable rooted river vectors for the mapped Rappbode and Bouleau
   outlet domains. Measure headwater-path dispersion, confluence positions and
   common terminal distance with the existing rooted-path analysis, retaining
   the distinction between stream paths and hillslope source paths.
2. Compare events with similar flow duration and amplitude, not all storms
   pooled as if they had the same input. The available Bouleau short-event
   sample is too small for a stable matching analysis, so expand the independent
   catchment/event sample on data availability before fitting form effects.
3. Link those descriptors to spreading predictions from the already completed
   controlled routing experiments. Prediction of an observed outlet response
   requires the incoming DOC signals or an explicitly estimated mobilisation
   component. Do not substitute DOC–flow peak lag for a directly measured
   tributary-to-outlet travel time.
4. Prioritize a simultaneous upstream/downstream DOC event archive to separate
   arrival-time differences from post-confluence spreading. This closes the
   specific geometry-to-response link that outlet-only records leave unresolved.

The intended conclusion remains: **tributary path diversity and shared
downstream routing can change how an incoming DOC pulse is distributed in time**.
The present field results are evidence about outlet behaviour, while the
previous controlled routing experiments supply the isolated structural tests.
Neither is relabelled as the other. No new neural model was trained, no land-cover
class was substituted for network form, and the existing DOC manuscript/model
results were not overwritten.

## Sources and reproducibility

- [Musolff (2024), Rappbode archive](https://www.hydroshare.org/resource/9be43573ba754ec1b3650ce233fc99de/)
- [Werner et al. (2019), original Rappbode map and catchment description](https://bg.copernicus.org/articles/16/4497/2019/)
- [Prijac (2023), Bouleau 2018 hourly archive](https://doi.pangaea.de/10.1594/PANGAEA.959043)
- [Prijac (2023), Bouleau 2019 hourly archive](https://doi.pangaea.de/10.1594/PANGAEA.959044)
- [Prijac et al. (2023), original map, calibration and flow methods](https://hess.copernicus.org/articles/27/3935/2023/)
- Kervidy inputs and BD Topage geometry: the preserved `doc_river_kervidy_geometry_v1`
  and `doc_river_kervidy_pulses_v1` source manifests.

Public data/papers are attributed under their stated licences. The Rappbode map
retains the original base-map attribution **© Google, GeoBasis-DE-BKG**; the
map-evidence composite is a research comparison, not a newly surveyed vector map.
The unavailable Vermont NEWRnet and Wood Brook leads are retained in the source
inventory rather than silently replaced by their DOC-free companion products.

Reproduction order (all via the repository's uv environment):

```bash
uv run python scripts/fetch_doc_river_multicatchment_pulses_v1.py
uv run python scripts/analyze_doc_river_multicatchment_pulses_v1.py
uv run python scripts/plot_doc_river_multicatchment_pulses_v1.py
uv run python scripts/plot_doc_river_multicatchment_pulses_v1.py --chinese
```

Raw downloads are locally retained under the gitignored data directory. Source
hashes, provider checksums, complete event inventories, summary CSVs and figure
receipts are saved in this independent versioned experiment directory.
