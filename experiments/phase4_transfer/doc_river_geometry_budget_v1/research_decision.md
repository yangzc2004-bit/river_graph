# River geometry and the timing of DOC export

## Main finding

River geometry can change the height, width and timing of an incoming DOC
response while leaving its integrated additional carbon unchanged. A lower
concentration peak therefore does not, by itself, demonstrate less carbon
export or DOC removal. Separately routing water and carbon makes this distinction
explicit and connects the previous geometry experiments to measured outlet
concentration and discharge.

This version combines 3,872 controlled scenarios on 121 measured tributary
path pairs with complete carbon budgets for 161 flow-selected outlet events
in Kervidy, Rappbode and Bouleau. Each event has two nested bounded windows,
giving 322 complete budgets. The geometry pairs feed 38 receivers in 17
connected monitoring systems; the outlet events represent three catchments.
These are different evidence units, not 161 independent river forms or matched
field replications of the controlled experiment.

## How structure is connected to the response

The geometry experiment uses actual directed source-to-receiver paths, common
downstream lengths and area-proxy branch flow shares from the existing screened
inventory. Source concentration and incoming water pulses are identical within
each paired manipulation. Water flux and carbon flux pass through the same
causal transport operator before concentration is recovered by division:

```text
real tributary paths and confluence layout
                  ↓
arrival-time distribution and shared downstream spreading
                  ↓
routed water flux Q(t) and carbon flux J(t)
                  ↓
DOC concentration C(t) = J(t) / Q(t)
                  ↓
peak height, peak time, width and integrated export
```

Baseline concentration is 5, source concentration increment is 1, and a common
Gaussian pulse has SD 0.15 or 0.30 in normalized scenario time. Water rises by
one or three times its baseline. Lengths are normalized by the square root of
basin area; no measured velocity converts these times into hours. Gamma
spreading and processing rates are imposed process scenarios. Area shares
approximate relative water contributions and are not measured event discharges.
The two monitored branches have a previously reported median drainage coverage
of 29.3%; these scenarios describe their combination rather than a complete
receiving-basin carbon budget.

### Path differences redistribute the same carbon

With source SD 0.15 and a threefold water increment, actual unequal paths reduce
the receiver-average concentration increment by **0.0414**, compared with equal
arrivals at the same weighted mean path; the connected-system 95% interval is
**[0.0308, 0.0519]**. The mean increment changes from approximately **1.000 to
0.9585**. This difference is about 4.1% of the prescribed *input increment*,
not a 4.1% reduction in measured ambient DOC concentration.

The additional-carbon arrival SD increases by **0.0854 [0.0582, 0.1195]**
scenario time units, but its centroid and integrated mass are unchanged to
numerical resolution. Shifting both water and DOC release at each source to
compensate for the path difference reverses this effect. Thus the signal
depends on the combination of paths and release timing, not a tributary count
alone. The four prespecified pulse duration/water-amplitude combinations are
retained in the full contrast table.

### Shared downstream spreading widens the response

Reallocating fixed total paths between branches and the common segment has
exactly zero effect under deterministic, conservative routing. Confluence
position becomes relevant in this experiment when the shared segment is
assigned a distributed travel-time process.

Under the imposed gamma operator, a longer shared segment changes the mean
concentration increment from **0.8942 to 0.7315**, a difference of
**−0.1626 [−0.1958, −0.1359]**. Additional-carbon arrival SD increases by
**0.1318 [0.0795, 0.1821]**, and concentration half-width increases by
**0.4060 [0.2797, 0.5009]** scenario units. The total additional carbon remains
one relative to its input. Because total mean paths are fixed, the carbon
centroid remains unchanged within grid precision; spreading is not synonymous
with a later mean arrival.

The concentration maximum actually moves **0.0881 [0.0625, 0.1156] scenario
units earlier** in this fixed-total-path gamma comparison. A broader,
right-skewed response can have an earlier maximum while preserving its mean.
Consequently peak delay, distribution centroid and width must be reported
separately; none is a substitute for another.

For these conservative, identical-source-shape scenarios the moment relation is:

```text
variance of additional-carbon arrival time
  = variance of the source pulse
  + weighted variance of (branch travel time + source release offset)
  + variance introduced by the shared downstream travel distribution
```

An independent analytical variance calculation agrees with the numerical
operator. Concentration itself does not obey this simple additive identity:
its changing water denominator must also be considered.

### Total carbon decreases only in the explicit processing sensitivity

Applying a first-order rate of 0.25 to the additional DOC pulse in the shared
segment reduces retained additional carbon by **0.1086 [0.0827, 0.1330]**,
while water is conserved. The mean retained fraction becomes **0.8914**.
This is an imposed reactive-pulse sensitivity; baseline DOC is conserved.
It is not a measured 10.9% river removal rate. It demonstrates the different
signature of processing: a lower peak accompanied by a smaller cumulative
carbon plateau. Conservative spreading lowers a peak without that plateau loss.

## What measured DOC and flow show

The observed analysis retains the preceding hourly 20% prominence event rules.
All timing-eligible events are inventoried, including nonpositive DOC responses.
Only exact, complete paired hourly records receive integrated budgets. One
Kervidy event lacks one of 39 required hourly pairs and remains without a
complete budget; no interpolation fills it.

DOC in mg/L multiplied by discharge in m3/s gives carbon flux in g/s.
Adjacent recorded endpoints are integrated to kg C. The two windows are the
flow-defined start to flow return, and the same start to the saved DOC response
end. The latter can be shortened by another flow event and is not guaranteed
to contain the complete DOC recovery. Both are **bounded-window exports**.

### Concentration timing differs from carbon-flux timing

Among resolved positive DOC responses with complete budgets:

| Outlet | Events | Median concentration peak minus flow peak | Median carbon-flux peak minus flow peak, 95% month-block interval | Median bounded carbon yield, kg/km2 |
|---|---:|---:|---:|---:|
| Kervidy | 58 | 2 h | 1 h [0, 1] | 34.15 |
| Rappbode | 62 | 3 h | 1 h [0, 1] | 28.38 |
| Bouleau | 8 | 22 h | 9 h [0, 31] | 37.66 |

The table reports separate medians, not median paired differences. Increasing
DOC concentration late in an event does not necessarily postpone the bulk of
carbon export, because discharge may already be declining. In the same
response windows, carbon minus water fraction exported after the flow peak is:

- Kervidy: **−0.30 percentage points [−0.87, −0.10]**.
- Rappbode: **+1.00 percentage points [+0.46, +1.32]**.
- Bouleau: **+2.04 percentage points [+0.74, +4.55]**, from only eight responses.

These are small and nonuniform shifts in export allocation despite the
concentration-peak delays. The corresponding all-timing-eligible results,
including nonpositive DOC responses, use 62, 82 and 17 complete events and are
reported alongside the positive-response subset.

### Peak height alone does not rank event export

Exploratory within-catchment Spearman associations between peak concentration
increment and bounded carbon yield are **−0.073 [−0.336, +0.258]** at Kervidy
and **+0.090 [−0.194, +0.366]** at Rappbode. Their raw associations do not
establish that a higher concentration peak gives a larger event export.
After rank adjustment for measured runoff volume and response-window duration,
the associations become **+0.732 [+0.537, +0.811]** and
**+0.695 [+0.515, +0.782]**. These conditional associations include the
mathematical relationship between concentration and integrated export; they
are not estimates of river geometry or ecological causation.

Higher peaks can also be wider in observations: peak increment versus DOC
half-width has rank association **+0.403 [+0.110, +0.676]** at Kervidy and
**+0.411 [+0.046, +0.654]** at Rappbode. Field inputs are changing between
events, whereas the geometry manipulation fixes input. The positive field
association therefore cannot be used to infer that changing channel paths
raises both peak and width. Bouleau's small samples remain descriptive and do
not receive these association intervals.

Rappbode signals were already smoothed with a 2.5-hour moving average and had
short gaps interpolated by the provider. Bouleau uses only observed/calibrated
DOC, excluding RF-generated gap-fill concentrations. Only Kervidy currently
has validated rooted river vectors among these three cases. Consequently,
the outlet comparison does not quantitatively rank elongated versus
tributary-rich morphology.

## Scientific conclusion and next field test

The useful structural proposition is now specific:

> Path diversity and shared downstream routing determine how an incoming
> carbon pulse is distributed in time. Water-weighted mixing determines its
> concentration peak; actual processing can additionally change its total.

This explains what a structure-aware DOC model should represent: an explicit
distribution of source arrival times and water contributions, with a separate
process term if there is evidence for carbon loss. A generic same-month graph
message or a fixed high/low DOC label for a whole river form does not encode
these distinctions.

The next empirical test needs simultaneous branch and receiving DOC/Q during
the same events, accompanied by quantitative rooted geometry. It would check
whether measured incoming loads combined with path dispersion predict outlet
peak, timing and width, and whether a carbon-budget discrepancy remains after
accounting for other tributaries and lateral sources. Complete upstream and
downstream budgets are necessary before estimating removal. Current outlet-only
budgets cannot supply that missing input. River geometry remains the explanatory
target; this step does not replace it with land-cover differences or retrain
the reconstruction model.

## Reproduction and verification

```bash
uv run python scripts/analyze_doc_river_geometry_budget_v1.py --bootstrap-draws 5000
uv run python scripts/verify_doc_river_geometry_budget_v1.py
uv run python scripts/plot_doc_river_geometry_budget_v1.py
uv run python scripts/plot_doc_river_geometry_budget_v1.py --chinese
```

Conservative routing preserves all added carbon and water to floating-point
precision. Independent endpoint sums reproduce all 322 field budgets. The
deterministic junction null is checked across all 484 path/forcing combinations;
24 geometry-selected distributed-kernel cases are checked at half the time step.
The maximum concentration-increment resolution difference is 0.00031, well below
the displayed structural differences. The first internal fractional-deposition
artifact and its correction are recorded in `numerical_notes.md`.

The figures, event inventories, full forcing contrasts, analysis sources and
field-window limitations are preserved together. Existing results are not
overwritten, and no training automation is restarted.

Source archives and processing methods:

- [Rappbode public archive](https://www.hydroshare.org/resource/9be43573ba754ec1b3650ce233fc99de/).
- [Werner et al. 2019, Rappbode DOC and discharge methods](https://bg.copernicus.org/articles/16/4497/2019/).
- [Bouleau 2018 archive](https://doi.pangaea.de/10.1594/PANGAEA.959043) and
  [2019 archive](https://doi.pangaea.de/10.1594/PANGAEA.959044).
- [Prijac et al. 2023, Bouleau calibration and flow methods](https://hess.copernicus.org/articles/27/3935/2023/).
- Kervidy source receipts and corrected DOC/discharge records remain in the
  previous `doc_river_kervidy_geometry_v1` and `doc_river_multicatchment_pulses_v1`
  directories; real tributary path lineage remains in `doc_river_routing_mechanisms_v1`.
