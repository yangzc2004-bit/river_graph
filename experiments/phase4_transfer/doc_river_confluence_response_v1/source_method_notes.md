# Measurement scope and source notes

## Krycklan

Eleven SITES objects supply half-hourly discharge on a fixed UTC+1 clock. File
preambles describe stage-based conversion using weirs or rating curves. These
are not optical-DOC estimates. Match their actual recorded times; the chemistry
collection supplies separate laboratory observations and cannot provide the
missing half-hour carbon curves. The original twenty seasonal windows and four
mapped configurations are retained even when a branch gauge is absent.

## Tom's Creek

The public HydroShare archive was last updated in November 2022 and supplies
preprocessed chemistry means and survey points. The
[published methods](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2022WR034224)
describe laboratory DOC, conservative-salt pulse estimates of discharge and
velocity, and October width/depth surveys. Flow measurements occurred within
three days after chemistry sampling. They represent campaign conditions, not
synchronized event carbon/flow trajectories. Downstream sampling transects were
selected using lateral conductivity uniformity. Consequently conductivity is a
mixing diagnostic within that selected transect, not a new independent test of
the location of complete mixing.

The archived downstream values are three lateral-position means. Upstream
columns sometimes repeat the same mean, while others contain three distinct
values. Their distinct-value counts are retained; the repetitions do not increase
the number of confluences. Our equal lateral mean is explicitly distinct from a
measured flux-weighted cross-section concentration.

One summer Con-1 campaign has inconsistent supplied weight normalization. Our
reference weights use `Q_main / (Q_main + Q_trib)` consistently; every supplied
row's mixture and weight sum are saved for comparison. This is a newly defined
reanalysis, not a claim to execute the archived author R script unchanged. That
script also refers to columns not present under identical names in the CSV.

The channel survey contains 31 missing depths annotated as rock/tree obstructions.
They remain missing. Con-1 contains separate channel/pool survey IDs; their widths
are not silently merged. One Con-1 downstream survey ID has distances of 10 and
15 m in different depth rows. Its distance is recorded as unresolved with both
bounds, while its unchanged width remains usable. Width CV replays the author
summary for Con-2 through Con-5; Con-1 differs with this explicit survey-ID grain
and is reported in the comparison table. No raw source correction is made.

The author residence times are normalized to 100 m, as defined by the archived
supplement. They are transit-duration summaries, not evidence of a separately
identified storage compartment or DOC removal rate. All five confluences occur
along the same mainstem; they are not five independent river basins.

## Public sources

- [SITES stream-water balance collection](https://meta.fieldsites.se/collections/2M74g6oPabfeXXoOkaiGbFs2/Svartberget%20-%20Krycklan%20stream%20water%20balance.json).
- [SLU-linked actual Krycklan map](https://ttiwarir.github.io/krycklan-map/).
- [Plont data, code and supplement; CC BY 4.0](https://www.hydroshare.org/resource/c5e687fa040e4707ba922002bafd18fd/).
- Other candidate archives and their role are recorded in `public_observation_inventory.csv`.
