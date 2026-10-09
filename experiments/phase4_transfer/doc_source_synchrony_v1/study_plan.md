# Is there transferable temporal information in source DOC departures?

The completed source chemistry, longer-history and full-encoder experiments
do not improve the current complete model. Before fitting another branch,
inspect a distinct information source: DOC departures observed at other source
stations in the same or preceding months. The existing static source profiles
aggregate information; they do not establish useful event-time synchrony.

Use only training stations and DOC cells in partitions142/143/144, with the
cached seed42 station-blocked OOF environmental predictions. Load source
ecology from the already saved, source-only input archive. Receiving-station
DOC values are never read, and no model is selected or fitted in this audit.

Compute native source residuals `DOC - expm1(OOF log prediction)`. Report pair
correlation after subtracting each source station's mean residual. Compare
the five nearest ecological source stations with the five farthest, using
the frozen normalized nine-dimensional ecology. Every pair excludes self.
For lags0,1,3, align destination montht with donor montht-lag; preserve calendar
gaps and never wrap the series. Display support for all pairs; correlations
with fewer than12 common observed months or constant series are marked
unavailable rather than counted as zero.

Control: shuffle a donor's residual values within its calendar month across
years, preserving its availability and seasonal pattern (seed42). This is
a diagnostic control and never becomes a predictor input. Aggregate eligible
pair correlations within recipient station, then average recipients within
partition, then give partitions equal weight. Record counts at each level;
pairs/stations/partitions share data, so these are descriptive associations,
not independent replications or evidence of physical transmission.

A coherent real-versus-shuffle advantage motivates a separately designed
source-innovation transfer experiment. If it is absent or support is sparse,
record that finding and prioritize new environmental information. Do not infer
receiving-site prediction gains from these source associations. No earlier
geographical/external model is restarted and no target result tunes the audit.
