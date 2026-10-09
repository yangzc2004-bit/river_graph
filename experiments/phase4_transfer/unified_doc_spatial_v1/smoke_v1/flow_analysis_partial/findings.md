# DOC reconstruction benefit, hydrology and record availability

Descriptive analysis of saved predictions; no model is refitted.

**Implementation/interim check only:** the full nine-fit confirmation batch is incomplete.

Monthly discharge is input channel 1, verified against dataset metadata and `dataset.FEATURE_CHANNELS`.
The descriptor is population standard deviation divided by mean discharge over observed hydro months.
It is invariant to multiplicative flow-unit conversion. A station needs at least 12 observed months
and positive mean flow; other stations remain in an explicit insufficient-data group.
Signed discharge is retained: the source quality rules allow reverse flow in tidal/backwater reaches.
Each partition's training-station CV tertiles define lower, middle and higher variability.
No DOC labels or prediction errors determine these strata. The full historical hydro record is used
as a station descriptor, consistently with retrospective reconstruction; this is not event timing.

Record availability is the total number of observed DOC months from y_mask, before reserving support.
Training-station observation-count tertiles define lower, middle and higher record availability.
This descriptor uses availability indicators only, never concentration values. The model still
hides all target-station DOC from base inputs and gives every station exactly the same K support
budget. Lower record availability therefore describes the monitored record, not fewer input
support labels or a directly observed target temporal history.

Station responses first average the existing seed losses. Stratum MAEs weight each station by
its fixed query count within a partition, then average nonempty partitions equally. Empty strata
retain zero counts and undefined error, rather than being assigned zero error. Repeated stations
are counted once per partition and their unique count is also reported. Associations are descriptive;
they do not establish that flow variability causes the model's gain.

| Comparison | Variable | Stratum | Reference MAE | Candidate MAE | Reduction (%) | Partitions | Station cases / unique | Query cases |
|---|---|---|---:|---:|---:|---:|---:|---:|
| hybrid_calibration_k5_vs_k0 | flow_variability | higher | 3.5999 | 2.8956 | 19.56 | 1 | 18 / 18 | 835 |
| hybrid_calibration_k5_vs_k0 | flow_variability | insufficient | 1.2081 | 1.0780 | 10.77 | 1 | 11 / 11 | 610 |
| hybrid_calibration_k5_vs_k0 | flow_variability | lower | 1.6825 | 1.2709 | 24.46 | 1 | 22 / 22 | 1247 |
| hybrid_calibration_k5_vs_k0 | flow_variability | middle | 1.7734 | 1.3287 | 25.08 | 1 | 20 / 20 | 1283 |
| hybrid_calibration_k5_vs_k0 | record_availability | higher | 2.0244 | 1.5108 | 25.37 | 1 | 26 / 26 | 2874 |
| hybrid_calibration_k5_vs_k0 | record_availability | lower | 2.2748 | 1.9492 | 14.31 | 1 | 27 / 27 | 517 |
| hybrid_calibration_k5_vs_k0 | record_availability | middle | 1.9212 | 1.7383 | 9.52 | 1 | 18 / 18 | 584 |
| hybrid_vs_et_k5_calibrated | flow_variability | higher | 2.8679 | 2.8956 | -0.96 | 1 | 18 / 18 | 835 |
| hybrid_vs_et_k5_calibrated | flow_variability | insufficient | 1.1000 | 1.0780 | 2.00 | 1 | 11 / 11 | 610 |
| hybrid_vs_et_k5_calibrated | flow_variability | lower | 1.2785 | 1.2709 | 0.60 | 1 | 22 / 22 | 1247 |
| hybrid_vs_et_k5_calibrated | flow_variability | middle | 1.3358 | 1.3287 | 0.53 | 1 | 20 / 20 | 1283 |
| hybrid_vs_et_k5_calibrated | record_availability | higher | 1.5151 | 1.5108 | 0.28 | 1 | 26 / 26 | 2874 |
| hybrid_vs_et_k5_calibrated | record_availability | lower | 1.9541 | 1.9492 | 0.25 | 1 | 27 / 27 | 517 |
| hybrid_vs_et_k5_calibrated | record_availability | middle | 1.7281 | 1.7383 | -0.59 | 1 | 18 / 18 | 584 |
