# DOC reconstruction benefit, hydrology and record availability

Descriptive analysis of saved predictions; no model is refitted.

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
| hybrid_calibration_k5_vs_k0 | flow_variability | higher | 4.3264 | 3.5466 | 18.02 | 3 | 59 / 46 | 2173 |
| hybrid_calibration_k5_vs_k0 | flow_variability | insufficient | 1.2515 | 0.9610 | 23.21 | 3 | 29 / 23 | 1860 |
| hybrid_calibration_k5_vs_k0 | flow_variability | lower | 1.4892 | 1.1670 | 21.63 | 3 | 66 / 55 | 5506 |
| hybrid_calibration_k5_vs_k0 | flow_variability | middle | 1.8251 | 1.5673 | 14.13 | 3 | 59 / 48 | 3326 |
| hybrid_calibration_k5_vs_k0 | record_availability | higher | 1.7107 | 1.3557 | 20.75 | 3 | 72 / 59 | 9307 |
| hybrid_calibration_k5_vs_k0 | record_availability | lower | 3.2106 | 2.6036 | 18.91 | 3 | 71 / 62 | 1291 |
| hybrid_calibration_k5_vs_k0 | record_availability | middle | 2.6148 | 2.2539 | 13.80 | 3 | 70 / 52 | 2267 |
| hybrid_vs_et_k5_calibrated | flow_variability | higher | 3.4980 | 3.5466 | -1.39 | 3 | 59 / 46 | 2173 |
| hybrid_vs_et_k5_calibrated | flow_variability | insufficient | 0.9614 | 0.9610 | 0.04 | 3 | 29 / 23 | 1860 |
| hybrid_vs_et_k5_calibrated | flow_variability | lower | 1.1563 | 1.1670 | -0.93 | 3 | 66 / 55 | 5506 |
| hybrid_vs_et_k5_calibrated | flow_variability | middle | 1.5405 | 1.5673 | -1.74 | 3 | 59 / 48 | 3326 |
| hybrid_vs_et_k5_calibrated | record_availability | higher | 1.3432 | 1.3557 | -0.93 | 3 | 72 / 59 | 9307 |
| hybrid_vs_et_k5_calibrated | record_availability | lower | 2.6247 | 2.6036 | 0.80 | 3 | 71 / 62 | 1291 |
| hybrid_vs_et_k5_calibrated | record_availability | middle | 2.1976 | 2.2539 | -2.56 | 3 | 70 / 52 | 2267 |
