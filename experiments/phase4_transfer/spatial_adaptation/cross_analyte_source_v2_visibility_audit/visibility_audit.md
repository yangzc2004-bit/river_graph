# Cross-analyte source-selection visibility audit

## Conclusion

The exploratory v1 result is not a strict spatial-zero-shot result. Its
`analyte_profile()` function loads the complete pH and specific-conductance
arrays and computes station summaries from every available month. It receives
no split or visibility mask. For the outer E3 target stations, the profile
therefore consumed 2,521 pH labels and 2,511 conductance labels in cells that
also belong to the hidden DOC test role. Those cells represent 39.6% and
35.2% of the target-station pH and conductance profile cells, respectively.
The same defect is present in the internal spatial validation split: 1,330 pH
and 1,284 conductance labels overlap the hidden validation role.

This is a target-station auxiliary-analyte label leakage. It does not alter
the DOC training labels, but it means the v1 source-station selection used
future/hidden pH and conductance measurements from the target station.

## Strict v2 rerun

`run_spatial_cross_analyte_source_v2.py` keeps the old v1 directory unchanged.
For each internal and outer split it masks every pH/EC label on the DOC target
stations before computing the station profile. The descriptor scaler is also
fit only on source-station rows. Hydro, ecology, static and graph descriptors
remain label-free.

The strict rerun selects `k=160` on internal validation. Outer E3 MAE is
2.735 (mean over seeds 42--44), compared with 2.508 for the old v1 run at the
same k. The selected-k strict result is therefore about 9.0% worse than v1.
This drop is expected: v1's target pH/EC profiles carried substantial local
information that is unavailable under zero-shot spatial transfer.

The strict v2 result should be used for a zero-shot claim. The v1 result may
remain as a separate auxiliary-analyte-informed diagnostic only if its task is
explicitly described as allowing full target-station pH/EC profiles, and it
must not be presented as a future-label-free spatial transfer result.

## Reproducible files

* `metrics.csv` — strict v2 internal and outer metrics.
* `leakage_audit.json` — exact v1 overlap counts for pH and conductance.
* `strict_profile_audit.json` — masked profile row/cell counts.
* `spec.json` — datasets, hashes, split and visibility definition.
* `summary.md` — full rerun table.

