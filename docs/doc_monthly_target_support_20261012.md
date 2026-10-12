# Sampling support of the frozen monthly DOC target

The 22,571 frozen labels are means of accepted available DOC sample records within calendar months. They are not continuously monitored monthly concentration means. The cached raw data reproduce every label to float32 rounding precision (maximum absolute difference 0.000003052 mg/L). Labels and experiment masks remain unchanged.

- 17,117 observed months (75.84%) contain one accepted sample record.
- 17,403 observed months (77.10%) contain samples on only one distinct day.
- Median sample records per observed month: 1; median distinct sample days: 1; median first-to-last sample-day span: 0 days.
- Across the five spatial query regions, 71.91% of scored months have one sample day.
- Across the three temporal scored cores, 84.12% have one sample day. The same cores are reused under wider surrounding gaps.

There are 147 repeated date/value records. These cannot be classified as duplicate physical samples without sample identifiers. They are reported, not removed. High frozen DOC concentrations are also retained; correct units and parameter codes do not independently establish physical measurement validity.

The distinction matters when aligning daily-discharge variability with the concentration target: a calendar-month flow descriptor summarizes days beyond the actual concentration sampling day. Sampling variability or event timing may contribute to a weak monthly relationship, but these counts do not prove either mechanism. This remains retrospective reconstruction of an available-sample monthly target.

For paper wording, use *monthly observed-sample DOC reconstruction* for the validated target. A complete model grid is an estimate of that target; accurate true monthly means, daily event trajectories and arbitrary river-reach values need separate observations and validation. A future sampling-date-aligned study should retain the current endpoint version and explicitly define its new target and masks.

Evidence: [sampling summary](../experiments/phase4_transfer/doc_dynamic_reconstruction_v1/sampling_audit/summary.csv), [cell-level support](../experiments/phase4_transfer/doc_dynamic_reconstruction_v1/sampling_audit/observed_month_sampling.csv), [raw-input provenance](../experiments/phase4_transfer/doc_dynamic_reconstruction_v1/sampling_audit/provenance.json).
