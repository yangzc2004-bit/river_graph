# Current donor availability in sparse DOC transfer

## Question

Earlier source timing comparisons restricted both current and historical arms
to donors having observations in both periods. That was appropriate for the
timing contrast, but it can suppress a genuine current source observation merely
because an older same-season record is absent. Test whether the existing model
benefits from all permitted contemporaneous source DOC experience.

## Fixed source study

Use142/143/144 source training/validation roles and42/43/44 training seeds.
Carry the complete full-source-level study, anomaly-only attention, retained
complete model and strong trees unchanged. Do not use scored geographical or
external labels to select this version. This is one availability change, not
a layer, window, loss or attention-size sweep.

Two new learned-attention arms share the same expanded validity:

1. Every finite current source OOF relative residual, including its permitted
   source seasonal mean.
2. Individual donor seasonal means only under that same expanded validity.

Compare both with the previously fitted full-current model restricted to matched
earlier-year availability. Candidate station identities, ecological prior,
daily-flow keys, receiving query, aggregate readout features and architecture
are unchanged. In particular, the old three aggregate descriptors still use
their original matched population; this study expands individual donor access.
The seasonal arm does not remove current aggregate information from the model.

Keep20 ecological candidates plus the zero prior, two32-dimensional heads,
37,900 parameters,12-month causal receiving GRU,30epochs/patience5 and the
existing native loss/validation fusion.18neural fits,90 reused double-held
references, no new forest fit. Source query and donor exclusion are unchanged;
receiving K0 has no water-quality input. Current donor values come from the
same or earlier allowed month, never a future source state. Source seasonal
statistics are frozen from source training. No new data is acquired.

## Analysis and continuation

Report MAE/Q90/bias, source partition and seed direction, station-equal MAE,
paired5,000 station intervals, newly usable donors/months, prior mass and raw
Shannon entropy. Compare expanded current with expanded seasonal and old matched
current in both native-only and complete procedures. A supported increment in
source roles can enter a separate fixed geographical study. If expanded access
does not help, retain the earlier donor convention and close this mechanism.
Keep all versions and support/test products; no region or K-specific winners.
