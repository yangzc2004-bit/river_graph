# Graph upgrade continuation R1

This directory contains the continuation after the first mechanism pilot.
The first pilot remains intact under `../m1`, `../m2` and `../m3` and is
treated as exploratory evidence.

## Changes in this revision

- The multi-scale model computes causal short and seasonal convolutions over
  the chronological hidden sequence and carries a causal trend GRU through
  time. It no longer materializes one duplicated rolling window per
  station-month.
- M3 uses a 24-month declared long-history setting. The recurrent trend path
  is causal and can carry information from earlier months; it does not read
  future months.
- M2 run names include the lag mode (`learned`, `fixed`, `static`, or
  `none`) so the physical controls cannot overwrite one another.
- New sidecars use configuration schema V4 and incompatible cached artifacts
  are refused unless `--force` is explicitly supplied.
- Age/support diagnostics use station-month aligned arrays and fixed,
  interpretable strata rather than a test-derived median split.

## Scientific sequence

1. Complete the four M2 lag controls at the matched 30-epoch budget.
2. Use the refactored M3 on the same two holdout families if M2 remains
   informative or if the temporal representation is still the main
   limitation.
3. Compare any surviving mechanism with the temporal random-forest baseline
   before combining mechanisms.

The continuation is an experiment revision; it does not rewrite the earlier
pilot tables or their interpretations.
