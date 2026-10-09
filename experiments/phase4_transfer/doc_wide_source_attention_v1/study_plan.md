# Wider source experience for sparse DOC reconstruction

## Question

Is the fixed pool of 20 ecological neighbours discarding useful source
observations? The current-availability source study raised supported receiving
cells to approximately 92–93%, but only 3.5–5.0 donors per month were usable.
Test a larger candidate pool while preserving the same learned information
selection and environmental reference. This is a donor-access experiment,
not a larger network.

This plan is written before inspecting the current-availability geographical
replication, whose 25 packages have finished and whose exact replay/analysis
is in progress. Development uses source training/validation roles only.

## Fixed change and source experiment

- Increase the ecological candidate limit from 20 to **60**. Keep the
  source-fitted distance scale (nearest-five ecological distances), the same
  exponential prior and unit zero prior. The original nearest 20 remain the
  prefix of the wider pool. No candidate-count sweep.
- Candidate access uses finite contemporaneous source OOF relative residuals
  without requiring an earlier-year observation. Restore the permitted source
  seasonal mean, as in the preceding source-selected current-availability
  model. Only the individual pool changes; the three matched aggregate
  readout descriptors remain unchanged.
- Actual current values and individual seasonal-only values use the same
  wider validity. Carry the preceding all-current 20-donor model, matched
  full/anomaly-only procedures, retained complete and trees unchanged.
- Keep the same ecological encoder, daily-flow source keys, 12-month receiving
  GRU query, two 32-dimensional heads, first-order concentration conversion,
  native loss, tail weight, validation scales and ecological fusion.
  There are still 37,900 parameters; 30 epochs, patience 5.
- Splits 142/143/144 × seeds 42/43/44: nine packages, 18 neural fits,
  90 reused double-held reference fits, no new forest or acquired data.
- Receiving K0 has no DOC, pH or conductivity. Query fold A is absent from the
  source library; each donor reference excludes A and the donor fold B.
  Source preprocessing is fixed before inference. Current source values and
  daily-flow keys are calendar aligned; no future source observation is read.

## Evaluation and continuation

Compare wide versus preceding current-availability complete and native-only
models, then actual versus seasonal-only wide values. Separately report
accumulated gains versus retained complete and strong trees. Average seeds
within source partition, then weight partitions equally; use 5,000 paired
station bootstrap draws. Report MAE, Q90, bias, station-equal scores, partition
and seed directions, supported-cell fractions, usable donor counts and raw
attention entropy/prior mass. The repeated seeds do not increase the number
of station-month observations.

An incremental source-role benefit can enter a fixed geographical replication;
otherwise close the wider-pool mechanism and retain the preceding candidate.
Do not choose a pool size, model or K-specific winner from scored geographical
or external data. Preserve completed source, geographical, external and
portable products. Inspect the scientific figure and save a short English
research decision. Saved fitting states and exact replay are required to
interpret the matched comparison.
