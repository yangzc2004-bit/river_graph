# K2: upstream message source isolation

## Question and model

Can upstream information correct the RF-local OOF residual when the correction
cannot be produced by a free local predictor?

The spatial module aggregates only upstream edge messages. It retains first-hop
messages when adding the second hop, so stations with a single upstream reach
remain represented. No raw receiver self path is added. Ecology and hydro remain
part of sender states; receiver observation age and support modulate memory decay.

The temporal module is a 12-month causal GRU with **no bias parameters** and no
history-valid input channel. Padding skips state updates. Its readout uses the
existing zero-initialized head weight, without a head bias. Therefore zero
messages produce zero correction throughout training, including at isolated
stations in an otherwise connected graph. This property is tested after
optimization, with nonzero head weights and arbitrary head bias.

The null uses the same model and empty edges. It is exactly RF-local by design;
it is not a competing learned local residual model. K1's no-message learned
residual remains a separate reference. A positive difference against this null
would establish that this message architecture improves the RF base, but would
not alone prove that correct river connectivity, rather than a correction
associated with having upstream neighbors, explains the result.

## Runs and reporting

- DOC; temporal extrapolation (`e2a_strict`) and station holdout
  (`e3_spatial_seed42`); seeds 42, 43, 44.
- Two arms: upstream message-only and empty-message null; 12 products.
- Restore the original K1 budget: **30 maximum epochs, patience 5**, hidden 64,
  two spatial layers, dropout 0.1, Adam 0.001, 200-tree RF, five OOF station folds.
- Use unchanged K1 train/validation/test visibility roles, target transform and
  query cells. All early stopping uses validation labels only.
- Report each seed, mean per-seed MAE, Q90 sample count/error, and graph-delta
  variation. Paired station bootstrap resamples stations with all their cells;
  each draw recomputes cell-weighted mean error and averages over seeds. Thus
  the main table, point difference, percentage and interval estimate one quantity.
- Retain K1 RF-context and learned residual results as context. No selection of
  a different model, mask, seed or epoch uses test performance.

This remains a mechanism experiment, with scientific follow-up chosen after
examining effect size, training curves and message sensitivity.

## Development history

The original 30-epoch double-forward attempt (`k2_source_isolation`) and the
large-chunk benchmark (`k2_bench`) were interrupted for runtime reasons. Their
partial traces remain available. The first message-only fast attempt
(`k2_source_isolation_fast`) was interrupted because GRU offsets could produce
corrections even without a message at a node. A centered dual-GRU attempt
(`k2_source_isolation_v2_fast`) was interrupted for runtime reasons. The three
`k2_smoke*` directories are engineering runs, not comparable scientific results.
None of these directories enters the v3 analysis. Some engineering outputs were
seen; this is a documented implementation revision. The proposed eight-epoch
screen is superseded by the original 30-epoch maximum after eliminating the
second temporal forward.
