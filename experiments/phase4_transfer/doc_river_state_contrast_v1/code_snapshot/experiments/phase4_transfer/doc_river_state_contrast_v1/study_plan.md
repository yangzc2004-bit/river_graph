# Upstream-to-local environmental message contrast

## Question

Does the existing river correction benefit from explicitly comparing an
upstream environmental state with the receiving site's current state? The last
node-expansion version improved MAE only 0.063% beyond its preceding state model;
uniform allocation was slightly better and matched connectivity was unresolved.
This version changes message values, not candidate reach or attention capacity.

## One mechanism change

For each allowed upstream candidate and lag 0/1/3, the 93-dimensional value is
`state(upstream, t-lag) - state(receiver, t)`. Both sides use the same frozen
chemistry-free ecological encoder and observation-aware GRU. Receiver time is
explicitly the CURRENT query month, not the source's lagged month. The subtraction
is an environmental information contrast, not a DOC concentration gradient,
physical travel-time model or statistical causal effect. Path/attention features,
queries, eligibility, donor matching and zero-message option remain bitwise
unchanged from the preceding absolute-state experiment.

## Development and input roles

ST357 source partitions 142/143/144, seeds 42/43/44. Receiving K0 sites have no
DOC/pH/conductance inputs. Environmental candidates are known-atlas covariates
except the entire receiving fold/validation role; labelled DOC experience stays
source-only. Source environmental normalization and existing encoder are retained.
Daily descriptors are retrospective within-month inputs. This is transductive
known-atlas source-role DEVELOPMENT, not geographical or external confirmation.
The complete and observed-DOC anchors used these validation roles previously;
source training anchors are fitted, not complete-neural OOF predictions.

## Controlled arms and budget

36 new fits: 3 partitions x 3 seeds x four arms:

- `contrast_upstream`: genuine upstream contrast values, learned allocation.
- `contrast_uniform`: same contrasts/support, uniform allocation.
- `contrast_matched_upstream`: real contrasts with common real/control validity.
- `contrast_matched_nonupstream`: matched non-ancestor contrasts with identical
  slot/path descriptors and common validity; assigned paths are fictitious
  control slots, not physical paths.

All branches start from the SAME frozen observed-DOC anchor, never add their
correction to the preceding absolute-state correction. Keep 20 candidate slots,
3000-km real-path cap, causal 12-month measured-hydro support, lags 0/1/3,
2 heads x32, 12098 trainable parameters, Adam 0.001, batch 512, 30 epochs,
patience 5 and source-Q90 tail weight 2. Epoch zero remains selectable.

## Comparisons and completion

Copy the complete model, strong trees, observed-DOC branch, prior source-only
state and expanded absolute-state dynamic/uniform/matched arms unchanged.
Primary new-mechanism comparison: contrast_upstream versus expanded_upstream.
Also report accumulated gains versus complete/observed anchors, dynamic versus
uniform contrast, matched real versus non-upstream and matched contrast versus
its absolute counterpart. Native MAE, Q90 MAE, log1p error, station-equal error,
all partition/seed results and support/path strata; 5000 paired whole-station
bootstrap draws jointly across partitions, seed first and partitions equal.
Candidate coverage should be IDENTICAL to the absolute-state experiment.

Rebuild/replay all nine input packages and all 36 checkpoints; independently
verify subtraction and arithmetic, inspect scientific figures, write a research
decision and run pytest, Ruff and historical audit. Keep old results, fitting
caches and execution snapshots; do not tune this version on seen geographical
or external outcomes or replace the declared primary arm after viewing results.
