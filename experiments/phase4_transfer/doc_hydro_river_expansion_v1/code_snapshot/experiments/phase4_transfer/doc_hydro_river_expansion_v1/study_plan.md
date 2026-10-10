# Hydro-only river-node expansion

## Scientific question

Does covariate-only upstream coverage, omitted by the DOC source-station role,
limit the existing river operator? Keep the ecological encoder, observation-aware
GRU, observed-DOC branch, source preprocessing and residual allocation capacity.
Expand only the eligible environmental node pool; no additional chemistry enters.

## Information and development roles

ST357 partitions 142/143/144 and training seeds 42/43/44. Source-validation
receiving sites have no DOC/pH/conductance inputs. Training episodes exclude the
whole receiving station fold from the environmental bank. Validation excludes
the whole validation station role. Every other ST357 node may supply known
temperature/discharge, their visibility, calendar, ecology and retained daily
hydrology. Previously held target sites may supply these covariates, never water
quality values, masks, histories, support or statistics. The labelled DOC
experience library stays source-only. This is known-atlas transductive covariate
use, not independent geographical or external inductive validation.

The complete predictor and observed-DOC branch stay frozen. Source training
anchors are fitted predictions, not complete-neural OOF predictions. Source
validation previously selected those anchors and again selects each new branch;
all estimates are developmental. No geographical/external labels select models.

## Label-free audit before new training

Coverage audit reads frozen query identities, no DOC values. There are 71 newly
eligible covariate-only nodes in each partition. Usable upstream support rises
from 28.997% to 31.888%, 32.236% to 36.147%, and 63.463% to 64.347%; 66/120/31
query cells gain support, zero lose it. Known-node static/ecology normalization
is retained from the original model; monthly hydro statistics use original
source train cells. Every original source state must replay exactly.

## Controlled arms

Four new fits per package, 36 fits total:

1. `expanded_upstream`: genuine upstream environmental states.
2. `expanded_uniform`: same states/support with uniform allocation.
3. `expanded_matched_upstream`: real donors with common real/control validity.
4. `expanded_matched_nonupstream`: non-ancestor donors matched on drainage area
   and label-free hydro availability; same slot/path descriptors and validity.
   Their assigned path descriptors are control slots, not physical river paths.

Each adds a zero-initialized state correction to the same frozen observed-river
prediction. Two heads, dimension 32, at most 20 genuine ancestor nodes within
3000 km, lags 0/1/3 months; causal preceding-12-month hydro availability. These
monthly lags describe available information, not estimated water travel times.
30 epochs, patience 5, Adam 0.001, batch 512, source Q90 weight 2; epoch-zero
baseline remains eligible. Daily descriptors are retrospective within-month
information. No new measured hydrology is acquired in this version.

Fixed complete model, strong trees, observed-DOC branch, previous combined state
branch and previous matched pair are copied intact. Primary compare expanded
upstream against previous `observed_state`, observed-DOC and complete anchors.
Uniform and matched non-upstream controls diagnose allocation and connectivity.

## Analysis and completion

MAE, RMSE, R2, log1p MAE, source-Q90 tail MAE, station-equal error; seed mean
within partition, then three partitions equal. 5000 paired whole-station draws
jointly across partitions. Report newly supported sites/cells, previously
supported cells, unavailable cells, and actual path-distance groups. Always
distinguish gain over the complete predictor from the additional pool-expansion
gain. Evaluate all arms, replay checkpoints/inputs, inspect scientific plots,
write an English research decision and run existing tests, lint and historical
audit. Preserve previous results and execution code. No automatic deployment
or change to the manuscript's established river claims follows a small gain.
