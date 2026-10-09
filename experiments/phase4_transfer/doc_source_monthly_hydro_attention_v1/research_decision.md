# Research decision: monthly donor hydro keys

All nine source-development packages are complete:142/143/144 ×42/43/44,
27neural fits,90reused double-held references and no new forest fit. The three
arms share38,156 parameters, the same receiving GRU, full relative source
values and30-epoch/patience5 budget. Four new key coefficients start at zero.
Current source monthly temperature/discharge is compared with source observed-
period means and four zero key columns. The mean control retains current hydro
visibility. It is a spatial source-transfer control, not an online climatology.

## Result

| Complete procedure | Source-validation MAE, mg/L | Q90 MAE, mg/L |
|---|---:|---:|
| Current monthly hydro keys |1.736354|8.923656|
| Source-mean hydro keys |1.735215|8.910214|
| Monthly key columns removed |1.736968|8.927563|
| Preceding full-source model |1.736968|8.927563|
| Preceding anomaly-only model |1.748102|8.932044|
| Actual retained complete model |1.769053|9.075420|

The isolated current-hydro gain over the preceding full-source model is
**0.035% [-0.178%,0.242%]**, with five of nine packages, two of three
partitions and two of three training-seed averages improving. Its Q90 gain is
0.044% [-0.089%,0.226%]. Both intervals cross zero. The zero-key control
reproduces the preceding full-source summary, supporting the intended input
comparison.

Current monthly keys do not outperform source-mean keys: overall gain-0.066%
[-0.201%,0.094%], one positive partition and no positive seed average. Q90
gain is-0.151% [-0.300%,-0.052%]. This experiment provides no evidence that
the added month-specific source hydro values improve donor selection beyond
the receiving hydro history, source ecology and existing daily descriptors.

Native-only isolated gain over the preceding full-source neural model is
0.017% [-0.230%,0.243%], only one positive partition and seed average.
Native Q90 gain is0.052% [-0.099%,0.260%]. The complete model's1.848%
[1.031%,2.788%] overall gain and1.672% [0.529%,3.225%] Q90 gain against
retained include the earlier full-source/relative improvements; they are not
the contribution of these four new source key columns. Likewise, its0.672%
gain over anomaly-only includes restoration of seasonal source values.

Mean learned monthly-key coefficient norm is1.218 for current values and1.417
for source means, compared with zero for the input-removal control. Thus the
new parameters learned; this is not a disconnected-branch failure. Matched
validation donors have mean observed temperature support3.162 and discharge
support2.655. The experiment retains every arm, partition and seed.

## Decision and next experiment

Close this source monthly-key mechanism without a geographical matrix or a
hyperparameter scan. Keep the preceding source model for further development;
the separate seasonal-source geographical study did not establish an additional
geographical improvement over anomaly-only relative attention. Keep that
earlier anomaly-only procedure as the strongest tested geographical candidate.
These two judgments refer to different evaluation populations.

Next, `doc_current_availability_attention_v1` tests usable source information.
The timing controls required both current and earlier donor observations, which
can hide genuine current DOC when no older observation exists. Open every
permitted finite current source value, retaining the same20 candidates, source
keys, aggregate readout and37,900-parameter first-order model. Compare expanded
current values with expanded individual seasonal means and the already fitted
matched-current model. All choices and fitting again use source roles only.
The source library and OOF references still exclude query and donor folds;
receiving sites provide no water quality. This is an availability change,
without acquiring data or altering the evaluated geographical/external models.

All nine candidate/library/initial-state/neural/fusion/diagnostic replays pass
bitwise.5,000 paired station draws average seeds within source partition and
partitions equally, keeping each station's months together. Figures were
generated and their actual PNG inspected.999tests/two skips, Ruff and the
historical artifact audit pass for this version. Large fits and executed source
copies are retained; Git submission remains pending under the read-only index
policy.
