# Region-hidden source training: development decision

## Result

All nine fixed packages are complete and verified. Retain the preceding
station-hidden native-residual procedure. The tested region-hidden training
does not improve source-validation new-site reconstruction.

| Procedure | Equal-partition K0 MAE, mg/L | Mean bias, mg/L | Q90 MAE, mg/L |
|---|---:|---:|---:|
| Retained complete |1.769053|-0.638608|9.075420|
| Regional complete |1.833923|-0.627568|9.252839|
| Regional neural residual alone |1.834233|-0.620723|9.239027|
| Regional environmental trees |1.898415|-0.581776|9.425894|
| Strong station-hidden trees |1.866362|-0.611014|9.394458|

Regional complete gain over the actual retained model is -3.667%
[-5.500%,-1.929%], with zero of three positive partition directions.
The regional environmental reference itself worsens by1.717%
[0.543%,2.984%] against the matched strong trees. The new neural branch
still reduces its own regional tree's error by3.381% [1.379%,5.557%].
Complete fusion gives3.397% [1.481%,5.476%] over that new reference. These
within-version residual gains do not overcome the full procedure's regression.

The harder regional observation regime therefore changes both the source OOF
reference and learned correction without delivering a net source-validation
benefit. This experiment does not show that geographical outages are irrelevant:
the evaluation sites are the retained source-development validation roles,
not a fresh whole-region confirmation. It does show that this particular
training replacement should not displace the current model.

## Experiment and verification

Partitions142/143/144 × seeds42/43/44 retain the original architecture,
initial backbone states, native MAE/tail objective and30-epoch budget. Source
HUC4 groups are kept intact in observation views and nested forest OOF fitting.
The held outer regions are absent from both forest fitting labels and DOC
context. These source input views are not a neural cross-fitting ensemble.

The source node table mixes HUC8 and HUC12 representations. V1 stopped before
any fit because the reader accepted only HUC8. V2 restores the appropriate
width, including leading zeros, and derives HUC4. V1 sources/configuration/log
are retained. No scientific setting or prediction was silently replaced.

`scripts/analyze_doc_regional_source_training_v1.py --bootstrap-draws 5000`
checks all nine saved identities, complete source OOF coverage, outer/inner
region exclusion and unchanged parent predictions. Neural and complete product
replay is bitwise; forest accumulation roundoff is below the documented1e-12
relative/absolute tolerance. The5,000-draw bootstrap retains months within
station, shares station multiplicities across development partitions, averages
seeds within partition and weights partitions equally. Source Q90, signed bias,
station metrics and hydrological availability accompany the primary MAE.

The full test suite passed935 tests, with two skips; Ruff and the historical
artifact audit passed. `figures/source_development_comparison.png` was actually
inspected; vector PDF/SVG and source hashes are available alongside it.

## Next mechanism

Restore the retained station-hidden source views and environmental reference.
The existing ecological/GRU residual uses a linear scalar head over recurrent
states and selected interactions. Add a small nonlinear residual readout,
initialized to zero alongside the original linear path. Test its ability to
represent hydro-ecological conditional corrections with the same source inputs,
starting backbone weights, training budget and matched parent predictions.

Previous geographical and independent-basin queries have not been scored or
used to choose either this regional mechanism or the next nonlinear head.
