# Chemical-state station calibration: completed

Nine packages cover station partitions142/143/144 and training seeds42/43/44.
The nonlinear chemical decoder, forests, recurrent backbone and ecological
memory remain fixed. Two source-fitted chemical coordinates augment the two
existing GRU coordinates used by the station-support calibrator. Mask-only
coordinates provide a matched information control. A source-validation rule
chooses among legacy, mask-augmented and chemical-augmented representations
separately for each pipeline and K; all raw curves are also reported.

## Performance

MAE is mg/L, averaging seeds within partition and then partitions equally.
K is the number of reserved DOC readings at each held station. These readings
provide retrospective station calibration; every K uses the same query cells.

| Model | K0 | K1 | K3 | K5 |
|---|---:|---:|---:|---:|
| Retained general integrated model | 1.803538 | 1.755368 | 1.620090 | 1.564984 |
| Chemical neural model, legacy calibration | 1.764947 | 1.738600 | 1.614240 | 1.568820 |
| Chemical neural model, mask coordinates | 1.764947 | 1.738600 | 1.613222 | 1.569553 |
| Chemical neural model, chemical coordinates | 1.764947 | 1.738600 | 1.595816 | 1.556278 |
| Chemical neural model, validation-selected coordinates | 1.764947 | 1.738600 | 1.595851 | 1.556939 |
| Chemical tree, legacy calibration | 1.802261 | 1.768522 | 1.565090 | 1.537905 |
| Chemical tree, validation-selected coordinates | 1.802261 | 1.768522 | 1.560343 | 1.535534 |

At K3 the chemical coordinates improve the integrated neural model over its
legacy coordinates by1.141%: paired station-bootstrap change
−.018423[−.029765,−.008136]. Against the mask-coordinate control the change is
−.017406[−.028567,−.007286]. This supports useful measured chemical-state
information in station calibration, beyond availability flags alone.

The complete validation-selected neural pipeline improves K3 MAE by1.496%
over the retained general model: change−.024239[−.044780,−.007424]. All three
partitions improve. Its Q90 MAE falls6.900415→6.793270, a1.553% reduction,
change−.107144[−.200464,−.018690]. Ordinary-DOC change is
−.015302[−.033658,+.000366]; recall and false-high changes are unresolved.

At K5 the selected neural pipeline improves numerically to1.556939:
0.757% versus its legacy chemical calibrator and0.514% versus the retained
general model. All nine package estimates improve versus the general model,
but the overall paired interval remains−.008045[−.023271,+.003925]. Its Q90
MAE6.518383 is lower than6.586994, with an interval crossing zero. These are
promising K5 development estimates, rather than established K5 superiority.

The selected chemical tree remains stronger at K3 by.035508
[.007203,.067219]. Its K5 MAE1.535534 is also numerically lower than the
neural pipeline; that difference+.021405[−.004193,+.049284] is unresolved.
The new calibration therefore improves the neural model without demonstrating
an overall win over the chemical tree.

## Model decision and next step

Retain the validation-selected chemical calibration as the improved
chemistry-informed neural candidate. Its whole K curve is a single frozen
validation-selected procedure; no target-selected splice is introduced.
The preceding general model and chemical tree remain reported references.
K0/K1 are unchanged by construction, preserving the preceding K0 decoder gain.

The source-only diagnosis finds strong transfer of station-mean residuals
(correlation.880), but weak within-station shape transfer (.063→.118 after
chemical augmentation). A small follow-up will test bounded, query-specific
interpolation of the remaining support residual, retaining the existing linear
fit. The pilot will first use source validation. Chemical distance itself has
only a weak relationship with mismatch; the follow-up tests a prediction
operator and does not assume a demonstrated distance-reliability mechanism.

## Execution and products

Calibration took17.10 summed seconds; no neural network or forest was fitted.
Independent replay reconstructed18 source PCA fits,54 direct adapters,
27 ecological mixers and108 representation choices. All468 query panels
(2,006,940 rows),144 copied legacy panels and162 K0/K1 invariance panels
match exactly. The copied native grids total2,101,302 rows and remain
byte-identical to their parents. New representations and positive-K query
products are stored separately. The independent row-local batch-shape check
uses1e-12 tolerance; saved products and controls replay bitwise.

The full suite has814 passed and2 skipped; Ruff passes and the historical
artifact audit exits0. All22 fixed contrasts use5000 paired whole-station
bootstrap draws with shared station multiplicities across overlapping
partitions. Full curves, directions, Q90 detection tradeoffs and availability
strata are in analysis/; reproducible source-only diagnosis is in diagnostics/.

Measured pH or conductance is known for98.43% of observed-DOC query cells but
only19.735% of genuinely DOC-missing cells. Missing chemistry retains the
preceding final prediction exactly. Monthly chemistry may come from different
sample times. These reused random-station partitions are development evidence
within the Mississippi cohort, not external-basin confirmation or forecasts
before chemistry measurements arrive. The current encoder uses its empty-edge
self path; the gain is not evidence of new river-message value.
