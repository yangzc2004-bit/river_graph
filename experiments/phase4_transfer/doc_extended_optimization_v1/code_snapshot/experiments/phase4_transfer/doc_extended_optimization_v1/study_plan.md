# Longer optimization of the retained unmonitored-station DOC model

## Research question

The retained effective ecology/observation-GRU/native-residual model was refit
for at most 30 epochs with patience 5. All nine source fits stopped before the
cap (7–24 epochs); one selected epoch 2. This experiment tests whether a longer
validation plateau allowance finds a better solution with the SAME information
and structure. Early stopping itself is not evidence of undertraining.

## Single change

Use maximum 120 epochs and patience 15. Keep the retained log-trained
station-hidden environmental tree, its cached station-blocked OOF references,
all raw and readout inputs, source normalization, ecology/GRU/decay initial
weights, zero scalar head, learning rates, batch size, loss, tail weight 2 and
validation scale candidates. Use the unchanged ecological-memory integration
after selecting the neural checkpoint. No new model head, weather data,
reference distribution or ensemble weighting is introduced.

## Development population and products

Source partitions 142/143/144 × seeds 42/43/44, nine fits. Strip auxiliary
water-quality channels and remove old target test truth before constructing
training arrays. Preserve the retained source validation panel and its five
prediction arms; add the longer neural-only and complete predictions. Reuse
the original valid OOF cache rather than refitting the same trees. Save the
neural checkpoint before memory export for resumption.

Compare the complete procedure with the actual retained complete model,
strong station-hidden trees and preceding full model. Report overall/Q90 MAE,
bias, partition/seed directions, training trajectory and 5,000 paired station
draws. Verify matched inputs and initial states, saved-state replay, source
OOF population and optimizer-history prefix. Inspect the figures. A synthetic
training test checks that extending the budget does not change the common
epoch trajectory. New fits use a separate directory and saved execution code.

## Interpretation

A better source result would support more patient optimization of the current
structure. A flat or worse result would close this schedule question without
attributing failure to unavailable weather information. Keep one complete
procedure across regions and K. Previous geographical/external results are
untouched; source validation selects checkpoints. Any later confirmation must
preserve those earlier evaluations and identify its retrospective status.
The common training prefix makes a non-worse neural validation minimum expected
by construction; its sign is not independent evidence of improved transfer.
Magnitude and later selected epochs determine whether confirmation is worthwhile.
