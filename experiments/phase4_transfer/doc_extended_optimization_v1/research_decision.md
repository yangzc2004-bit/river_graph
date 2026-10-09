# Research decision: longer optimization of the retained DOC structure

## Results

Nine source-validation fits are complete (partitions 142/143/144 × seeds
42/43/44). Maximum epochs increased from 30 to 120 and patience from 5 to 15;
the environmental reference, OOF predictions, input features, normalization,
backbone initialization, learning rates, loss and architecture were unchanged.

| Procedure | K0 MAE (mg/L) | Q90 MAE (mg/L) | Signed bias (mg/L) |
|---|---:|---:|---:|
| Retained complete model | 1.769053 | 9.075420 | -0.638608 |
| Retained neural-only model | 1.772035 | 9.053055 | -0.625464 |
| Longer optimization, complete | 1.767218 | 9.071679 | -0.636894 |
| Longer optimization, neural-only | 1.770248 | 9.038023 | -0.621215 |
| Strong station-hidden trees | 1.866362 | 9.394458 | -0.611014 |

The complete model's gain over the actual retained complete model is 0.104%
[-0.068%, 0.281%] from 5,000 paired station draws. Only three of nine packages
change positively; two partition averages improve. Q90 gain is 0.041%
[-0.087%, 0.196%]. The neural-only gain over the matched retained neural-only
model is 0.101% [-0.107%, 0.320%]. These small changes do not justify replacing
the current release or repeating the geographical/external matrix.

Three fits select later checkpoints: split143/seed42 selects epoch41 rather
than19, split144/seed42 selects37 rather than9, and split144/seed44 selects27
rather than17. The other six select their original checkpoint. All stop between
17 and56 epochs, below the new120 cap. More patient optimization finds some
later useful updates, but it does not produce a substantial performance gain.
Increasing the maximum cap alone is therefore not a promising next step.

## Scientific reading and decision

The optimizer trajectory before the original stopping point is bitwise
identical in all nine fits. A non-worse neural source-validation minimum was
expected because the longer search includes every original checkpoint. Its
positive sign is not an independent demonstration of improved spatial transfer.
The relevant result is its very small magnitude. The apparent 5.31% gain over
strong trees is mostly the retained model's existing benefit, not a new5.31%
optimization improvement.

Keep the retained complete procedure and its deployed/paper products. Archive
the extended schedule as a completed optimization experiment. No additional
epoch, patience or learning-rate scan is initiated from these results.

The renewed source studies have now tested reference objectives, conditional
medians, ecological/hydrological inputs, auxiliary pretraining, nonlinear
readouts, reference histories and stopping budget. Their small or negative
increments suggest that simple changes to the current readout and optimization
are not delivering the desired transfer improvement. This is an empirical
pattern, not proof of an irreducible error floor.

One remaining representation question is simultaneous source chemistry
supervision. Previous sequential pH/conductance pretraining improved auxiliary
reconstruction but worsened DOC after the subsequent DOC fit. Joint training
can test shared environmental states while maintaining the DOC objective
throughout. It is a distinct source-only experiment, with a matched label-shuffle
control and the original30-epoch DOC budget. New-site chemistry remains absent.
Its own plan is saved separately; this result does not establish its usefulness.

## Verification and products

All nine saved neural/memory predictions replay bitwise. Initial weights, model
settings apart from the two declared budget parameters, raw/readout inputs,
source OOF population, query cells and parent predictions were checked. Common
optimizer histories match bitwise. Analysis uses5,000 paired station draws;
PNG/PDF/SVG were generated and the PNG inspected.964 tests pass, two skip;
Ruff and historical audit pass. Large fits remain local. Related files are in
the whitelist; Git index writes remain unavailable under the workspace policy.
These are source-development results; previous ST357 geographical withholding
and independent-basin results remain unchanged.
