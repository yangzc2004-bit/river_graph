# Concentration-relative correction in the existing DOC neural residual

The source concentration/hydro correction pilot did not improve the retained
full model: conditional calibration changes source-validation MAE1.769053 to
1.782415, with0/3 positive partition directions. Its tree-only effect is small
and uncertain. Adding another source-OOF environmental bias can duplicate the
existing neural correction. Preserve that result; do not add its branch to the
deployed model or select its coefficients using geographical/external queries.

## One model change

Replace the existing signed native-concentration residual head's output rule
with a concentration-relative rule:

`DOC = max(0, expm1(log1p(environmental DOC) + scale * neural log residual))`.

Keep the same ecological encoder, observation-aware GRU,12-month causal inputs,
daily hydro/regime readout features, station-hidden environmental reference,
station-blocked OOF training reference and retained initial backbone weights.
The head still starts at zero. This is not the earlier Gaussian/mixture NLL
experiment: it directly minimizes the same native mg/L absolute-error loss
with source-Q90 weight2. Change no loss weights or model dimensions. Use30
epochs, patience5 and the existing global scales{0,.25,.5,1} on source validation.

The relative readout tests whether an absolute mg/L correction transfers poorly
between concentration regimes. It does not assume that a fitted power-law
correction represents a physical concentration process.

## Matched source development

Use142/143/144 × seeds42/43/44. Start each new neural fit from the saved **initial**
weights of the preceding30-epoch native candidate, rather than continuing its
trained head. Retain old native and complete results, old full model and strong
trees. Save the new relative expert and its unchanged ecological-memory fusion
as two arms. Memory acts on the new expert's native-mg/L predictions and uses
the same source/validation selection procedure.

Each receiving validation station has no water-quality inputs. Source
training/validation roles are available; old target, geographical and external
query labels do not participate. Report source-only model development, not a
fresh prospective confirmation. Source-validation means/paired station
bootstrap preserve seed-then-partition averaging, with5,000 draws. Include bias,
Q90, hydro and concentration strata. Do not combine different K/partition winners.

Keep the relative model only if it improves the retained complete procedure
with consistent source-validation directions. Otherwise retain the existing
native model and analyze how the concentration parameterization changes errors.
No new geographical/external scoring is launched before this comparison is
understood and the next retained version is fixed.
