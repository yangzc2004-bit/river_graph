# Exact concentration-ratio correction from source experience

## Question and prospective change

The completed source experiment transfers dimensionless, season-centered source
OOF log1p departures through a first-order conversion. Test the exact inverse
conversion for this source branch. This plan is saved before opening the
relative-value geographical numerical analysis. Only source roles142/143/144
are used for development.

Let B be the frozen receiving environmental reference, L the existing native
local temporal readout and u the projected relative donor state. Compare:

    First order: prediction = B + L + (1+B)*u
    Exact ratio: prediction = B + L + (1+B)*expm1(u)

Both preserve the old local native readout, native tail-weighted loss and final
nonnegative prediction rule. At zero source output their prediction and source
projection gradient coincide. This does not repeat the earlier whole-model
log-output/log-loss experiment: only the source correction's inverse transform
changes. It is a statistical concentration-ratio transfer, not a conservation
or physical-process constraint.

Use the same ecology self encoder, observation GRU,37,900 parameters, two32-
dimensional heads, candidate pool, sources, references, initial weights and
30epochs/patience5. Source values remain double-held-fold OOF log1p residuals
minus donor seasonal means. Receiving reference is station-OOF during training
and full-source fitted during validation. No receiving DOC/pH/conductance at K0.
Reuse90 forests; fit no new reference.

Nine fixed packages (partitions142/143/144 ×seeds42/43/44), three arms:
exact-current learned, exact-current fixed ecological prior, exact earlier-season
learned. Preserve all saved first-order, native-value and retained comparators.
No region, K or donor-weight winner is selected. Report complete and neural-only
MAE, Q90, bias, station-equal error, source/reference strata, allocation and5,000
paired station intervals. Isolate exact-vs-first-order comparisons and retain
negative results. First check zero fusion, derivative, causal inputs and reload,
then one technical package and all nine; make any geographical decision from
the completed source evidence. Keep evaluated geographical/external products
and the portable release unchanged during development.
