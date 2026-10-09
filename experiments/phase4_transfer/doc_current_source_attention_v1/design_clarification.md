# Input and comparison interpretation

The three new attention arms have identical candidate pools, individual donor
ecology/current hydrology keys, parameter allocation and fitting budgets. The
historical arm replaces both donor values and the preceding aggregate innovation
readout with earlier-season values. Support, hydrological keys and visibility
stay matched. The fixed-prior arm sets learned query/key score contributions to
zero, retaining connected zero gradients and the same value-output projection.

All seven retained daily-hydro interactions plus innovation index38 remain.
Extra width is41; attention adds6,274 allocated/trainable parameters to the
31,626-parameter source-innovation model, for37,900 total. Fixed-prior query/key
weights have zero influence and do not update, so equal allocation does not
mean equal effective adaptive capacity.

The saved enriched-tree comparator receives the previous aggregate innovation,
support count and ecological weight mass. It uses the same source-data origin,
but does not receive this attention operator's individual donor hydrological
keys. Treat it as a strong saved aggregate-information comparator; a gain over
it would not isolate neural architecture with completely identical raw inputs.
The two new matched attention controls address adaptive allocation and timing.
No tree is refitted in this version.

Current-month daily flow and source observations are retrospective reconstruction
inputs. This study does not forecast before the month ends. Source seasonal
statistics are fitted on permitted spatial-training records and fixed at
inference; future-input tests hold those fitted statistics unchanged.
