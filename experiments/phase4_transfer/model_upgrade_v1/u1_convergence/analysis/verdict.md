# U1 convergence pilot verdict

U1 compared the 10- and 30-epoch EcoHydroGraph budgets for DOC and specific
conductance under strict temporal extrapolation and unmonitored-station
holdout. The 60-epoch arm was stopped after the 30-epoch arm met the entry
criterion; partial files are retained but are not used in the comparison.

## Result

The 30-epoch budget is accepted for subsequent pilot work. Relative to 10
epochs, the mean MAE reduction was 13.7% (DOC, strict temporal), 22.4% (DOC,
unmonitored stations), 10.6% (specific conductance, strict temporal), and
26.9% (specific conductance, unmonitored stations). The 10-epoch T8/T9 result
remains frozen and is not rewritten.

## Reproduction

`.venv/bin/python scripts/analyze_u1_convergence.py`
