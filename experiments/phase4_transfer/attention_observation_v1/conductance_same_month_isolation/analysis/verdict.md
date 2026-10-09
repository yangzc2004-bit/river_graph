# Conductance three-arm isolation

The strict temporal conductance pilot compares no-message, same-month upstream attention, and same-month plus edge-lag attention.

Mean raw-unit MAE:

- no-message: 105.87
- same-month upstream attention: 105.59
- same-month plus lag attention: 104.87

The full river arm is about 0.72 units (0.68%) better than the same-month arm on mean MAE, but the improvement is driven mainly by seed 43; seed 42 slightly favors the same-month arm and seed 44 is tied. Therefore the data support a small, analyte-specific graph correction for conductance, while the incremental lag contribution remains weak and seed-sensitive.
