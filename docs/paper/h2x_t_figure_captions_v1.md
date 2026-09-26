# H2X-T figure captions (v1)

## Figure 1. Five-seed temporal improvement across analytes and missingness regimes

Heatmap of relative MAE reduction from H2X-T compared with the matched H2X
snapshot model. Each cell averages five training seeds on the same hidden
station-month query set. E1 is random cell masking, E2a/E2b are temporal
missingness families, and E3 holds out spatial stations. Green indicates a
larger reduction. Values are reported in native analyte units through the MAE
calculation; the displayed percentage is `100 × (1 − MAE_H2X-T / MAE_H2X)`.

## Figure 2. Dependence on temporal window length

Mean E2a/E2b MAE for lookbacks of 1, 3, 6, and 12 months, using three matched
training seeds. The curves show the strongest window dependence for pH and
specific conductance and little change for DOC. The figure is a diagnostic of
the matched training budget; it does not separate the amount of historical
information from the computational effect of a longer GRU unroll.

## Figure 3. Descriptive station traces under E2a masking

H2X-T five-seed ensemble median predictions and observed values for one fixed,
data-availability-selected station per analyte. Shading is ±1 seed standard
deviation and represents ensemble dispersion, not a calibrated prediction
interval. The traces illustrate seasonal reconstruction and local departures;
they are not used to define a primary endpoint.

## Figure 4. Spatial distribution of E2a test error

Station-level mean absolute error for the H2X-T five-seed ensemble median on
E2a hidden cells. Panels show DOC, pH, and specific conductance. Point colors
use an analyte-specific scale. Stations with fewer hidden cells contribute
fewer cells to the station mean; the panel is descriptive rather than a
sampling-priority analysis.
