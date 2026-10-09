# Transfer seasonal source bias alongside contemporaneous departures

## Scientific question

The completed concentration-relative model subtracts each source station's
calendar-month OOF residual mean before transfer. That removes persistent
seasonal bias as well as avoiding native-unit scale differences. Test whether
some useful new-station information was removed. The existing static ecological
memory estimates annual native residual means; it does not provide this
calendar-month relative donor value to learned hydrological attention.

For permitted source station j and month t, split its OOF log1p residual into

    r_jt = log1p(y_jt) - reference_jt
         = seasonal_mean_j,month(t) + current_departure_jt.

Compare four new arms in the existing first-order relative attention:

1. full_current: current departure plus seasonal mean, learned allocation;
2. full_fixed: the same full residual, fixed ecological prior;
3. full_historical: earlier-year same-calendar-month departure plus the
   corresponding seasonal mean, learned allocation;
4. seasonal_only: seasonal mean alone, with identical candidate availability,
   hydro keys, learned allocation and parameter count.

Keep the complete first-order anomaly-only model, its native ablation and
all actual retained/tree comparators. Do not carry the exact inverse operator:
its isolated increment was0.052% with an interval crossing zero.

## Fixed implementation

Use source roles142/143/144 ×training seeds42/43/44, four arms per package,
36 neural fits. Reuse90 double-held-fold forests. Seasonal means come only
from the allowed donor library; query fold A is absent, and each donor fold B's
reference excludes A and B. Source-training seasonal fitting uses its permitted
spatial record; it is not a temporal forecasting protocol. Receiving sites have
no DOC/pH/conductance at K0.

Candidate pool20 plus zero prior, hydrology, matched current/historical validity,
aggregate inputs, source/reference transforms, initial weights,37,900 parameters,
native local readout/loss,30epochs/patience5 and ecological-memory fusion stay
fixed. Scale individual source state by one plus the frozen receiving reference,
as in first-order relative attention. Do not increase heads, layers or lookback.
Seasonal-only changes individual donor values; its retained three aggregate
readout features still use current innovations. It is not a control removing
all contemporaneous source information from the entire model.

Before fitting check reconstruction of full residual from seasonal mean plus
departure, unchanged candidates/availability, query exclusion, future-input
causality, zero head and reload. Complete one technical package and then all
nine without selecting the first package's numerical outcome. Evaluate full
and native MAE/Q90, bias, station-equal error, source/reference strata and5,000
paired station intervals. Compare full-current directly with anomaly-only and
seasonal-only. Keep old evaluated regions, external case and portable release.

The performance goal is an increment over the strongest source candidate,
not merely over the older complete model. If retained seasonal bias adds no
clear benefit, keep anomaly-only and close this mechanism. A geographical
decision follows the completed source evidence; no different winner by region
or K is assembled.
