# River-form mechanisms in observed tributary signals

## Question and basis

Which operations of a river network can modify DOC fluctuations: integration
of unequal tributaries, asynchronous tributary changes, or unequal path timing?
Can the existing observations distinguish these operations from channel loss?

This is a diagnostic extension of the completed real-morphology, routing and
flow-response studies. Their results have already been seen. Keep the 59
connections and identical 3,026 common connection-months from the observed
transport study; do not select rivers using an effect size or refit a model.
The whole-river outline classes remain fixed. A monitored pair describes two
branches, not every tributary in its receiver's complete network.

## 1. Tributary integration

Project native DOC in both sources, their area-weighted mixture and the outlet
onto the same intercept, calendar sine/cosine and linear-year design, on the
same dates. Native scale preserves the exact linear mixing identity, unlike
separately transformed log concentrations. Save fitted calendar coefficients.

Let a and b be source anomalies, w the fixed source-area share, V the variance,
and rho the source Pearson correlation. Use the population variance consistently.

Vref = w Va + (1-w) Vb

Vmix = w^2 Va + (1-w)^2 Vb + 2w(1-w) Cov(a,b)

Vref - Vmix = w(1-w) [(sqrt(Va)-sqrt(Vb))^2
                          + 2 sqrt(Va Vb)(1-rho)]

Report the two contributions: amplitude imbalance and asynchronous changes.
Equalize both source amplitudes as a counterfactual; the remaining reduction is
2w(1-w)(1-rho). Compare this with balanced branches (w=0.5) and synchronized
changes (rho=1). These are identities/scenarios, not significance tests of an
independently estimated river effect. Compare measured outlet variability with
the mixture; any difference can include ungauged inputs, changing flow shares,
sampling and channel processes, and is not automatically DOC removal.

## 2. Unequal arrival opportunity

Reuse the original current/previous-month interpolation at fraction=1 and its
original maximum-path scale. Decompose branch-specific minus mean-delay input:

gap = w(1-w)(pathA-pathB)/max_path * [(Aprevious-Anow)-(Bprevious-Bnow)]

Report gap in mg/L and |gap|/(1+same-month mixture), with fixed descriptive
thresholds 1% and 5%. Equal paths, parallel source changes or a dominant branch
remove this opportunity. Apply the saved mean-delay outer-fold calibrator to
both inputs without changing coefficients; this isolates input sensitivity,
not a new fitted model or a new test of prediction superiority. Neither the
path scale nor interpolation fraction estimates travel time or physical speed.

## 3. Channel-process identifiability

Attach measured positive current-month discharge at both sources and receiver,
respecting x_mask. Count coverage and inspect two availability conditions:
source drainage coverage >=0.8 and (Qa+Qb)/Qreceiver within [0.8,1.2]. This
screen was inspected before this plan: about 101 connection-months in five
receivers. Report unique receiver-months separately. Report flow-weighted
concentration mismatch only as apparent monthly departure, never as a measured
retention rate. Monthly DOC and monthly flow cannot identify contemporaneous
loads, intervening source/sink budgets or event travel times.

## Aggregation and outputs

Equal dates within a connection, connections within receiver, then receivers.
Use 5,000 whole connected-monitoring-system bootstrap draws (seed 42) and HUC4
sensitivity. Keep single-system class intervals unavailable. All class summaries
describe the observed population; elongated and sparse classes do not have
independent replication here. Do not rank entire river forms from these pairs.

Deliver exact decomposition tables, observed anomaly products, arrival-input
sensitivity, water-budget availability, English/Chinese scientific figures,
and a research decision that links shape to operations and identifies the next
measurable structural question. Preserve prior results and models.

## Scale and influence diagnostic added after the native summaries

The initial native summaries show a difference between geometric SD ratios and
arithmetic variance ratios, including a large outlet/source ratio at 03303280.
Retain every connection. Add calendar-adjusted log1p SD ratios on the identical
dates and leave-one-monitoring-system summaries to expose scale/influence.
These are supplementary diagnostics after seeing native results; they do not
replace the native mixing identity or establish physical channel loss.
