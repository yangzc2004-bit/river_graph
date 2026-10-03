# Selective DOC residual learning across stations

The native DOC residual can improve high concentrations, but does not yet
combine that sensitivity with the ordinary-concentration performance of the
current overall model. Extending optimization does not remove this trade-off.
We now change the training objective while keeping the representation fixed
to the latest final-self-plus-ecology tuning configuration.

Source/validation diagnosis motivates two separate questions. Source records
are uneven: the most observed 10% of stations provide approximately 38–40% of
cells. Tail reconstruction remains strongly downward biased, while false-high
predictions can increase even as average ordinary overprediction falls. A
blanket prediction gate is therefore not introduced. We compare regression
costs and the weight assigned to source stations independently.

## Four matched losses

Let e = max(0, source-OOF context + neural delta) - source truth. Q90 is computed
from source-training labels. Uniform observed-cell order is unchanged.

1. **tail2:** current control, w=1+1(y>=Q90), L=sum(w*abs(e))/sum(w).
2. **mae:** ordinary MAE, w=1, without additional overprediction penalty.
3. **selective:** retain tail2 weights and add
   0.5*1(y<Q90)*relu(e) to each cell's weighted error, retaining sum(w) as
   denominator. The penalty applies to final prediction error, not delta.
4. **station:** retain twofold within-station tail emphasis, with
   w_i=tail_weight_i/sum(tail_weight_j for j at the same source station).
   L=sum(w_i*abs(e_i))/sum(w_i). Every source station has equal total weight
   after tail weighting. No overprediction penalty is combined with this arm.

The truth-defined group is used only in source training. Inference needs no
group label, classifier or target-query observation. Ordinary overshoot costs
1.5 times undershoot in the selective arm; high-DOC slopes remain twofold.
Loss normalization uses the fixed complete-source mean, rather than a random
minibatch denominator.

## Fixed training and products

Use three existing station partitions (142–144) and seeds42–44: 36 fits.
Retain the original expert initialization; final self/ecological layers and
GRU/decay train with their current learning rates, alongside the zero-initialized
concentration head. Maximum epochs60, patience5, dropout off, empty-edge self
path, source-fold-hidden inputs, source OOF bases, and 12-month causal windows
are unchanged. Overall unweighted validation query MAE selects epoch and scale,
including the exact context-only fallback.

The source forest, support representation and ecological memory remain fixed.
Source-validation support/mixing selection is repeated for each new base using
the same grids. Query cells and reserved retrospective supports are unchanged
at K=0/1/3/5. Carry the previous direct and integrated encoder predictions and
the earlier overall ecological-affine model unchanged as references.

## Research comparisons

Compare each loss against tail2 in direct and integrated products at K0/K5,
and compare station versus selective to distinguish sampling from overshoot
cost. Retain all K curves and every failed model. Report overall/raw/log error,
Q90 error and bias, ordinary error and bias, false-Q90 rate, Q90 recall, station
gain/harms and source-validation choices. A reduction in false alarms alone
does not establish improvement if true high-DOC detection also deteriorates.

Use the existing equal-partition estimator and joint whole-station bootstrap.
These are previously examined same-cohort development partitions; the new
loss definitions are set before the new models are evaluated. Earlier paper
endpoints and experiments remain unchanged.
