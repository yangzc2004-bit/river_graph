# Dynamic river messages for unmonitored DOC reconstruction

## Research question

Can current hydrology, observation age and causal upstream history improve the
complete DOC model while preserving its predictions where river observations
are absent? This version returns to neural modelling after the river morphology
and flow-memory studies. Historical flow-memory diagnostic gains are not gains
of this network. Monthly lag slots represent available information, not measured
travel times.

## Architecture

Retain each fitted ecological encoder, observation-aware GRU, source similarity
attention, environmental tree base and ecological integration. Freeze all of
them. The GRU state, ecology, current daily-derived hydrology and its preceding
monthly change query a two-head, 32-dimensional sparse river operator. Real
upstream paths supply environmental-OOF log-DOC departures, age, path attributes,
source hydro at the candidate month and its preceding-month change. Source
departures can be retained for at most 12 months. Invalid candidates are hidden;
a zero-message candidate is always available. The bias-free output starts at
zero. Final DOC = max(0, expm1(log1p(complete DOC) + river delta)). Without an
eligible observation, output is bitwise the retained complete prediction.

## Fixed experiment

Source-role partitions 142/143/144, seeds 42/43/44. Receiving source-validation
stations have no DOC, pH or conductance input. K0 evaluates all saved validation
DOC query cells. There are five trained arms per package (45 fits):

1. Static same-month upstream pooling.
2. Dynamic same-month upstream attention.
3. Dynamic upstream attention over lags 0/1/3 (primary candidate).
4. Dynamic lag attention with availability-matched real upstream observations.
5. The same dynamic lag operator with matched non-ancestor sources.

Retain complete current DOC and the matched strong environmental tree baseline
without refitting. Each neural branch has at most 30 epochs, patience 5, two
heads, dimensions 32, Adam learning rate .001, batch 512 and source Q90 tail
weight 2. Source-validation raw MAE selects epoch, including the zero correction
at epoch 0. All five arms and all packages are reported. No regional or K-wise
winner is assembled. The static ablation has the same allocated network but
fixed uniform allocation/reliability; effective trainable mechanisms differ.

## Training and control meaning

Each receiving training fold is excluded from its donor library. Environmental
references additionally exclude the donor fold. Matching uses source drainage
area and observation availability, never DOC values. Matched real/control
neighbourhoods have identical path slots, common validity and maximum age;
non-ancestor checks include all mapped ancestor paths and COMID aliases.
The unrestricted primary candidate retains all actual upstream observations;
its gain versus a matched control alone does not isolate connectivity. The
matched-upstream versus matched-nonupstream contrast answers that question.

The frozen complete model was already fitted on source labels and selected on
these validation roles. Complete-model training predictions are fitted outputs,
not OOF predictions. Environmental donor residuals are double-held OOF. This is
a source-development comparison, not independent geographical/external
confirmation. Prior seen geographical/NEON outcomes are not used for training
or selecting this version. Missing archived forests are unnecessary: verify
consumed retained inputs/checkpoints against their original completion hashes.
Old completions and archived files are not edited.

## Readout

MAE, RMSE, R2, log1p MAE, source-Q90 tail MAE and bias. Seeds are averaged within
partition, then three partitions equally. Use 5,000 paired whole-station
bootstrap draws jointly across partitions. Report station-equal MAE and per-seed
and per-partition direction. Diagnostics: observed support/no support, fixed
path bands <=50 / 50–200 / >200 km, age <=1 / 2–6 / 7–12 months, lag mass,
entropy and correction size. No-support predictions must equal the complete
base. Repeated seeds do not create new ecological samples.

After all fits, independently replay saved products, verify hidden-label and
future-information contracts, inspect figures, and write a research decision.
This version stays in its own directory and preserves all earlier experiments.
