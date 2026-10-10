# Dynamic river-message research decision

## Result and model decision

The protected dynamic river extension is implemented and the full comparison
is complete: three source-role station partitions, three seeds and five neural
arms per package, **45 fitted river branches**. The existing ecology encoder,
observation-aware GRU, environmental trees, similarity attention and integration
remain frozen. Real river attention is an additional sparse upstream operator.

The primary model reduces DOC MAE from **1.723002 to 1.717226 mg/L**, a
**0.3353%** improvement. Its 95% whole-station interval is **−0.0392% to
+0.6793%**. This is a small development gain, not an established overall native
MAE improvement. Keep the complete current model as the released predictor and
retain this branch as the river candidate for further development.

The clearest useful refinement is protecting the existing nonriver prediction.
Every cell without a usable upstream DOC observation retains its complete-base
prediction bitwise. Unlike the earlier jointly refitted river model, adding
river learning cannot degrade those cells through changes to local parameters.

## Fixed comparison

| Model | Equal-partition MAE mg/L | Q90 MAE mg/L | MAE gain versus complete current |
|---|---:|---:|---:|
| Complete current model | 1.723002 | 8.895024 | — |
| Static same-month upstream pooling | 1.718552 | 8.862655 | 0.2583% |
| Dynamic same-month upstream messages | 1.718173 | 8.854349 | 0.2803% |
| Dynamic upstream messages, lags 0/1/3 | 1.717226 | 8.852561 | 0.3353% |
| Support-matched dynamic upstream | 1.717396 | 8.860705 | 0.3254% |
| Matched non-upstream dynamic control | 1.721678 | 8.891393 | 0.0769% |
| Retained matched environmental trees | 1.866362 | 9.394458 | — |

The dynamic lag model is 7.9907% better than the tree baseline, but most of that
gap already existed in the complete model. The river addition itself contributes
the 0.3353% increment above; the 7.9907% must not be attributed to river learning.

Primary Q90 MAE improves **0.4774%** [−0.0938%, +1.1610%]. Log1p MAE improves
**0.4501%** [+0.0783%, +0.8659%]. The transformed-error contrast is more stable
than the native-concentration and high-DOC contrasts. Q90 is the source-training
threshold for each package, not a threshold chosen from receiving outcomes.

All three partition means improve: **0.0869%, 0.5640%, 0.3872%** for 142, 143,
144. Eight of nine seed packages improve; one retains the zero correction.
Station-equal MAE improves about **0.205%**. Seed repetitions are not additional
ecological observations. There are 140 distinct receiving stations and 7,897
distinct station-months, with 8,857 partition-cell occurrences.

## What the controls establish

Dynamic lag messages improve only **0.0771%** over static same-month pooling
[−0.0925%, +0.2292%], and **0.0551%** over dynamic same-month messages
[−0.0346%, +0.1466%]. These comparisons do not establish that learned allocation
or the additional lag slots are necessary for the native-MAE gain.

The matched upstream arm improves **0.2487%** over matched non-upstream sources
[−0.0304%, +0.5327%] on native MAE. Its log1p contrast is **0.3083%**
[+0.0487%, +0.5998%]. Real upstream pairing has a small positive development
signal; its distinct native-scale contribution remains unresolved. The
support-matched upstream-versus-base contrast has a positive native-MAE interval
[+0.0166%, +0.6459%], but that is a different contrast from real versus
non-upstream and cannot replace it.

Matching retains the same physical path slots and intersects real/control
observation validity, with the same maximum observation age. Donors are matched
on drainage area and availability, without concentration values. Controls are
outside all mapped upstream paths and COMID aliases. The unrestricted primary
model uses all permitted real observations; the matched arms answer the
connectivity question on their shared availability.

## Where prediction changes

With usable upstream observations, native MAE improves **1.2534%**
[+0.0670%, +2.7607%]: 50 distinct stations and 2,836 distinct cells. Without
usable observations, every prediction is unchanged: 122 distinct stations,
5,164 cells. These station sets overlap across months and partitions.
The observation-supported fraction is 22.73%, 21.74% and 52.48% across the
three partitions. Limited supported coverage dilutes any upstream benefit.

The <=50 km path group improves **2.4934%** [−0.5581%, +5.1348%], based on
23 stations. Paths 50–200 km improve 0.1849%; paths over 200 km change −0.1605%
with only six stations. The nearest eligible path is a geometry descriptor;
the group's months can include periods without a usable source observation.

The 7–12-month observation-age group has a positive estimate of 3.9924%, but
only 15 stations and 52 cells. This remains a small descriptive subgroup and
does not establish measured travel time or a general long-memory benefit.

On supported months, mean attention allocation is approximately 0.38 to current
source states, 0.27 to previous-month states, 0.20 to three-month-old states and
0.15 to the zero-message alternative. Allocation does not collapse to one lag.
Some retained observations repeat across slots; lag mass is information
allocation, not a physical travel-time distribution.

## Architecture implemented

```
environmental trees + ecological encoder + observation-aware GRU
                 + ecological-source attention + fixed integration
                                  |
                                  +---- complete current DOC (frozen)
                                  |
                                  +---- queried GRU state / ecology / hydro
                                             |
real upstream paths + observed source departures + age + causal hydro history
                                             |
                                 dynamic two-head river attention
                                             |
                                     zero-initialized log correction
                                             |
                           DOC = max(0, expm1(log1p(base) + correction))
```

The new attention query has 90 features, including the existing 64-dimensional
GRU state. Edge/lag features have 36 columns: eight path descriptors, source
hydro, its preceding-month change, source-versus-receiver hydro, observation age
and the lag identity. The branch allocates 9,064 parameters. The static arm has
the same allocated network, but uniform weights and reliability; only its
six output coefficients provide an effective fitted mechanism. This difference
is disclosed rather than described as identical effective model complexity.

Each arm receives at most 30 epochs with patience 5. There were 499 total
training epochs over the 45 branches; selected epochs range 0–27. Epoch 0 is
the exact complete predictor. Model selection uses the declared source
validation raw MAE, so positive per-package changes are partly selected. These
intervals characterize the development panel and do not remove selection
optimism. The retained base was also historically selected on these roles.

Training receiver folds never enter their own observed donor libraries.
Environmental donor residuals use references excluding both query and donor
folds. Complete-model training predictions are fitted source outputs, **not
complete-model OOF predictions**. No receiving DOC, pH or conductivity value is
used as an inference input. Lagged source-hydro reads use the candidate month
and its preceding month only. The fixed source preprocessing/library remains an
offline source fit; this is not a chronological rolling-origin refit study.

## Next research direction

Keep the protected architecture. Do not enlarge attention or graph depth merely
to amplify this result. The next useful experiment is to expand the dynamic
river state beyond observed concentration departures: transmit causal upstream
hydro/environment representations when upstream DOC is absent, while marking
estimated states separately from observations. Compare observed-only messages,
hydro-state-only messages and their combination on source development roles.
Maintain the same local predictor and real/non-ancestor controls.

If that increases the incremental signal, evaluate one complete fixed candidate
on held geographical roles. The present results are ST357 source-role model
development; **no geographical or independent external-basin confirmation of
this new branch has been run**. Existing historical geography/external results
remain evidence of their own model versions. No new automation was started.

## Reproducibility and verified products

- Nine completed parquet/sidecar packages; 45 local fitted river checkpoints.
- Rebuilt all input banks and frozen queries from bound retained source files.
- Exact stored-prediction replay for all 45 branches; exact silent/no-support
  fallbacks; normalized attention and matching contracts verified.
- Independent arithmetic reproduces primary MAE, Q90, log1p error and all
  5,000-draw primary station-bootstrap interval endpoints.
- Two figures in PNG/PDF/SVG, inspected after correcting overlapping lag labels.
- Full pytest after the archive-availability guard: **1,437 passed, 3 skipped**.
  Ruff and historical artifact audit pass. Existing warnings are retained.

The first full test run exposed an unrelated portable geography test whose
guard checked a completion marker although its forests had been archived.
The guard now explicitly skips when any required local fit asset is absent;
the initial failure is recorded in pytest.log, and no numerical model code or
historical result was altered. No archive was silently restored. Consumed
retained files were checked individually against their original completions;
this is not a claim of full restoration of historical forest assets.

Large fitting-input caches and new weights remain local. Old result directories
and protocol endpoints are unchanged. Reproduction commands are in README.md.
