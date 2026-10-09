# Research decision: current-source attention across unmonitored regions

## Completed study

All25 fixed packages are complete: HUC4 regions1013/1019/0708/1030/1101,
each with seeds42–46. The learned current-source, earlier-season and fixed-prior
arms account for75 neural fits. All250 double-held-fold forest references are
reused; no additional forest was fitted. The receiving station supplies no
DOC, pH or conductance at K0. The same candidate procedure is used in every
region and at every K.

This is a retrospective replication on already evaluated ST357 geographical
roles. It does not provide independent external-basin validation. The source
development decision preceded fitting, and no geographical result was used to
choose attention size, candidate pool, region or K-specific model.

## Overall and high-DOC results

The primary population contains all valid target DOC cells. Average seeds
within each region, then weight the five regions equally. Intervals use5,000
paired station resamples, keeping a station's months and compared predictions
together. Station-equal MAE is a separately reported estimand.

| Procedure | MAE, mg/L | Station-equal MAE | Q90 MAE, mg/L |
|---|---:|---:|---:|
| Learned current-source complete |2.266609|2.493343|10.422181|
| Retained complete |2.282266|2.504735|10.519953|
| Fixed-prior attention complete |2.272573|2.499081|10.472983|
| Earlier-season attention complete |2.279778|2.502667|10.509915|
| Preceding aggregate-source complete |2.272858|2.498975|10.471173|
| Strong station-hidden trees |2.344885|2.547670|10.520142|
| Trees with aggregate current-source features |2.314529|2.516639|10.481605|
| Older complete model |2.360283|2.556920|10.632267|

The complete current-source attention procedure improves the **actual retained
complete model by0.686% [0.031%,1.397%]**, with four of five region averages
improving. Its Q90 MAE improves **0.929% [0.431%,1.511%]**, with all five
directions positive. This is a small reproducible performance increment.
The desired5% overall improvement over both primary references is not reached.

The gain over bare strong trees is3.338% [0.391%,6.137%], positive in all
five regions. The gain over trees already receiving aggregate current-source
features is2.070% [-1.188%,5.171%], positive in three regions. Those trees
share the data origin but do not receive the attention operator's individual
donor hydrological keys; this contrast does not isolate architecture under
identical raw inputs. The3.969% improvement over the older complete model
includes earlier retained improvements and is not this branch's incremental gain.

## What the matched controls show

Against the fixed ecological-prior attention arm, overall improvement is
0.262% [-0.059%,0.590%], with three positive regions. Against earlier-season
attention it is0.578% [-0.125%,1.336%], with four positive regions. These
overall intervals do not establish an adaptive-allocation or timing advantage.

The Q90 contrasts are clearer: learned current-source attention improves the
fixed-prior arm by0.485% [0.142%,0.908%] and the earlier-season arm by0.835%
[0.309%,1.440%], positive in all five regions. The preceding aggregate-source
complete procedure is improved by0.468% [0.117%,0.893%] on Q90, again five
positive regions. Thus the reproducible allocation result is concentrated in
high-DOC reconstruction.

The native neural prediction has MAE2.265679, versus retained neural-only
2.265595: -0.004% [-0.425%,0.361%] gain. Its Q90 error improves0.921%
[0.438%,1.415%], with five positive regions. Native Q90 improvements over
fixed-prior and earlier-season arms are0.481% [0.139%,0.865%] and0.757%
[0.244%,1.286%], respectively. The full-procedure overall improvement includes
validation-selected ecological-memory fusion; it is not an established overall
neural-only gain. Both component and complete contrasts remain in the tables.

## Regional and adaptation behaviour

| Held-out HUC4 | Learned current complete MAE | Retained complete MAE | Gain |
|---|---:|---:|---:|
|1013|5.600284|5.626815|0.472%|
|1019|1.640676|1.619163|-1.329%|
|0708|1.723738|1.745890|1.269%|
|1030|1.252436|1.260627|0.650%|
|1101|1.115911|1.158835|3.704%|

There are62 improved and42 worsened station averages across104 receiving
stations. Station-equal overall MAE improves0.455%. Complete signed bias is
-0.665207mg/L versus retained-0.649449mg/L; the overall MAE improvement does
not remove underprediction. Every region's result is retained.

The K-curve query always excludes the same five candidate support cells,
including K0. Its K0 is therefore distinct from the all-observed primary K0.

| K | Current attention complete MAE | Retained complete MAE | Gain |
|---|---:|---:|---:|
|0|2.280020|2.296495|0.717%|
|1|2.149819|2.160459|0.493%|
|3|1.962479|1.987946|1.281%|
|5|1.891899|1.909790|0.937%|

K0/K3/K5 paired overall intervals are[0.047%,1.445%],
[0.814%,1.684%] and[0.335%,1.400%]. K1's[-0.901%,1.573%] crosses zero.
Q90 gains over retained complete are positive with supported paired intervals
at every K. The aggregate-source trees still have lower K3/K5 MAE point
estimates1.953828/1.871110; no per-K winner is substituted into the procedure.

Q90 classification remains a separate challenge. Region1013 has904 unique
tail cells and mean recall0.9843. Region0708 has66 cells and recall0.0121;
1019 has43 cells and zero recall. Regions1030/1101 have15/11 cells, zero
recall and unstable flags. Repeated seeds do not multiply these ecological
sample counts. Lower tail MAE does not establish reliable high-DOC detection.

## Allocation and environmental diagnostics

The learned current branch's mean zero-innovation prior mass is0.671,
versus0.761 for the fixed-prior arm and0.796 for earlier-season attention.
Mean normalized entropy is0.573,0.659 and0.524, respectively. The learned
branch reallocates information without collapsing all mass to a single donor.
These are information-allocation diagnostics, not physical transport coefficients.

Source-support, ecological-novelty, source-distance and hydrological-availability
strata are saved, including their contributing-region counts. Improvement also
occurs in the no-current-donor group, so the entire complete gain cannot be
assigned to donor values alone. The zero-hydro-channel stratum worsens slightly;
ecology/source-distance summaries have differing region support and remain
descriptive. Their outcomes do not select additional regional models.

## Verification and next research

All25 candidate arrays, double-held exclusion roles, initial weights, complete
predictions, support outputs and attention diagnostics replay bitwise. The
primary and fixed-query tables,5,000 station intervals and vector figures have
been generated. The PNG was actually inspected. Numerical analysis reuses
that completed full replay after rechecking every completion identity; the
execution note records this runtime optimization. Fitting code and products
were not changed. The current workspace passes988 tests with two skips,
Ruff and the historical artifact audit. The completed fitting version itself
had985 passing tests, before the three new relative-value tests were added.

Keep this candidate and its small overall/high-DOC increments as research
results. The existing portable release and previously evaluated external basin
remain intact during the next source experiment.

The next experiment tests **concentration-relative source departures** within
the same attention branch: log1p source OOF residuals, donor-season centering,
and conversion through one plus the frozen receiving environmental prediction.
Its plan was saved before opening these numerical geographical results. It
changes individual donor value units, while retaining the native loss, old
aggregate readout, parameter count, source roles and matched controls. Develop
on142/143/144 again; do not fit a geographically selected correction or alter
the previously tested external products.
