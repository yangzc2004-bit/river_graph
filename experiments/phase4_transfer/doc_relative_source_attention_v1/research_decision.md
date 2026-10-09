# Research decision: transfer relative source DOC departures

## Completed experiment and main result

All nine source-development packages are complete: partitions142/143/144 and
seeds42/43/44, three matched arms and27 neural fits. Ninety double-held-fold
forest references were reused; no additional forest was fitted. The receiver
provides no DOC/pH/conductance at K0. These are source-validation development
results, including validation checkpoint and fusion selection.

Change only the individual donor value representation from native concentration
departures to season-centered OOF log1p departures. Convert the weighted state
back through one plus the frozen receiving reference. Native tail-weighted
training, old aggregate innovation inputs, ecological self encoder/GRU,
37,900 parameters and30-epoch/patience5 budget remain the same. This is
distinct from the earlier unsuccessful whole-model log-concentration objective.

| Complete procedure | MAE, mg/L | Q90 MAE, mg/L | Station-equal MAE |
|---|---:|---:|---:|
| Relative-current learned |1.748102|8.932044|2.205626|
| Relative-current fixed prior |1.748209|8.940732|2.206211|
| Relative earlier-season learned |1.768525|9.073935|2.232689|
| Native-current learned |1.756645|8.980932|2.216040|
| Retained complete |1.769053|9.075420|2.232407|
| Strong station-hidden trees |1.866362|9.394458|2.330711|

Average seeds within each source partition, then weight partitions equally.
Five thousand joint paired station draws preserve months and repeated task
appearances of a station. Relative-current complete improves the actual
retained complete procedure by **1.184% [0.610%,1.839%]**, positive in all
three partitions, all three seed averages and all nine packages. Its Q90
gain is **1.580% [0.744%,2.864%]**, also positive in all nine packages.

Against native-current attention, its overall gain is **0.486%
[0.166%,0.827%]**, positive in every package and partition. Q90 gain is
0.544% [0.220%,1.169%], positive in two partitions and six packages.
Relative departures therefore add a useful source-development increment to
the preceding absolute-departure branch. Complete signed bias improves from
retained-0.638608 to-0.605140mg/L but underprediction remains.

The6.336% [3.715%,9.195%] gain over strong trees combines the retained
architecture and this new increment; it is not the isolated relative-value
benefit. Saved source-enriched trees receive aggregate innovations rather than
individual donor hydrological keys, so they are not an identical-raw-input
architecture control.

## Mechanism interpretation

Relative-current improves the matched earlier-season complete arm by1.155%
[0.546%,1.806%], positive in all nine packages. Its gain over the relative
fixed-prior complete arm is only0.006% [-0.530%,0.498%], with one positive
partition; Q90 gain0.097% [-0.290%,0.561%] is likewise unresolved. The
source evidence supports current relative information, while it does not
establish learned allocation as the source of the complete improvement.

Neural-only relative-current improves retained neural-only by1.149%
[0.522%,1.851%], positive in all nine packages. Against native-current
neural-only the gain is0.525% [0.223%,0.848%], also nine positive packages.
Its Q90 gains in these contrasts are1.668% [0.788%,3.029%] and0.295%
[0.008%,0.735%]. This increment is present before memory fusion, unlike
the preceding geographical study's nearly unchanged neural-only overall MAE.

Current relative attention assigns mean zero-innovation prior mass0.379,
versus0.662 for fixed prior and0.604 for the earlier-season arm. Mean
normalized entropies are0.801/0.897/0.776. Allocation does not collapse.
The receiving-concentration groups use tertiles of source OOF reference
predictions, not receiving truth. All three group point estimates improve
over native attention; their comparisons remain descriptive. Mean matched
donor count is3.23, with available current donors in84.23% of selected cells.

The value conversion is a first-order log1p-to-native residual transformation.
It does not impose concentration conservation or identify a physical process.
The source-only seasonal statistics are frozen during retrospective monthly
inference; this is not a pre-month forecast.

## Verification and research choice

All nine packages replay bitwise: libraries/candidates, double-held exclusions,
initial weights, neural and memory products and attention diagnostics. Units,
zero fusion, future-input isolation and save/reload tests pass. Five thousand
station draws and actual PNG inspection are complete. The workspace has988
passing tests and two skips; Ruff and the historical artifact audit pass.
All old products and large local fitting caches are retained.

Proceed to a separate fixed geographical replication of **all three relative
arms**, comparing each complete procedure with the retained release and saved
native-current attention. The consistent overall source increment justifies
this replication; it does not yet establish a geographical improvement.
Keep five existing HUC4 regions and seeds42–46, reuse the250 pair references,
and retain all K outcomes. The fixed-prior arm remains a scientific comparator,
not a per-partition winner. No additional value conversion, attention dimension,
region or K selection is made from these source results. The portable release,
previous external evaluation and main manuscript remain at their evaluated
versions during confirmation.
