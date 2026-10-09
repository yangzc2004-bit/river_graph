# River geometry and the timing of DOC export

Connect channel path diversity, confluence location and shared downstream
routing to concentration peaks, timing, spreading and carbon export. This
version adds varying water flux to the previous controlled routing experiments
and measures carbon budgets in the three existing hourly outlet archives.

## Geometry experiment

Use all 121 previously screened independent tributary pairs (38 receivers,
17 connected monitoring systems), with their measured directed paths and
area-proxy flow shares. Normalize distance by the square root of basin area;
scenario time is dimensionless, not an estimated channel travel time.

Route water and carbon flux separately and recover concentration as carbon
flux divided by water flux. Sources share baseline concentration 5 and a
Gaussian concentration increment of 1; water rises by 1 or 3 times its
baseline, with Gaussian SD 0.15 or 0.30. Input carbon and total water are fixed
within every paired structural manipulation. The four contrasts are:

1. Actual path dispersion versus equal arrivals at the same weighted mean
   path. This changes path diversity alone.
2. Actual arrivals versus source pulses timed to compensate for path delays.
   Both water and concentration at each source are shifted together.
3. A short versus long shared segment, at unchanged total source paths. An
   imposed causal gamma travel-time distribution in the shared segment has
   shape 4 and mean equal to shared length; branches have deterministic travel.
   Include deterministic shared routing as the junction-position null.
4. Conservative shared routing versus imposed first-order processing of the
   additional DOC pulse, at rate 0.25 per scenario time unit. This is a process
   sensitivity, not a measured DOC removal rate. The baseline DOC is conserved.

Compare peak concentration increment, concentration/flow peak offset, carbon
excess-flux centroid and SD, and retained additional carbon. Receiver means
precede 5,000 connected-system bootstrap draws. Keep all forcing scenarios;
the narrower, threefold water pulse is a descriptive primary display.

## Observed export

Reuse the frozen common-hourly 20% prominence event inventories. Keep every
timing-eligible event, including nonpositive DOC responses. Integrate only
complete exact joint hourly records from flow-defined start through the saved
response end. No new filling or temporal smoothing. An incomplete interval
remains in the inventory without a complete budget.

DOC [mg/L] times discharge [m3/s] gives carbon flux [g/s]. Integrals use observed
adjacent hourly endpoints and report bounded-window kg C and kg C/km2, not
uncensored full-storm exports. Separate total export, the constant-antecedent-
concentration counterfactual, signed concentration-induced export, and positive
concentration-induced export. Compare carbon and water shares exported after
the flow peak on the same interval. Report both flow-return and response-end
windows, so an arbitrary follow-up does not silently define total export.

Within each catchment, use month-block intervals and exploratory rank
associations between peak, width, lag and export, adjusting for recorded flow
volume and duration where feasible. These event associations cannot estimate
between-form effects: only Kervidy has a validated rooted vector network, and
repeated events are not independent river geometries. Keep raw archival
processing limitations and the small Bouleau sample visible.

## Output

New reproducible analysis, English research conclusions and English/Chinese
scientific figures under this versioned directory. Previous outputs remain
intact. No training or reconstruction-model tuning is required.
