# Sampling resolution on measured river branches

## Scientific question

Do the archived DOC samples resolve tributary timing and downstream buffering,
or does the apparent monthly signal combine measurements taken on different
days? Keep river structure central: independent branches organize relative
arrival; the common trunk can translate a signal, while storage or changing
flow could also broaden it.

Retain the preceding 59 branch pairs, 22 receivers, 11 connected monitoring
systems and 3,026 connection-months. This follow-up has seen the previous
monthly results and the preliminary sampling-date audit. It is exploratory.
No model is retrained, and no new river-shape classification is introduced.

## Measurements and comparisons

1. Extract the original uncensored, mg/L DOC records using the existing WQP
   policy. Reconcile their row-weighted monthly means with the frozen dataset
   and the previous connection-month table. Preserve laboratory-result counts,
   sampling activities, unique calendar days, sampling times and time zones.
2. Average replicate results within an activity. Metadata for an activity must
   agree. Convert known local times to UTC using the explicitly reported fixed
   time-zone abbreviation; retain missing times without imputing them.
3. For each connection-month select one activity per station by minimum
   calendar-day span, then minimum UTC span where all times are known, total
   distance to the receiver date, dates and activity identifiers. Selection
   never reads DOC values. Same-day sampling is not assumed to coincide with
   water travel or guarantee upstream-before-downstream order.
4. Report all day-span cuts {0, 1, 3, 7, 29}. At each cut compare original monthly
   means with selected activities on exactly the same months and pairs. Require
   at least 24 common months per pair for the existing calendar-adjusted
   covariance/variance analysis. Report source correlation, mixing variance
   reduction, its asynchrony component and outlet/mixture log-SD ratio. These
   are descriptive concentration signals, not DOC removal rates. Also separate
   changes from month selection and changes from within-month aggregation.
5. Average pairs within receiver, give receivers equal weight, and bootstrap
   whole connected monitoring systems 5,000 times. Save paired differences and
   denominators at every cut; do not select a preferred cut from DOC results.
6. Join actual daily discharge at each sample's local date and the receiver's
   local sample date, with no interpolation. Reuse the established first-series,
   duplicate and finite-value policy. Report availability, positive-flow
   availability and observed flow change across the sampling offset separately.
   Local calendar dates are those used by the daily NWIS archive, not UTC dates.
7. Count months with at least three distinct DOC sample days at all three
   stations, including the five connections with mapped common-trunk storage.
   Check whether these data can resolve an event sequence before estimating
   propagation times or downstream widening. Three days per month is a data
   opportunity screen, not proof that an event was observed.

## Products

Save activity-level evidence, station cadence, date-selected triplets, daily
hydrology coverage, all date-cut comparisons, bilingual scientific figures and
an English research decision. Choose illustrated sampling windows from date
coverage and river geometry, never DOC response. Plot DOC as observed points,
not a daily interpolated concentration curve. Preserve all preceding results.
