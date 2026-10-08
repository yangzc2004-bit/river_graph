# Kervidy: actual river geometry and observed flow–DOC response

Date: 2026-10-09

## Purpose

Connect the newly acquired corrected optical DOC record to its actual monitored
outlet, publicly mapped river network and discharge companion. This is a single
catchment process case for the river-form study. It does not supply independent
elongated-versus-broad replication.

## Geometry and observations

Use the public GeoSAS Naizin river layer and AgrHyS outlet metadata. Inspect the
delivered geometry and attributes, locate the gauge in the mapped network and
distinguish upstream catchment geometry from downstream context. Use a published
catchment boundary if available; do not orient an undirected export by assuming
all reaches near the gauge are upstream. Retain the required map attribution.

Retrieve public discharge stream 2 through the catalog-linked SensorThings
service, scoped to corrected DOC availability (October 2020–September 2023).
Keep the native UTC clock and rating-curve method. Convert dm³/s to m³/s. A
one-minute distributed grid is not automatically a one-minute independent field
measurement. Query a quarter-hour subset if the public service supports it;
otherwise document the available bounded retrieval. Preserve missing, zero and
negative values separately. Do not interpolate DOC or flow to create matches.

## Joint records and event display

Report exact UTC matches and, separately, nearest flow matches within two
minutes, without borrowing across longer gaps. Preserve source timestamps and
match offsets. No time shift is fitted from DOC.

If continuous flow is acquired, select the largest observed positive flow in
each calendar year with corrected DOC, using flow alone. Display a fixed
seven-day window (three days before/four days after), retaining every attempted
window and boundary/missingness status. Require at least 90% occupied nominal
15-minute DOC and flow bins and no gap over one hour for a dense joint window.
This is a chronological process illustration, not an independent storm catalog.
Partial 2020 and 2023 observation years remain labelled partial. A peak lag or
width is only interpreted after checking that the rise and recovery are present;
record coverage itself does not establish complete events.

## Outputs

Save compact geometry and joint-record inventories, the actual network/gauge
map and flow-selected DOC/flow panels, plus an English research decision. Keep
raw downloads locally. Retain the earlier external-record audit and all previous
river-form experiments. No new model training or DOC-dependent source selection.
