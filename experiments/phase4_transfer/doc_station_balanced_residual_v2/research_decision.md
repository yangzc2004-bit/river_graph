# Station-balanced residual: source-development result

All nine packages completed on source roles142/143/144 and seeds42/43/44.
The cell-weighted control was retrained from each parent's initial backbone
and reproduced its saved prediction within1e−10. Neural/raw/readout inputs
match their saved parent exactly. No geographical or external query was used.

Equal-station loss does not improve the primary source-validation estimate:

| Complete procedure | MAE | Station-equal MAE | Q90 MAE |
|---|---:|---:|---:|
| Existing integrated upgrade |1.769053|2.232407|9.075420|
| Station-balanced integrated |1.771209|2.231255|9.044320|
| Existing native-only residual |1.772035|2.238860|9.053055|
| Station-balanced native-only |1.773018|2.234954|9.022023|

Seeds are averaged within each partition, then the three partitions equally.
The integrated candidate's relative gain is−0.12%, with5,000 paired station
bootstrap95% interval−0.65–0.39%; only1/3 partitions and3/9 packages improve.
The native-only gain is−0.06% [−0.73%,0.59%]. Small descriptive station-equal
and tail changes do not justify replacing the complete candidate.

The existing integrated upgrade remains the current model. Equal-station
weighting is not advanced to a new geographical or external test. The external
difference between cell-weighted and station-equal scores is not resolved by
this simple loss change. No further station-weight or tail-weight scan is run.

The initial v1 attempt was retained after a pre-training forest roundoff replay
failure. V2 preserves that attempt and uses numerical forest replay. Its control
training independently confirms that this repair did not change the matched
scientific comparison.

Next, complete two matched temporal compatibility refits and portable ensemble
delivery, then integrate the independent external and geographical evidence into
the paper. Additional model ideas must return to source roles and address a
new mechanism, rather than repeatedly evaluating this external query set.
