# Geographical replication of current source availability

## Scientific question and fixed candidate

Can using contemporaneous donor DOC without an earlier-year availability
requirement improve reconstruction in a different river region? The source
study `doc_current_availability_attention_v1/research_decision.md` selected this
mechanism before this geographical fitting. Its isolated source-validation MAE
gain is 0.804% versus matched full-source attention; accumulated gain versus the
retained complete model is 2.603%. The current source-value and seasonal-only
arms have the same expanded validity. Do not infer geographical success from
source validation.

## Design

- Five fixed HUC4 regions: 1013, 1019, 0708, 1030, 1101, with the existing
  cyclic-next validation region and all remaining stations as sources.
- Five fixed training seeds: 42, 43, 44, 45, 46.
- Receiving K0 has no DOC, pH or conductivity inputs or their histories.
- Every finite permitted current source OOF residual is usable, without
  requiring a prior-year observation. Source seasonal statistics use allowed
  training data. The double-held query/donor reference exclusion is retained.
- Keep 20 ecological candidates plus a zero prior, two heads of dimension 32,
  the same daily-flow keys, 12-month GRU query, original matched aggregate
  descriptors, first-order concentration conversion and native prediction
  loss. There are 37,900 parameters; 30 epochs and patience 5.
- Fit actual current source values and individual seasonal-only source values:
  two neural fits per package, 50 total. Reuse all 250 pair references; no
  forest refit or new data. Carry matched full-source, anomaly-only, retained
  complete and tree predictions unchanged.
- Retain the complete candidate for every region and K, with native-only as
  an ablation. No region-specific or support-count-specific model choice.

## Execution and analysis

First execute 1013/42 to verify technical fitting, saved state, candidates,
source roles and product replay. Do not use this test score to modify the
model. Retain this completed package and finish all 25 fixed packages.
Existing fitted versions and point predictions remain untouched.

Primary K0 includes every valid DOC test cell. The separate K=0/1/3/5 curves
use the existing fixed query excluding all five support candidates. Average
seeds within region, then weight the five regions equally. Report MAE, bias,
Q90 error and recall/sample counts, station-equal error, source support,
ecological novelty, hydro availability and attention allocation. Use 5,000
paired station bootstrap draws, keeping a station's months together.

Compare actual current availability with matched full-source, anomaly-only,
seasonal-only, retained complete and strong tree procedures, including each
native-only comparison. Distinguish the isolated availability gain from gains
already supplied by earlier versions. Show all regions and support counts.

This is a retrospective geographical role withholding within ST357, not an
independent external-basin evaluation. Previous geographical scores have been
seen, but no scored target selects this version's mechanism or settings; those
were chosen from source roles. Preserve the existing portable/external release
and scores. Further mechanism development returns to 142/143/144 source roles.
