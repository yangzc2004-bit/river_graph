# Unmonitored-station DOC reconstruction: source-retrieval development

Date: 2026-10-05. Authorized by the project-wide remaining-work plan.

## Scientific task

Reconstruct DOC at stations with no DOC, pH or conductance inputs. Retain the
current environmental ExtraTrees, ecological encoder, observation-aware GRU,
daily-flow descriptors and native residual decomposition. Source stations may
provide their visible DOC observations. This is monthly retrospective
reconstruction; current-month hydro and visible source observations are allowed.

Development uses source training and source-validation roles in existing
station partitions142/143/144, seeds42/43/44. No target-label metrics are
generated or used for model selection. Primary validation K0 includes every
valid validation DOC cell. Separate K curves reserve the same five support
candidates at every K. Previously evaluated target roles remain historical.

## Comparisons

1. Current daily-head, ecological-memory integrated model, chemistry absent.
2. Strong matched-input ExtraTrees (existing39 columns plus8 daily descriptors).
3. Existing static ecological-affine residual transfer.
4. Current model with learned source retrieval.
5. Hydro-pretrained encoder/GRU with the same native residual head and static memory.
6. Hydro-pretraining plus source retrieval.

The four learning arms are 1/4/5/6. Parent states are frozen and reused only
within their original source roles. Initial new DOC fits:30epochs,patience5.

## Source experience

Five outer source-station folds define pseudo-target episodes. Every episode
removes the entire outer fold from input visibility, forest fitting and the
donor bank. Donor residuals come from an inner station-blocked cross-fit inside
the remaining source stations. The final bank uses source-only five-fold OOF
residuals. A new query needs no DOC labels and no cohort-specific node identity.

Each donor supplies a regularized native residual profile conditioned on
environmental prediction, calendar season and hydro. Retrieve up to20
ecologically nearest donor stations. Two32-dimensional attention heads use
query ecology, current/past hydro, calendar and frozen/current GRU state.
Zero-initialized projection learns an increment to the existing static memory.
Temporal and memory experts still estimate the same forest residual and are
blended, not added twice. Fusion scale is selected on source validation.

## Hydro pretraining

Mask20% of valid temperature/discharge inputs and their visibility, with a
fixed source-only validation mask. Reconstruct source-standardized temperature
and log1p(nonnegative discharge). Fine-tune the existing encoder/GRU, at most
30epochs, without changing backbone width/lookback. Masked hydro must also be
removed from any input-derived support features; daily descriptors remain
excluded from the reconstruction head. Save the pretrained state, then train
the ordinary DOC residual with the existing native-MAE/tail-weight2 objective.

## Evaluation and next studies

Report all-validation K0 MAE/RMSE/R2/Q90, station-balanced MAE, source novelty,
hydro completeness, retrieval entropy/effective donor count and station effects.
Development is not a new independent confirmation. Geographical confirmation
rotates target HUC4s1013,1019,0708,1030,1101; the next listed block is validation.
Start with three seeds, then five for the retained complete procedure and two
fixed main controls. Working target:5% K0 improvement over current and matched
trees, positive paired interval, at least three improved geographical blocks.

External DOC-only candidate availability is screened in order02040104 then
02030103:50 stations with>=6 DOCmonths,>=36 calendar months,>=1000 fixed queries,
no ST357 station overlap, constructible common inputs. Chemistry is optional.
External selection never uses model outcomes. This new DOC-only scope does not
overwrite the old three-analyte external protocol.

Every iteration keeps its full fitted procedure and tests. New outputs and
runtime snapshots live here; old results remain unchanged. Large forests/input
caches remain local. Basic label/future isolation, replay, pytest and Ruff are
maintained as research infrastructure.
