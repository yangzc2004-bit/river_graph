# INCA-C process-model reproduction feasibility

Date: 2026-10-11. Representative: the public Mobius continuation of INCA-C, with integrated PERSiST hydrology. The investigation includes compilation and native example execution, not just a search for a repository.

## Finding

The available implementation can run a complete supplied example on this host. Its unmodified Storgama parameter and input files produced **10,227 daily DOC concentrations from 1985-01-01 through 2012-12-31**, all finite and nonnegative, in the source-declared mg/L unit. This establishes executable availability. It does **not** reproduce the original 2007 paper's numerical results or constitute a DOC performance comparison on our Mississippi cohort.

## Inspectable evidence

- [Original INCA-C paper](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2006WR004960): carbon sorption/desorption/mineralisation and transport through soils and streams, designed for natural/seminatural forested and peat-dominated catchments.
- [Author repository and model description](https://github.com/NIVANorge/Mobius/tree/master/Applications/IncaC). Source commit: `e95aa17f551efe3c96d87940899c74b8b398e71d`.
- [Mobius framework paper](https://gmd.copernicus.org/articles/14/1885/2021/).
- Local native engine, module, wrapper, parameters and input data: `cache/doc_process_model_v1/Mobius/`. Mathematical equations and native example files were not changed. A macOS compiler flag `-D__unix__` enables the repository's existing POSIX symbol-export branch. This is a platform build adjustment, not a replacement DOC implementation.
- [Execution audit](../experiments/phase4_transfer/doc_process_model_audit_v2/audit.json), [example output](../experiments/phase4_transfer/doc_process_model_audit_v2/storgama_example.parquet), [provenance sidecar](../experiments/phase4_transfer/doc_process_model_audit_v2/storgama_example.meta.json), and [reproducible probe](../scripts/probe_doc_process_model_v2.py). The audit records source/dependency hashes, exact build command, native inputs, output hash and 88 registered parameter definitions. Parameter definitions are not all independent calibration variables.

The first Langtjern example attempt failed on an unrecognised legacy parameter name (`Organic soil water initial SOC mass`). That native failure is preserved in [the first audit](../experiments/phase4_transfer/doc_process_model_audit_v1/audit.json). The working Storgama example was selected for input/code compatibility, not by its DOC error. No silent parameter renaming or deletion was used.

## Limits to full paper reproduction

The public README explicitly says that its supplied examples are not correctly calibrated. Its continuation also differs from earlier INCA-C: PERSiST is integrated, additional later carbon processes are included, particulate carbon is omitted, and terrestrial carbon inputs are grouped as litterfall. Therefore a functioning public implementation is distinct from replicating the exact original equations, forcing, initial states, calibration and numerical performance of Futter et al. (2007). Those matches are not established here.

## Requirements for our matched-data comparison

The current DOC cohort supplies monthly water-temperature/discharge covariates and ecological/static attributes. These are not a complete PERSiST/INCA-C project. A new application requires daily meteorological forcing (including air temperature and precipitation), mapped catchments/reaches and land classes, soil-layer/hydrological parameters, initial carbon stores and the selected carbon/temperature/sulfate/litterfall parameterisation. Optional native input series and defaults must be recorded explicitly, rather than assuming every listed series is mandatory.

The comparison must also establish reach-to-station mapping, hydrological warm-up, source-only calibration and regional parameter transfer without target DOC, and the same DOC concentration/observation aggregation convention. Daily simulations should be matched to the original sampling/aggregation support when constructing a monthly score. The northern forest/peat setting motivates starting with a suitable smaller catchment before claiming applicability throughout the mixed Mississippi network.

**Decision:** public implementation/example execution is feasible and demonstrated; exact original-paper reproduction remains unverified; immediate matched Mississippi reproduction is not ready because a native forcing/parameter project and an audited calibration/transfer protocol have not been assembled. No physical-model error score is inserted into the interpolation benchmark.
