# Sidecar schema repair

The residual pilot was trained with configuration schema V6 in the persisted
`config` object, but the runner wrote the top-level `config_hash_version` as
its old literal V5. The predictions and configuration hash were unchanged.
After training stopped, the 24 residual pilot sidecars were updated to expose
the already-recorded V6 schema consistently. The runner now reads the schema
version from the config and will write V6 for future residual runs.
