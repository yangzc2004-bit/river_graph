# Stage 2C controls status

Plan-only freeze complete. No model has been trained and no prediction product has been generated. Review `execution_plan.json` before using the following explicit command:

```bash
.venv/bin/python scripts/run_phase4_stage2c_controls.py --execute --ack-plan-sha256 5f2dcba55e45fe1967f16b6bb4813bc8a075ca9049e814473cd2aeafdd388fcb
```

The reviewed plan contains 180 units (135 GNN and 45 EcoRF fits).
