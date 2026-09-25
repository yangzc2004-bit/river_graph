# Stage 2C controls status

Plan-only freeze complete. No model has been trained and no prediction product has been generated. Review `execution_plan.json` before using the following explicit command:

```bash
.venv/bin/python scripts/run_phase4_stage2c_controls.py --execute --ack-plan-sha256 cb421a967b7261f075c0c4f11a82120e3d31dcc8023682e23cc904137e72eb61
```

The reviewed plan contains 180 units (135 GNN and 45 EcoRF fits).
