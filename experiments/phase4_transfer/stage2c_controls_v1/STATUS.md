# Stage 2C controls status

Plan-only freeze complete. No model has been trained and no prediction product has been generated. Review `execution_plan.json` before using the following explicit command:

```bash
.venv/bin/python scripts/run_phase4_stage2c_controls.py --execute --ack-plan-sha256 632f553fd96880dee892f32f581790aac3c02e9b1b0809ccd0b9eca87bbb1163
```

The reviewed plan contains 180 units (135 GNN and 45 EcoRF fits).
