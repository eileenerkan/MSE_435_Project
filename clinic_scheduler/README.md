# Clinic Scheduler

Course project implementation for MSE 435 / MSE 546 scheduling experiments.

## Run

```bash
python -m clinic_scheduler.scheduler --week all --policy all --output ./results/
```

## Notes

- Comparison order is: historical clinic baseline, unconstrained solver optimum (`Optimal`), then `Policy A` through `Policy F`.
- `--policy Optimal` or `--policy unconstrained` runs the solver with `policy_params = {}`.
- `--policy Historical Baseline` renders the clinic's actual historical schedule without optimization.
- Week 1 Tuesday appointments are skipped because the clinic is closed.
- `Admin Time` appointment types are excluded from optimization.
- Visual outputs and KPI summaries are saved under `results/week{N}/`.
