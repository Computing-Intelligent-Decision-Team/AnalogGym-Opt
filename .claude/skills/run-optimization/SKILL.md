---
name: run-optimization
description: Configure, launch and monitor a GRPO sizing run for a circuit (amp_* or ldo_*). Use when the user wants to optimize a circuit, start/resume training, choose the PVT mode, watch progress, or collect the resulting Pareto designs.
---

# Run a GRPO optimization

## Launch

```bash
python main_AMP_grpo.py --circuit amp_smc --steps 300 --pvt-mode proxy
```

- `--circuit`: any name from `circuit_configs/` (omit to use the `CIRCUIT_NAME` constant).
- `--steps`: training steps (default 300). Each step simulates 8 designs; a TT-only step
  costs seconds-to-minutes of ngspice per design.
- `--pvt-mode`: `tt` = nominal corner only (fastest), `proxy` = TT training with a VAE
  proxy + selective real PVT verification (default, recommended), `full` = all 21 PVT
  corners every step (very slow).

Long runs should go in the background; capture stdout to a log file and poll it.
Requires the Python env with `torch` + `torch_geometric` installed — if the import
fails, ask the user which conda env they train in and use that interpreter.

## Monitor and interpret

- The log prints per-step rewards, per-metric tables and Pareto updates.
- Results land in `training_saves/` (checkpoints, recommended designs) — the final
  summary at the end of the log lists the Pareto-front candidates and, when PVT
  verification is on, the worst-corner-verified ones.
- `reward >= 0` = all scored targets met; reward decomposes into
  constraint / FOML / FOMS / Area objectives.

## Changing what "good" means

Performance targets live in the `performance:` section of `circuit_configs/<name>.yaml`
(targets are goals for scoring, `baseline` values feed the FOM normalization). After
editing a config, run `python tools/validate_configs.py` before training.

## Guardrails

- One run per checkout: training, `tools/evaluate_design.py` and
  `tools/gen_op_stats.py` all share `simulation_output/` and must not run concurrently.
- A new circuit needs its `<PREFIX>_op_mean_std.json` (observation normalization);
  if missing, generate it first: `python tools/gen_op_stats.py <circuit> --sims 30`.
- Do not flip `op_initial`/`dev_initial` in `main_AMP_grpo.py` casually: `dev_initial`
  regenerates `dev_params.spice` (safe), `op_initial` re-runs 100 random sims (slow).
