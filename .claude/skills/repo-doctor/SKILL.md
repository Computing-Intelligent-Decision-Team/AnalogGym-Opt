---
name: repo-doctor
description: Health-check and troubleshoot the AnalogGym-Opt repo — validate circuit configs, smoke-run ngspice testbenches, and diagnose common failures (ngspice missing, OpenMP crash, broken measurements, missing op stats). Use when something fails, after pulling changes, or before a release.
---

# Repo health check

## Standard sweep

```bash
python tools/validate_configs.py        # config <-> instance consistency (fast)
python tools/run_instance_sims.py --all # every testbench through ngspice (slow)
```

`validate_configs` checks: input files exist, env constructs, `action_dim` matches,
graph shape (num_nodes vs observation rows, edge ranges), hierarchy rows have
observation entries, `vars.spice` params match the `device:` section.

## Known failure signatures

| symptom | cause / fix |
|---|---|
| `OMP: Error #15 ... libiomp5md.dll` | torch/numpy OpenMP clash — set `KMP_DUPLICATE_LIB_OK=TRUE` (repo tools do this themselves) |
| `ngspice` not found / hangs on `--version` | install ngspice, ensure on PATH; the GUI build hangs on `--version`, use `ngspice -b` to test |
| `missing file for op_mean_std_path` | run `python tools/gen_op_stats.py <circuit> --sims 30` |
| `meas ... out of interval` + missing wrdata file | initial design point not functional (e.g. gain never crosses 0 dB) — fix `<PREFIX>_vars.spice`, keep yaml init in sync |
| `PermissionError` deleting `simulation_output/...` | two sim jobs in one checkout — the env wipes `<cwd>/simulation_output` on construction; run one job at a time (validators avoid this by chdir'ing to a temp dir) |
| `ModuleNotFoundError: torch_geometric` | training env not active — evaluation/validation tools work without it, training needs it |
| vars.spice extra/missing params in validation | someone edited yaml `device:` or `vars.spice` without the other — keep the same `W_/L_/M_/I_` name set on both sides |

## Notes

- Simulation outputs (`*_ACDC_*`, `*_op`, `*_Tran*`, wrdata files) are regenerable; only
  the netlist, testbenches, `vars.spice`, `dev_params.spice`, `*_op_mean_std.json` and the
  yaml are ground truth.
- `simulation_files/sky130_pdk/` is the only PDK copy the code uses.
