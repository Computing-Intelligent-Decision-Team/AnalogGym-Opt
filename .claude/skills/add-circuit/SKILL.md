---
name: add-circuit
description: Import a new circuit instance from the upstream AnalogGym benchmark (amplifiers, LDO) or convert other AnalogGym circuits (sensors, references) into this repo's format, then verify it end to end. Use when the user wants to add/support a new circuit, benchmark or topology.
---

# Add a circuit from AnalogGym

## Amplifiers (Pin_3 family) — fully automated

```bash
git clone --depth 1 https://github.com/CODA-Team/AnalogGym.git <upstream>   # if not present
python tools/import_analoggym_amp.py --upstream <upstream> --name <X>_Pin_3
```

This generates everything under `simulation_files/amp_<short>/` + `circuit_configs/amp_<short>.yaml`
(netlist with `W_Mx/L_Mx/M_Mx` groups and the bias exposed as an `Ib` pin, both testbenches,
vars, dev_params, YAML incl. the GNN graph). New upstream names must be added to
`KNOWN_CIRCUITS` in the importer first. The LDO variant is `tools/import_analoggym_ldo.py`.

## Verify (required, in this order)

1. `cd simulation_files/amp_<short>` then
   `ngspice -b -o ACDC.log <PREFIX>_ACDC.cir` and `ngspice -b -o Tran.log <PREFIX>_Tran.cir`
   — exit 0, no `Error` lines, all six output files produced
   (`_ACDC_DC`, `_ACDC_AC`, `_ACDC_GBW_PM`, `_op`, `_Tran`, `_tran.dat`).
2. If measurements fail (gain never crosses 0 dB, meas "out of interval"): the imported
   initial point is not functional. Fix the operating point in `<PREFIX>_vars.spice`
   (typical culprits: mirror devices too narrow for sky130 headroom -> raise W or M;
   compensation caps too small -> raise `M_Ck`; bias current wrong decade -> `I_Ib`).
   Mirror every change into the yaml `device:` init values.
3. `python tools/gen_op_stats.py amp_<short> --sims 30` (observation normalization; slow).
4. `python tools/validate_configs.py` must report the new config clean.
5. Optional end-to-end: `python tools/evaluate_design.py --circuit amp_<short>`.

## Conventions you must not break

- Caps: sky130 MiM 30x30 um units (`MF=M_Ck m=M_Ck`) — the unit size is hardcoded in
  `AmpEnv._get_obs`/area model, do not use other geometries.
- Resistors: `sky130_fd_pr__res_high_po_0p35`, length chosen so the upstream value is hit
  at `M_Rk=4`; resistors go in `device:`/graph/observation_matrix but NOT in `ckt_hierarchy`.
- Testbenches follow the repo template exactly (the env rewrites the sky130 include paths
  by string match) — never restructure them per-circuit.
- `dev_params.spice` write-order must match `ckt_hierarchy` (generate it via
  `DeviceParams.gen_dev_params`, never by hand).

## Other circuit classes (sensors, references)

Simulation-only instances need just `<NAME>_netlist.txt` + `<NAME>_TB.cir` under
`simulation_files/<name>/` following the existing sensor/ref dirs as templates
(sky130 include block, `.control` measurements, `wrdata` outputs; header comment naming
the upstream source). Verify with `python tools/run_instance_sims.py <name>`.
Giving such an instance a GRPO config additionally requires an env subclass that parses
its wrdata files into the MOO objective keys — follow `LdoEnv.py` as the reference and
register the category in `env_factory.py`.
