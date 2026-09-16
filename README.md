# AnalogGym-Opt: Multi-Objective GRPO Sizing on AnalogGym Circuits

GRPO-based analog circuit sizing (group-relative policy optimization with a relational-GNN policy,
multi-objective reward with Pareto tracking and optional PVT verification) evaluated on
circuits from the [AnalogGym](https://github.com/CODA-Team/AnalogGym) testing suite,
simulated with **ngspice** on the open-source **SkyWater sky130** PDK.

This repository accompanies our IEEE TCAD paper
[*AnalogGym-Opt: An LLM-Oriented Optimization Infrastructure for Analog Circuit Sizing
with GRPO*](https://ieeexplore.ieee.org/document/11690621) — if you use this code in
your research, please cite it:

```bibtex
@ARTICLE{AnalogGym-Opt,
  author={Li, Jintao and Zhi, Haochang and Long, Yongji and Liu, Weixuan and Zeng, Yanhan and Zhu, Keren and Yu, Shui and Li, Yun},
  journal={IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems},
  title={AnalogGym-Opt: An LLM-Oriented Optimization Infrastructure for Analog Circuit Sizing with GRPO},
  year={2026},
  pages={1-1},
  keywords={Analog circuit sizing;group-relative policy optimization;PVT variations;variational autoencoders},
  doi={10.1109/TCAD.2026.3733612}}
```

## Requirements

- Python 3.10+ with `torch`, `torch-geometric`, `numpy`, `pyyaml`, `tabulate`
  (optional: `wandb` for logging)
- [ngspice](https://ngspice.sourceforge.io/) on the `PATH`
  (tested with the Windows build; any recent version works)
- The sky130 model files are bundled under `simulation_files/sky130_pdk/` — no setup needed.

## Quick start

```bash
python main_AMP_grpo.py --circuit amp_nmcf --steps 300 --pvt-mode proxy
```

(Without arguments it falls back to the `CIRCUIT_NAME` / `QUICK_CONFIG` constants at
the top of `main_AMP_grpo.py`.)

Every circuit instance is defined by a YAML config in `circuit_configs/` (design
variables and their ranges, performance targets, netlist hierarchy, GNN graph) plus a
simulation directory under `simulation_files/<name>/` (netlist, ngspice testbenches,
initial design variables). The environment (`AmpEnv` for amplifiers, `LdoEnv` for LDOs,
chosen automatically by `env_factory.make_env`) writes design variables, runs the
testbenches with ngspice, and parses the measurements into rewards/observations.

## Supported circuit instances

### Operational amplifiers (GRPO-integrated, `AmpEnv`)

All are three-pin-compensation amplifier topologies from the AnalogGym Amplifier suite,
ported to sky130 (1.8 V). Every instance is verified to run in ngspice with all
measurement files parsing correctly.

| config | AnalogGym topology | devices |
|---|---|---|
| `amp_nmcf`   | Leung_NMCF    | 24T+2C   |
| `amp_nmcnr`  | Leung_NMCNR   | 24T+2C+R |
| `amp_dfcfc1` | Leung_DFCFC1  | 26T+2C   |
| `amp_dfcfc2` | Leung_DFCFC2  | 26T+2C   |
| `amp_pfc`    | Ramos_PFC     | 24T+2C   |
| `amp_raffc`  | Alfio_RAFFC   | 24T+2C   |
| `amp_smc`    | Fan_SMC       | 24T+1C   |
| `amp_affc`   | HoiLee_AFFC   | 30T+2C   |
| `amp_acbc`   | Peng_ACBC     | 27T+2C   |
| `amp_iac`    | Peng_IAC      | 34T+2C+R |
| `amp_tcfc`   | Peng_TCFC     | 32T+2C   |
| `amp_azc`    | Qu2017_AZC    | 25T+3C+4R|
| `amp_cfcc`   | Sau_CFCC      | 24T+1C   |
| `amp_dacfc`  | Song_DACFC    | 37T+2C   |
| `amp_clia`   | Tan_CLIA      | 25T+2C+R |
| `amp_az`     | Yan_AZ        | 21T+2C+3R|

### Low-dropout regulator (GRPO-integrated, `LdoEnv`)

| config | AnalogGym source | notes |
|---|---|---|
| `ldo_basic` | Basic_LDO (RGNN_RL suite) | 24T + feedback divider + MiM load cap. Measures dropout, line/load regulation, offset, PSRR, loop gain/GBW/PM at min/max load (5–55 mA), and load-step under/overshoot. LDO objectives are mapped onto the repo's MOO keys (see `LdoEnv.py`). |

### Temperature sensors and voltage references (simulation-only)

Ported to ngspice + sky130 from the AnalogGym *Sensing Front End* and *Voltage Reference*
collections. These ship as netlist + testbench pairs (no GRPO config yet); run them with:

```bash
python tools/run_instance_sims.py sensor_ptat_2t ref_three_output   # or --all
```

| instance | AnalogGym source | measures |
|---|---|---|
| `sensor_ptat_2t`      | PTAT_SENSOR (2T core)        | V(T) 0–120 °C, TC, LSB, line sensitivity, PSR, IDD |
| `sensor_ptat_classic` | ptat_classic (4T+R)          | same |
| `sensor_ptat_65`      | PTAT_65_classic1             | same |
| `sensor_fe_31_3t`     | front_end_31_3T_schematic    | same |
| `sensor_fe_11_6t`     | front_end_11_6T_schematic    | same |
| `sensor_fe_25_6t`     | front_end_25_6T_schematic    | same |
| `ref_three_output`    | Three-output voltage reference | vref1/2/3 vs T, supply sweep, IDD |
| `ref_sub_vi`          | Sub-threshold V & I reference  | vref vs T, supply sweep, IDD |

Note: these circuits were originally designed on confidential 180/65 nm PDKs; the sky130
port keeps the topology and relative sizing but absolute levels and temperature
coefficients shift. Each file documents its source and any clamped device sizes.

## Adding more circuits from AnalogGym

```bash
git clone --depth 1 https://github.com/CODA-Team/AnalogGym.git <upstream>

# amplifiers (Pin_3 family): netlist + testbenches + vars + dev_params + YAML with GNN graph
python tools/import_analoggym_amp.py --upstream <upstream> --name Fan_SMC_Pin_3
python tools/import_analoggym_amp.py --upstream <upstream> --all

# the Basic_LDO instance
python tools/import_analoggym_ldo.py --upstream <upstream>

# generate the .OP statistics used to normalize GNN observations (runs random sims)
python tools/gen_op_stats.py --missing --sims 30

# consistency checks over all configs + instances
python tools/validate_configs.py

# smoke-run every instance's testbenches through ngspice
python tools/run_instance_sims.py --all
```

The importer reproduces the repo conventions exactly (it regenerates the hand-made
`amp_nmcf` graph bit-for-bit): design-variable groups `W_Mx/L_Mx/M_Mx`, the bias current
exposed as an `Ib` pin, ideal capacitors quantized to 30×30 µm sky130 MiM units
(~1.82 pF each, multiplier `M_Ck`), ideal resistors mapped to high-res poly with
multiplier `M_Rk`, and the circuit graph (nodes = devices + Ib/VDD/GND + passives,
edge type 1 for supply/bias edges) generated from netlist connectivity.

## LLM integration (skills & MCP)

As an *LLM-oriented* optimization infrastructure, this repo is built to be driven by AI
coding agents, not just by hand. Two integration layers ship with it, both thin wrappers
over the same CLI tools — so an agent can inspect a circuit, simulate candidate design
points, launch/monitor GRPO runs, and add new benchmark circuits without any glue code:

**Claude Code skills** (`.claude/skills/`) — *skills are markdown playbooks that an AI
coding agent loads on demand*; they encode this repo's workflows, conventions and known
failure modes. They are picked up automatically when you open this repo in
[Claude Code](https://claude.com/claude-code) (no installation):

| skill | what it does |
|---|---|
| `evaluate-design`   | manual/LLM-driven sizing loops: simulate one design point, interpret metrics vs targets |
| `run-optimization`  | configure, launch and monitor GRPO runs |
| `add-circuit`       | import + verify new AnalogGym circuits |
| `repo-doctor`       | health checks and troubleshooting |

**MCP server** (`tools/mcp_server.py`, registered via `.mcp.json`) — the
[Model Context Protocol](https://modelcontextprotocol.io) is an open standard that lets
any LLM agent call external tools with typed schemas; this server exposes the optimizer
as eight such tools: `list_circuits`, `describe_circuit`, `evaluate_design`,
`run_instance_sims`, `validate_configs`, `start_training`, `training_status`,
`stop_training`. Requires `pip install mcp`; Claude Code picks it up from `.mcp.json`,
other clients run it over stdio: `python tools/mcp_server.py`.

Typical agent loop: `describe_circuit` → propose sizing → `evaluate_design` → read the
per-metric scores → refine, or hand off to `start_training` for a full GRPO run.

The underlying CLI works standalone too:

```bash
# simulate one design point and score it against the targets
python tools/evaluate_design.py --circuit amp_smc --vars M_M11=64 W_M8=4

# launch training with overrides
python main_AMP_grpo.py --circuit amp_smc --steps 300 --pvt-mode proxy
```

## Repo layout

```
main_AMP_grpo.py          training entry point (--circuit/--steps/--pvt-mode)
AmpEnv.py / LdoEnv.py     simulation environments (ngspice in, reward/observation out)
env_factory.py            picks the env class from the config category
circuit_configs/*.yaml    per-circuit configs (devices, targets, hierarchy, graph)
simulation_files/<name>/  netlist + testbenches + initial design point (+ reference outputs)
simulation_files/sky130_pdk/  bundled sky130 ngspice models
tools/                    importers, validators, evaluator, MCP server, smoke-run utilities
.claude/skills/           Claude Code skills (evaluate-design, run-optimization, ...)
```

## Credits

- Circuit topologies and testbench methodology: [AnalogGym](https://github.com/CODA-Team/AnalogGym)
  (CODA-Team), *AnalogGym: An Open and Practical Testing Suite for Analog Circuit Synthesis*.
- PDK: [SkyWater sky130](https://github.com/google/skywater-pdk).
