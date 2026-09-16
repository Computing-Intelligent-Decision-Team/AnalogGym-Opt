---
name: evaluate-design
description: Evaluate and iterate on analog circuit design points (manual/LLM-driven sizing) — simulate one set of W/L/M/Ib/cap values with ngspice and interpret every metric against its target. Use when the user wants to size a circuit by hand, test a specific design point, sweep a variable, or understand why a design misses its targets.
---

# Evaluate a design point

## Workflow

1. See what exists: `python tools/run_instance_sims.py --all` lists instances; configured circuits are the `circuit_configs/*.yaml` names (`amp_*`, `ldo_*`).
2. Inspect the circuit before proposing values — read `circuit_configs/<name>.yaml`:
   - `device:` section = design variables. Each transistor group `Mx` has `W_Mx` (um), `L_Mx` (um), `M_Mx` (multiplier); plus `I_Ib` (bias current, A), `M_Ck` (cap units of ~1.82 pF), `M_Rk` (parallel resistor multiplier, larger M = smaller R). Respect `range`; out-of-range values are clamped with a warning.
   - `performance:` section = targets the reward is scored against.
3. Evaluate (always from the repo root):
   ```bash
   python tools/evaluate_design.py --circuit amp_smc                       # config init point
   python tools/evaluate_design.py --circuit amp_smc --vars M_M11=64 W_M8=4
   python tools/evaluate_design.py --circuit ldo_basic --vars-file pt.json --json
   ```
   Runtime ~0.5–5 min per point (both testbenches). `--json` gives the full info dict.
4. Interpret the table: `num` = measured, `target` = goal. `reward >= 0` means every scored
   target is met. The reward decomposes into constraint / FOML / FOMS / Area objectives
   (for LDOs the same keys carry LDO metrics: FOML = load-step ripple, FOMS = GBW·CL/Power).

## Sizing heuristics for this codebase

- Phase margin scores flat-zero inside 45–90°; outside it dominates the constraint loss.
  First fix PM (compensation caps `M_C0/M_C1`, bias `I_Ib`), then push GBW/gain.
- These are sky130 1.8 V circuits: W=L=1 um mirrors need ~1.7 V of headroom at 20 uA —
  if everything rails, widen the mirror devices (W up, or M up) before touching anything else.
- Caps are quantized to ~1.82 pF units; `M_Ck=1` is already a sizeable cap.

## Guardrails

- Never edit `simulation_files/<name>/<PREFIX>_vars.spice` or the testbenches to try a
  point — the evaluator runs in a scratch copy precisely so the shipped instance stays intact.
- Do not run two evaluations/trainings concurrently in one checkout: they share
  `simulation_output/` and the environment wipes it on construction.
- If Python errors with an OpenMP/libiomp message, the tools already set
  `KMP_DUPLICATE_LIB_OK=TRUE`; only export it manually for ad-hoc python snippets.
