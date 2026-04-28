---
name: analoggym-opt-optimizer
description: Use this skill when working with the AnalogGym-Opt paper code in this repo, especially to inspect circuit YAML configs, validate simulation inputs, or run the GRPO optimizer in tt-only, tt-proxy, or full-pvt modes without editing the paper entry script.
---

# AnalogGym Opt Optimizer

This skill wraps the optimizer code in `Analoggym_opt_moo_Mahalanobis_paper/Analoggym_opt_moo_Mahalanobis_paper` with a stable CLI for large-model use.

## Quick Start

Run these commands from the repo root:

```bash
python skills/analoggym-opt-optimizer/scripts/optimizer_cli.py list-circuits
python skills/analoggym-opt-optimizer/scripts/optimizer_cli.py describe-circuit --circuit amp_dfcfc2
python skills/analoggym-opt-optimizer/scripts/optimizer_cli.py validate-circuit --circuit amp_dfcfc2
python skills/analoggym-opt-optimizer/scripts/optimizer_cli.py train --circuit amp_dfcfc2 --steps 300 --mode tt-proxy
```

For a no-risk config check before training:

```bash
python skills/analoggym-opt-optimizer/scripts/optimizer_cli.py train --circuit amp_dfcfc2 --steps 20 --mode tt-proxy --dry-run
```

## Workflow

1. Use `list-circuits` to see available YAML configs.
2. Use `describe-circuit` before touching a circuit you do not know.
3. Use `validate-circuit` before any training run, especially for a new YAML or edited netlist deck.
4. Use `train` to launch the paper entry script with explicit mode and step budget.

## Mode Selection

- `tt-only`: fast smoke test; no outer-loop PVT verification.
- `tt-proxy`: default paper-aligned mode; TT inner loop plus selective real-PVT verification.
- `full-pvt`: expensive mode; real full-corner evaluation inside training.

Default to `tt-proxy` unless the user explicitly asks for a cheaper smoke test or full-corner training at every step.

## What The Wrapper Exposes

The wrapper is intentionally narrow. It forwards the stable knobs that matter most for agent use:

- circuit name
- training steps
- PVT mode
- repo root override

Everything else stays on the paper defaults in `main_AMP_grpo.py`. That keeps model calls predictable and avoids ad hoc source edits.

## Runtime Notes

- The training entry still depends on the local Python environment and `ngspice`.
- `validate-circuit` checks YAML structure and resolved file paths, but it does not prove that PDK include paths inside the SPICE decks are valid.
- Result folders are created under `Analoggym_opt_moo_Mahalanobis_paper/Analoggym_opt_moo_Mahalanobis_paper/training_saves`.
- In `tt-proxy` mode, expect extra candidate summaries for verified-PVT recommendations.

## When To Read References

- Read `references/circuit-contract.md` when adding or debugging a circuit config.
- Read it before changing graph fields, device ranges, or simulation file paths.

## Ground Rules

- Do not edit `main_AMP_grpo.py` just to switch circuits or PVT modes; use the wrapper.
- Keep optimizer edits local and comment only where behavior is not obvious.
- Treat circuit YAMLs and simulation decks as the interface boundary for new circuits.
