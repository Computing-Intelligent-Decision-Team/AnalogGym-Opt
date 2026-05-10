# AnalogGym-Opt Demo

AnalogGym-Opt is a GRPO-based optimizer for analog circuit parameter search. This repository is the public demo package submitted with the TCAD paper review. It keeps the executable framework and one amplifier benchmark, `amp_dfcfc2`. The complete code and datasets will be open-sourced after the paper is accepted.

## Demo Scope

- Included circuit: `amp_dfcfc2`
- Included config: `circuit_configs/amp_dfcfc2.yaml`
- Included simulation templates: `simulation_files/amp_dfcfc2/`
- Included PDK dependency: one bundled Sky130 PDK copy under `simulation_files/sky130_pdk/`
- Excluded before paper acceptance: full benchmark set, complete datasets, and generated training results

The complete dataset and additional cases will be released after paper acceptance.

## Prerequisites

- Python 3.9 or newer
- Ngspice available on `PATH`
- PyTorch and PyTorch Geometric compatible with your Python/CUDA setup

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Check Ngspice:

```bash
ngspice -v
```

## Quick Start

Run the default demo:

```bash
python main_AMP_grpo.py --circuit amp_dfcfc2 --steps 300 --mode tt-proxy
```

Run a short smoke test:

```bash
python main_AMP_grpo.py --circuit amp_dfcfc2 --steps 1 --mode tt-only
```

Available modes:

- `tt-only`: TT-corner optimization only; useful for quick checks.
- `tt-proxy`: TT inner-loop training with selective PVT proxy/verification; this is the default demo mode.
- `full-pvt`: full PVT evaluation inside training; this is much more expensive.

## Outputs

Runtime outputs are disposable and are written to:

- `simulation_output/`: generated Ngspice work directories and logs
- `training_saves/`: checkpoints, plots, candidate summaries, and training histories

These directories are ignored by Git and can be deleted between runs.

## Repository Layout

- `main_AMP_grpo.py`: command-line entry point
- `grpo.py`: GRPO agent and training loop
- `AmpEnv.py`: Ngspice-backed circuit environment
- `models.py`: graph policy networks
- `circuit_config_loader.py`: YAML config loader and path resolver
- `circuit_configs/`: circuit definitions
- `simulation_files/amp_dfcfc2/`: read-only demo circuit templates
- `simulation_files/sky130_pdk/`: bundled Sky130 model files used by the demo

## Notes

The bundled Sky130 files are third-party process-design-kit assets required by the demo simulation decks. Keep their upstream license terms in mind when redistributing modified PDK content.

## License

This project code is released under the MIT License. See `LICENSE`.
