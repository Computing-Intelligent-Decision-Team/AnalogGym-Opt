# GRPO Analog Circuit Optimizer (Minimal Executable Version)

This repository is dedicated to training intelligent agents for analog circuit parameter optimization. The core algorithm is based on **GRPO (Group Relative Policy Optimization)**.

## 📌 Project Highlights
- **Minimal Executable Version**: This version strips away all redundant, debugging, and comparative scripts (such as DDPG baselines), keeping only the essential, standardized execution files.
- **Chinese Comments**: Key source files and the main entry script contain Chinese comments for better readability.
- **Test Case**: Includes the `dfcfc2` amplifier test case by default (config located at `circuit_configs/amp_dfcfc2.yaml`) for rapid execution and verification.

## 📂 Dataset Availability
> [!IMPORTANT]
> **The complete dataset will be officially released upon the paper's acceptance.**
> The current version contains only the core simulation and algorithmic framework required to run the optimization, ensuring a lightweight and executable setup.

## 🛠️ Prerequisites
Ensure the following dependencies are installed (Python 3.8+ is recommended):
- PyTorch
- NumPy
- Matplotlib
- Tabulate
- **Ngspice**: Must be installed and added to your system's PATH, as it is required for low-level circuit simulation.

## 🚀 Quick Start
Run the main script to start optimization training for the `dfcfc2` test case:

```bash
python main_AMP_grpo.py
```

Optimization outcomes, the best discovered circuit parameters, and training metrics will be logged in the `training_saves/` directory.

## 📄 Project Structure
- `main_AMP_grpo.py`: Main entry point for starting the training process.
- `grpo.py`: Core implementation of the Group Relative Policy Optimization (GRPO) algorithm.
- `AmpEnv.py`: Circuit interaction environment handling Ngspice simulation and reward computation.
- `models.py`: Neural network definitions for the Actor.
- `circuit_configs/`: Directory containing circuit specifications and constraints (e.g., `amp_dfcfc2.yaml`).
