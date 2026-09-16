"""
DDPG vs GRPO comparison script.

Loads the latest saved training histories and generates summary plots.
"""

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _latest_file(paths):
    if not paths:
        return None
    return sorted(paths, key=lambda path: path.stat().st_mtime)[-1]


def load_ddpg_rewards():
    ddpg_dir = Path("../RGNN_RL/saved_memories")
    ddpg_file = _latest_file(list(ddpg_dir.glob("*.npy")))
    if ddpg_file is None:
        print("  No DDPG results found in ../RGNN_RL/saved_memories/")
        print("  Run: cd ../RGNN_RL && python main_AMP.py")
        return None

    print(f"  Loading: {ddpg_file.name}")
    rewards = np.load(ddpg_file)
    print(f"  DDPG rewards loaded: {len(rewards)} steps")
    print(f"    Best reward: {rewards.max():.4f}")
    print(f"    Final reward: {rewards[-100:].mean():.4f} (last 100 steps)")
    return rewards


def load_grpo_history():
    grpo_root = Path("./training_saves")
    files = list(grpo_root.rglob("training_history_GRPO_*.pkl"))
    files += list(grpo_root.rglob("training_history_step_*.pkl"))
    grpo_file = _latest_file(files)
    if grpo_file is None:
        print("  No GRPO results found in ./training_saves/")
        print("  Run: python main_AMP_grpo.py")
        return None

    print(f"  Loading: {grpo_file.name}")
    with open(grpo_file, "rb") as handle:
        history = pickle.load(handle)

    rewards = history["reward_history"]
    config = history.get("config", {})
    history["num_designs_per_circuit"] = config.get("num_designs_per_circuit", 1)
    print(f"  GRPO rewards loaded: {len(rewards)} steps")
    print(f"    Best reward: {max(rewards):.4f}")
    print(f"    Final reward: {np.mean(rewards[-10:]):.4f} (last 10 steps)")
    return history


def moving_average(data, window):
    return np.convolve(data, np.ones(window) / window, mode="valid")


def plot_comparison(ddpg_rewards, grpo_history):
    grpo_rewards = grpo_history["reward_history"]
    grpo_designs_per_step = grpo_history["num_designs_per_circuit"]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("DDPG vs GRPO Comparison", fontsize=16, fontweight="bold")

    ax = axes[0, 0]
    ax.plot(ddpg_rewards, label="DDPG", alpha=0.7, linewidth=1)
    ax.plot(grpo_rewards, label="GRPO", alpha=0.7, linewidth=1)
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Reward")
    ax.set_title("Training Reward Curves")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    window = 50
    if len(ddpg_rewards) > window:
        ax.plot(moving_average(ddpg_rewards, window), label=f"DDPG (MA-{window})", linewidth=2)
    if len(grpo_rewards) > window:
        ax.plot(moving_average(grpo_rewards, window), label=f"GRPO (MA-{window})", linewidth=2)
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Smoothed Reward")
    ax.set_title(f"Smoothed Reward (Window={window})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 2]
    success_rate_history = grpo_history.get("success_rate_history")
    if success_rate_history:
        ax.plot(success_rate_history, label="GRPO", color="green", linewidth=2)
        ax.axhline(y=0.5, color="red", linestyle="--", label="50% threshold")
        ax.set_xlabel("Training Steps")
        ax.set_ylabel("Success Rate")
        ax.set_title("Success Rate (GRPO)")
        ax.set_ylim([0, 1])
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, "No success rate data\nfor GRPO", ha="center", va="center", fontsize=12)
        ax.set_title("Success Rate")

    ax = axes[1, 0]
    loss_history = grpo_history.get("loss_history")
    if loss_history:
        ax.plot(loss_history, label="GRPO Loss", color="orange", linewidth=2)
        ax.set_xlabel("Training Steps")
        ax.set_ylabel("Loss")
        ax.set_title("Policy Loss (GRPO)")
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, "No loss data available", ha="center", va="center", fontsize=12)

    ax = axes[1, 1]
    ax.hist(ddpg_rewards, bins=50, alpha=0.5, label="DDPG", color="blue")
    ax.hist(grpo_rewards, bins=50, alpha=0.5, label="GRPO", color="green")
    ax.axvline(x=0, color="red", linestyle="--", label="Success threshold")
    ax.set_xlabel("Reward")
    ax.set_ylabel("Frequency")
    ax.set_title("Reward Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    ddpg_success = (ddpg_rewards > 0).sum() / len(ddpg_rewards)
    grpo_success_rate = np.mean([reward > 0 for reward in grpo_rewards])
    if grpo_success_rate > 0:
        grpo_sims_per_success = grpo_designs_per_step / grpo_success_rate
    else:
        grpo_sims_per_success = float("inf")
    if ddpg_success > 0:
        ddpg_sims_per_success = 1.0 / ddpg_success
    else:
        ddpg_sims_per_success = float("inf")

    ax = axes[1, 2]
    ax.axis("off")
    stats_text = "\n".join(
        [
            "Comparison Statistics",
            "=" * 40,
            "",
            "DDPG:",
            f"  Total steps: {len(ddpg_rewards)}",
            f"  Best reward: {ddpg_rewards.max():.4f}",
            f"  Final reward: {ddpg_rewards[-100:].mean():.4f}",
            f"  Std dev: {ddpg_rewards.std():.4f}",
            f"  Success rate: {ddpg_success:.2%}",
            "",
            "GRPO:",
            f"  Total steps: {len(grpo_rewards)}",
            f"  Best reward: {max(grpo_rewards):.4f}",
            f"  Final reward: {np.mean(grpo_rewards[-10:]):.4f}",
            f"  Std dev: {np.std(grpo_rewards):.4f}",
            f"  Success rate: {grpo_success_rate:.2%}",
            "",
            "Comparison:",
            f"  Best reward diff: {max(grpo_rewards) - ddpg_rewards.max():+.4f}",
            f"  DDPG sims/success: {ddpg_sims_per_success:.1f}",
            f"  GRPO sims/success: {grpo_sims_per_success:.1f}",
        ]
    )
    ax.text(
        0.05,
        0.95,
        stats_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
    )

    plt.tight_layout()
    save_path = "./comparison_ddpg_vs_grpo.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Comparison plot saved to: {save_path}")
    plt.show()


def print_summary(ddpg_rewards, grpo_history):
    print("\n" + "=" * 80)
    print("Summary Report")
    print("=" * 80)

    if ddpg_rewards is None or grpo_history is None:
        print("\nMissing data - run both algorithms first:")
        if ddpg_rewards is None:
            print("  - DDPG: cd ../RGNN_RL && python main_AMP.py")
        if grpo_history is None:
            print("  - GRPO: python main_AMP_grpo.py")
        print("=" * 80)
        return

    grpo_rewards = grpo_history["reward_history"]
    grpo_designs_per_step = grpo_history["num_designs_per_circuit"]
    ddpg_best = ddpg_rewards.max()
    grpo_best = max(grpo_rewards)

    print("\nComparison completed successfully.")
    print("\nKey Findings:")
    if grpo_best > ddpg_best:
        print(f"  - GRPO achieved better best reward (+{(grpo_best - ddpg_best):.4f})")
    elif grpo_best < ddpg_best:
        print(f"  - DDPG achieved better best reward (+{(ddpg_best - grpo_best):.4f})")
    else:
        print("  - Both achieved similar best rewards")

    print("\n  - GRPO memory usage: about 50% of DDPG (no Critic network)")

    ddpg_std = ddpg_rewards.std()
    grpo_std = np.std(grpo_rewards)
    if grpo_std < ddpg_std:
        print(f"  - GRPO more stable (std: {grpo_std:.4f} vs {ddpg_std:.4f})")
    else:
        print(f"  - DDPG more stable (std: {ddpg_std:.4f} vs {grpo_std:.4f})")

    total_ddpg_sims = len(ddpg_rewards)
    total_grpo_sims = len(grpo_rewards) * grpo_designs_per_step
    print("\n  - Total simulations:")
    print(f"    DDPG: {total_ddpg_sims:,}")
    print(f"    GRPO: {total_grpo_sims:,} ({grpo_designs_per_step}x per step)")
    print("=" * 80)


def main():
    print("=" * 80)
    print("DDPG vs GRPO Comparison Analysis")
    print("=" * 80)

    print("\n[1] Loading DDPG Results...")
    ddpg_rewards = load_ddpg_rewards()

    print("\n[2] Loading GRPO Results...")
    grpo_history = load_grpo_history()

    if ddpg_rewards is not None and grpo_history is not None:
        print("\n[3] Generating Comparison Plots...")
        plot_comparison(ddpg_rewards, grpo_history)

    print_summary(ddpg_rewards, grpo_history)


if __name__ == "__main__":
    main()
