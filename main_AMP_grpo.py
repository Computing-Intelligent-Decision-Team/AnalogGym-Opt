"""Command-line entry point for the AnalogGym-Opt GRPO demo."""

import argparse
import inspect
import json
import os
import pickle
from datetime import datetime
from typing import Any, Dict, List

import numpy as np
import torch

from AmpEnv import AmpEnv
from circuit_config_loader import CircuitConfigLoader
from dev_params import DeviceParams
from grpo import GRPOAgent, Episode
from models import (
    ActorCriticGAT,
    ActorCriticGCN,
    ActorCriticMLP,
    ActorCriticRGCN,
    PolicyNetRGCN,
)
from reporting_metrics import (
    format_reporting_value,
    get_filtered_performance,
    get_reporting_config_keys,
    get_reporting_signals,
    iter_reporting_metric_items,
)
from utils import ActionNormalizer

date = datetime.today().strftime("%Y-%m-%d")
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# Configuration - Generic circuit settings
CIRCUIT_NAME = "amp_dfcfc2"
GNN = PolicyNetRGCN
QUICK_CONFIG = {
    "num_steps": 300,
    "enable_full_pvt_training": False,
    "enable_pvt_outer_loop": True,
}

MODE_TO_QUICK_CONFIG = {
    "tt-only": {
        "enable_full_pvt_training": False,
        "enable_pvt_outer_loop": False,
    },
    "tt-proxy": {
        "enable_full_pvt_training": False,
        "enable_pvt_outer_loop": True,
    },
    "full-pvt": {
        "enable_full_pvt_training": True,
        "enable_pvt_outer_loop": False,
    },
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run the AnalogGym-Opt GRPO demo.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--circuit", default=CIRCUIT_NAME, help="Circuit config name.")
    parser.add_argument("--steps", type=int, default=QUICK_CONFIG["num_steps"], help="Training steps.")
    parser.add_argument(
        "--mode",
        choices=sorted(MODE_TO_QUICK_CONFIG),
        default="tt-proxy",
        help="Training runtime mode.",
    )
    return parser.parse_args(argv)


def _rank_episodes(agent: GRPOAgent, episodes: List[Episode]) -> List[Episode]:
    if hasattr(agent, "_episode_rank_tuple"):
        return sorted(episodes, key=lambda ep: agent._episode_rank_tuple(ep), reverse=True)
    return sorted(episodes, key=lambda ep: float(ep.reward), reverse=True)


def _pareto_front_episodes(agent: GRPOAgent, episodes: List[Episode]) -> List[Episode]:
    feasible_episodes = [ep for ep in episodes if bool(getattr(ep, "pm_feasible", False))]
    if not feasible_episodes:
        return []

    dominates = getattr(agent, "_dominates_vector", None)
    objective_keys = list(getattr(agent, "objective_keys", []) or [])
    if dominates is None or not objective_keys:
        return _rank_episodes(agent, feasible_episodes)

    def objective_vector(ep: Episode) -> np.ndarray:
        return np.asarray(
            [float((ep.objective_rewards or {}).get(key, -1e6)) for key in objective_keys],
            dtype=np.float64,
        )

    archive: List[Episode] = []
    archive_vectors: List[np.ndarray] = []

    for ep in feasible_episodes:
        candidate_vec = objective_vector(ep)
        if candidate_vec.shape[0] != len(objective_keys) or not np.all(np.isfinite(candidate_vec)):
            continue

        keep_eps: List[Episode] = []
        keep_vecs: List[np.ndarray] = []
        dominated = False

        for existing_ep, existing_vec in zip(archive, archive_vectors):
            if np.allclose(existing_vec, candidate_vec, atol=1e-8, rtol=1e-8):
                existing_rank = (
                    tuple(agent._episode_rank_tuple(existing_ep))
                    if hasattr(agent, "_episode_rank_tuple")
                    else (float(existing_ep.reward),)
                )
                candidate_rank = (
                    tuple(agent._episode_rank_tuple(ep))
                    if hasattr(agent, "_episode_rank_tuple")
                    else (float(ep.reward),)
                )
                if candidate_rank > existing_rank:
                    continue
                dominated = True
                keep_eps.append(existing_ep)
                keep_vecs.append(existing_vec)
                continue
            if dominates(existing_vec, candidate_vec):
                dominated = True
                keep_eps.append(existing_ep)
                keep_vecs.append(existing_vec)
                continue
            if dominates(candidate_vec, existing_vec):
                continue
            keep_eps.append(existing_ep)
            keep_vecs.append(existing_vec)

        if not dominated:
            keep_eps.append(ep)
            keep_vecs.append(candidate_vec)

        archive = keep_eps
        archive_vectors = keep_vecs

    return _rank_episodes(agent, archive)


def _real_action_from_episode(env: AmpEnv, episode: Episode):
    performance = episode.performance or {}
    real_action = performance.get("real_action")
    if real_action is not None:
        return real_action
    try:
        return ActionNormalizer(
            action_space_low=env.action_space_low,
            action_space_high=env.action_space_high,
            action_space_step=env.action_space_step,
        ).action(np.array(episode.action, copy=True)).tolist()
    except Exception:
        return None


def _episode_record(agent: GRPOAgent, env: AmpEnv, episode: Episode, rank: int) -> Dict[str, Any]:
    rank_tuple = (
        list(agent._episode_rank_tuple(episode))
        if hasattr(agent, "_episode_rank_tuple")
        else [float(episode.reward)]
    )
    return {
        "rank": int(rank),
        "circuit_spec": episode.circuit_spec,
        "design_idx": int(episode.design_idx),
        "training_reward": float(episode.reward),
        "reward": float(episode.reward),
        "utility": float(getattr(episode, "utility", episode.reward)),
        "pm_feasible": bool(episode.pm_feasible),
        "pm_violation": float(episode.pm_violation),
        "selected_corner_idx": int(getattr(episode, "selected_corner_idx", -1)),
        "evaluation_source": getattr(episode, "evaluation_source", "unknown"),
        "rank_tuple": agent._json_safe(rank_tuple),
        "objective_rewards": agent._json_safe(dict(episode.objective_rewards or {})),
        "action_normalized": agent._json_safe(np.asarray(episode.action, dtype=np.float32)),
        "action_real": agent._json_safe(_real_action_from_episode(env, episode)),
        "performance": agent._json_safe(get_filtered_performance(episode.performance or {})),
        "performance_signals": agent._json_safe(get_reporting_signals(episode.performance or {})),
    }


def _record_detail_lines(record: Dict[str, Any], prefix: str = "  ") -> List[str]:
    lines = [
        (
            f"{prefix}design_idx={record.get('design_idx')} "
            f"source={record.get('evaluation_source')} "
            f"training_reward={format_reporting_value(record.get('training_reward'))} "
            f"utility={format_reporting_value(record.get('utility'))} "
            f"pm_violation={format_reporting_value(record.get('pm_violation'))}"
        )
    ]
    perf = record.get("performance") or {}
    if perf:
        lines.append(f"{prefix}raw_performance:")
        for key, value in iter_reporting_metric_items(perf):
            lines.append(f"{prefix}  {key}: {format_reporting_value(value)}")
    return lines


def _save_top_designs(
    agent: GRPOAgent,
    env: AmpEnv,
    circuit_name: str,
    episodes: List[Episode],
    subdir: str = "top_designs",
    title: str = "",
) -> List[Dict[str, Any]]:
    out_dir = os.path.join(agent._training_saves_dir, subdir)
    os.makedirs(out_dir, exist_ok=True)

    feasible_episodes = [episode for episode in episodes if bool(getattr(episode, "pm_feasible", False))]
    if not title:
        title = (
            f"Verified PVT Pareto candidates ({len(feasible_episodes)} PM-feasible designs) for {circuit_name}"
            if feasible_episodes and all(getattr(ep, "evaluation_source", "") == "full_pvt" for ep in feasible_episodes)
            else f"Final Pareto candidates ({len(feasible_episodes)} PM-feasible designs) for {circuit_name}"
        )
    records: List[Dict[str, Any]] = []
    summary_lines = [title, "=" * 80]

    for idx, episode in enumerate(feasible_episodes, start=1):
        record = _episode_record(agent, env, episode, rank=idx)
        records.append(record)

        per_design_path = os.path.join(out_dir, f"top_{idx:02d}_design.json")
        with open(per_design_path, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)

        summary_lines.append(f"#{idx}")
        summary_lines.extend(_record_detail_lines(record))
        summary_lines.append("-" * 80)

    if not records:
        summary_lines.append("No PM-feasible designs available for saving.")

    summary_json_path = os.path.join(out_dir, "top_designs_summary.json")
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    summary_txt_path = os.path.join(out_dir, "top_designs_summary.txt")
    with open(summary_txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines) + "\n")

    print(f"[OK] Top-{len(records)} PM-feasible design records saved to: {out_dir}")
    return records


def _resolve_reporting_schedule(config: Dict[str, Any]) -> None:
    total_steps = max(1, int(config.get("num_steps", 1)))
    designs_per_step = max(1, int(config.get("num_designs_per_circuit", 1)))

    if int(config.get("eval_interval", 0)) <= 0:
        config["eval_interval"] = max(25, min(100, total_steps // 4 if total_steps >= 4 else 1))
    if int(config.get("save_interval", 0)) <= 0:
        config["save_interval"] = max(
            int(config["eval_interval"]),
            min(200, max(50, total_steps // 2 if total_steps >= 2 else 1)),
        )
    if int(config.get("final_eval_num_designs", 0)) <= 0:
        config["final_eval_num_designs"] = max(
            20,
            min(80, int(round(total_steps * 0.08)) + 2 * designs_per_step),
        )
    if int(config.get("recommended_num_designs", 0)) <= 0:
        config["recommended_num_designs"] = max(
            5,
            min(8, max(5, int(round(config["final_eval_num_designs"] / 6.0)))),
        )


def _record_rank_tuple(record: Dict[str, Any]) -> tuple:
    rank_tuple = record.get("rank_tuple", [])
    if isinstance(rank_tuple, (list, tuple)):
        try:
            return tuple(float(x) for x in rank_tuple)
        except Exception:
            return tuple()
    return tuple()


def _record_objective_vector(record: Dict[str, Any], objective_keys: List[str]) -> np.ndarray:
    if record.get("pareto_objective_vector") is not None:
        try:
            return np.asarray(record.get("pareto_objective_vector", []), dtype=np.float64)
        except Exception:
            pass
    objective_rewards = record.get("objective_rewards", {}) or {}
    return np.asarray(
        [float(objective_rewards.get(key, -1e6)) for key in objective_keys],
        dtype=np.float64,
    )


def _merge_recommendation_records(
    agent: GRPOAgent,
    records: List[Dict[str, Any]],
    candidate_limit: int = 5,
) -> List[Dict[str, Any]]:
    if not records:
        return []

    source_priority = {
        "final_test": 4,
        "historical_archive": 3,
        "historical_objective_best": 2,
        "historical_best": 1,
    }
    objective_keys = list(getattr(agent, "objective_keys", []) or [])
    if not objective_keys:
        return records[:candidate_limit]

    dedup: Dict[tuple, Dict[str, Any]] = {}
    for record in records:
        if not bool(record.get("pm_feasible", False)):
            continue
        vector = _record_objective_vector(record, objective_keys)
        if vector.shape[0] != len(objective_keys) or not np.all(np.isfinite(vector)):
            continue
        key = tuple(np.round(vector.astype(np.float64), 8).tolist())
        candidate = dict(record)
        candidate["pareto_objective_vector"] = vector.tolist()
        current = dedup.get(key)
        candidate_key = (
            int(source_priority.get(candidate.get("candidate_source", ""), 0)),
            _record_rank_tuple(candidate),
        )
        current_key = (
            int(source_priority.get((current or {}).get("candidate_source", ""), 0)),
            _record_rank_tuple(current or {}),
        )
        if current is None or candidate_key > current_key:
            dedup[key] = candidate

    candidates = list(dedup.values())
    archive: List[Dict[str, Any]] = []
    archive_vectors: List[np.ndarray] = []
    for record in candidates:
        candidate_vec = np.asarray(record.get("pareto_objective_vector", []), dtype=np.float64)
        keep_records: List[Dict[str, Any]] = []
        keep_vectors: List[np.ndarray] = []
        dominated = False
        for existing_record, existing_vec in zip(archive, archive_vectors):
            if np.allclose(existing_vec, candidate_vec, atol=1e-8, rtol=1e-8):
                existing_key = (
                    int(source_priority.get(existing_record.get("candidate_source", ""), 0)),
                    _record_rank_tuple(existing_record),
                )
                candidate_key = (
                    int(source_priority.get(record.get("candidate_source", ""), 0)),
                    _record_rank_tuple(record),
                )
                if candidate_key > existing_key:
                    continue
                dominated = True
                keep_records.append(existing_record)
                keep_vectors.append(existing_vec)
                continue
            if agent._dominates_vector(existing_vec, candidate_vec):
                dominated = True
                keep_records.append(existing_record)
                keep_vectors.append(existing_vec)
                continue
            if agent._dominates_vector(candidate_vec, existing_vec):
                continue
            keep_records.append(existing_record)
            keep_vectors.append(existing_vec)
        if not dominated:
            keep_records.append(record)
            keep_vectors.append(candidate_vec)
        archive = keep_records
        archive_vectors = keep_vectors

    archive.sort(
        key=lambda record: (
            int(source_priority.get(record.get("candidate_source", ""), 0)),
            _record_rank_tuple(record),
        ),
        reverse=True,
    )
    return archive[:candidate_limit]


def _save_recommended_records(
    agent: GRPOAgent,
    records: List[Dict[str, Any]],
    subdir: str,
    title: str,
) -> List[Dict[str, Any]]:
    out_dir = os.path.join(agent._training_saves_dir, subdir)
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "recommended_candidates.json")
    txt_path = os.path.join(out_dir, "recommended_candidates.txt")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    lines = [title, "=" * 80]
    if not records:
        lines.append("No PM-feasible recommended candidates.")
    else:
        for idx, record in enumerate(records, start=1):
            lines.append(
                f"#{idx} source={record.get('candidate_source')} "
                f"training_reward={format_reporting_value(record.get('training_reward'))} "
                f"pm_violation={format_reporting_value(record.get('pm_violation'))}"
            )
            lines.extend(_record_detail_lines(record, prefix="  "))
            lines.append("-" * 80)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return records


def main(argv=None):
    args = parse_args(argv)
    global CIRCUIT_NAME, QUICK_CONFIG
    CIRCUIT_NAME = args.circuit
    QUICK_CONFIG = {
        "num_steps": max(1, int(args.steps)),
        **MODE_TO_QUICK_CONFIG[args.mode],
    }

    print("=" * 80)
    print("GRPO Training for Analog Circuit Optimization")
    print(f"Circuit: {CIRCUIT_NAME}")
    config_loader = CircuitConfigLoader()
    circuit_config = config_loader.get_circuit_config(CIRCUIT_NAME)
    print(f"[OK] Circuit configuration loaded for: {CIRCUIT_NAME}")
    print(f"  Device parameters: {list(circuit_config['device'].keys())}")
    print(f"  Reported metrics: {get_reporting_config_keys(circuit_config['performance'])}")
    print("=" * 80)

    dev_initial = False
    if dev_initial:
        ckt_hierarchy = tuple(tuple(item) for item in circuit_config.get("ckt_hierarchy", ()))
        dev_params_script = DeviceParams(ckt_hierarchy).gen_dev_params(
            file_name=os.path.basename(circuit_config["paths"]["op_results_path"])
        )
        with open(circuit_config["paths"]["dev_params_path"], "w") as f:
            for line in dev_params_script:
                f.write(f"{line}\n")

    op_initial = False
    if op_initial:
        print("\n[Step 0] Generating OP statistics...")
        env = AmpEnv(circuit_config)
        env._init_random_sim(100)
        print(f"[OK] OP statistics generated and saved to {CIRCUIT_NAME}_op_mean_std.json")

    print("\n[Step 1] Initializing environment...")
    env = AmpEnv(circuit_config)
    dummy_state, dummy_info = env.reset()

    quick_config = dict(QUICK_CONFIG)

    resolved_run_all_corners = bool(quick_config["enable_full_pvt_training"])
    resolved_pvt_outer_loop = bool(quick_config["enable_pvt_outer_loop"]) and not resolved_run_all_corners
    resolved_use_vae = bool(resolved_run_all_corners or resolved_pvt_outer_loop)

    config = {
        "num_steps": max(1, int(quick_config.get("num_steps", 200))),
        "num_circuits_per_step": 1,
        "num_designs_per_circuit": 8,
        "learning_rate": 1e-4,
        "max_grad_norm": 0.5,
        "eval_interval": 0,
        "save_interval": 0,
        "final_eval_num_designs": 0,
        "recommended_num_designs": 0,
        "device": circuit_config.get("train_device", "cpu") if torch.cuda.is_available() else "cpu",
        "quick_config": quick_config,
        "run_all_corners": resolved_run_all_corners,
        "pvt_outer_loop_enabled": resolved_pvt_outer_loop,
        "pvt_verify_topk_per_step": 1,
        "pvt_verify_uncertain_per_step": 1,
        "pvt_warmup_steps": 0,
        "pvt_warmup_verify_topk_per_step": 1,
        "pvt_warmup_diverse_per_step": 1,
        "pvt_offline_epochs": 30,
        "pvt_offline_patience": 8,
        "pvt_proxy_finetune_interval": 1,
        "pvt_proxy_finetune_epochs": 6,
        "pvt_proxy_finetune_patience": 3,
        "pvt_proxy_beta": 1.0,
        "pvt_proxy_prediction_samples": 20,
        "pvt_verified_archive_capacity": 64,
        "use_vae": resolved_use_vae,
        "vae_min_samples": 64,
        "vae_diagnostics_enabled": resolved_use_vae,
        "vae_save_per_objective_plots": False,
        "objective_keys": [
            "constraint_reward",
            "FOML_score",
            "FOMS_score",
            "Active_Area_score",
        ],
        "objective_weights": {
            "constraint_reward": 1.0,
            "FOML_score": 1.0,
            "FOMS_score": 1.0,
            "Active_Area_score": 1.0,
        },
        "covariance_whitening_enabled": True,
        "covariance_whitening_warmup_steps": 100,
        "covariance_whitening_min_samples": 128,
        "covariance_whitening_shrinkage": 0.25,
        "covariance_whitening_buffer_capacity": 4096,
        "utility_temperature": 0.15,
        "utility_baseline_temperature": 0.2,
        "utility_reference_alpha": 0.1,
        "utility_reference_update_interval": 10,
        "utility_reference_quantile_high": 0.9,
        "utility_reference_quantile_low": 0.1,
        "utility_reference_clip_delta": 0.25,
        "utility_archive_capacity": 512,
        "utility_norm_clip": 2.0,
        "constraint_base_penalty": 1.25,
        "constraint_violation_scale": 2.0,
        "constraint_infeasible_residual": 0.05,
        "use_wandb": False,
        "wandb_project": f"{CIRCUIT_NAME.upper()}_GRPO",
        "wandb_run_name": f"grpo_{CIRCUIT_NAME}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    }
    _resolve_reporting_schedule(config)

    print("\n" + "=" * 80)
    print("Quick Config:")
    print("=" * 80)
    for key, value in quick_config.items():
        print(f"  {key:30s}: {value}")
    print("=" * 80)

    print("\n" + "=" * 80)
    print("GRPO Configuration:")
    print("=" * 80)
    for key, value in config.items():
        print(f"  {key:30s}: {value}")
    print("=" * 80)

    print("\n[Step 3] Initializing GRPO agent...")
    policy_network = GNN().Actor(circuit_config)

    wandb_logger = None
    if config.get("use_wandb", False):
        try:
            import wandb as _wandb

            wandb_logger = _wandb
            _wandb.init(project=config["wandb_project"], name=config["wandb_run_name"], config=config)
        except Exception as e:
            print(f"[wandb] init failed: {e}. wandb disabled.")
            wandb_logger = None

    agent_kwargs = dict(
        env=env,
        policy_network=policy_network,
        circuit_name=CIRCUIT_NAME,
        num_designs_per_circuit=config["num_designs_per_circuit"],
        learning_rate=config["learning_rate"],
        max_grad_norm=config["max_grad_norm"],
        device=config["device"],
        run_all_corners=config.get("run_all_corners"),
        pvt_outer_loop_enabled=config.get("pvt_outer_loop_enabled"),
        pvt_verify_topk_per_step=config.get("pvt_verify_topk_per_step"),
        pvt_verify_uncertain_per_step=config.get("pvt_verify_uncertain_per_step"),
        pvt_warmup_steps=config.get("pvt_warmup_steps"),
        pvt_warmup_verify_topk_per_step=config.get("pvt_warmup_verify_topk_per_step"),
        pvt_warmup_diverse_per_step=config.get("pvt_warmup_diverse_per_step"),
        pvt_offline_epochs=config.get("pvt_offline_epochs"),
        pvt_offline_patience=config.get("pvt_offline_patience"),
        pvt_proxy_finetune_interval=config.get("pvt_proxy_finetune_interval"),
        pvt_proxy_finetune_epochs=config.get("pvt_proxy_finetune_epochs"),
        pvt_proxy_finetune_patience=config.get("pvt_proxy_finetune_patience"),
        pvt_proxy_beta=config.get("pvt_proxy_beta"),
        pvt_proxy_prediction_samples=config.get("pvt_proxy_prediction_samples"),
        pvt_verified_archive_capacity=config.get("pvt_verified_archive_capacity"),
        initial_reset_state=dummy_state,
        initial_reset_info=dummy_info,
        use_vae=config.get("use_vae"),
        vae_min_samples=config.get("vae_min_samples"),
        vae_diagnostics_enabled=config.get("vae_diagnostics_enabled"),
        vae_save_per_objective_plots=config.get("vae_save_per_objective_plots"),
        objective_keys=config.get("objective_keys"),
        objective_weights=config.get("objective_weights"),
        covariance_whitening_enabled=config.get("covariance_whitening_enabled"),
        covariance_whitening_warmup_steps=config.get("covariance_whitening_warmup_steps"),
        covariance_whitening_min_samples=config.get("covariance_whitening_min_samples"),
        covariance_whitening_shrinkage=config.get("covariance_whitening_shrinkage"),
        covariance_whitening_buffer_capacity=config.get("covariance_whitening_buffer_capacity"),
        utility_temperature=config.get("utility_temperature"),
        utility_baseline_temperature=config.get("utility_baseline_temperature"),
        utility_reference_alpha=config.get("utility_reference_alpha"),
        utility_reference_update_interval=config.get("utility_reference_update_interval"),
        utility_reference_quantile_high=config.get("utility_reference_quantile_high"),
        utility_reference_quantile_low=config.get("utility_reference_quantile_low"),
        utility_reference_clip_delta=config.get("utility_reference_clip_delta"),
        utility_archive_capacity=config.get("utility_archive_capacity"),
        utility_norm_clip=config.get("utility_norm_clip"),
        constraint_base_penalty=config.get("constraint_base_penalty"),
        constraint_violation_scale=config.get("constraint_violation_scale"),
        constraint_infeasible_residual=config.get("constraint_infeasible_residual"),
    )
    if "wandb_logger" in inspect.signature(GRPOAgent.__init__).parameters:
        agent_kwargs["wandb_logger"] = wandb_logger
    try:
        agent = GRPOAgent(**agent_kwargs)
    except TypeError:
        if "wandb_logger" in agent_kwargs:
            agent_kwargs.pop("wandb_logger", None)
        agent = GRPOAgent(**agent_kwargs)
    agent.reporting_schedule_summary = {
        "eval_interval": int(config["eval_interval"]),
        "save_interval": int(config["save_interval"]),
        "final_eval_num_designs": int(config["final_eval_num_designs"]),
        "recommended_num_designs": int(config["recommended_num_designs"]),
    }
    if getattr(agent, "pvt_outer_loop_enabled", False) and hasattr(agent, "configure_adaptive_pvt_schedule"):
        agent.configure_adaptive_pvt_schedule(config["num_steps"])

    print("[OK] GRPO Agent created")
    print(f"  Policy Network: {GNN().__class__.__name__}")
    print(f"  Device: {config['device']}")
    print(f"  PVT Enabled: {agent.run_all_corners}")
    print(f"  PVT Outer-Loop Enabled: {getattr(agent, 'pvt_outer_loop_enabled', False)}")
    print(f"  VAE Enabled: {agent.use_vae}")
    print(f"  Reporting Schedule: {getattr(agent, 'reporting_schedule_summary', {})}")
    print("  Effective PVT/VAE Runtime:")
    print(f"    TT Inner-Loop Training: {not agent.run_all_corners}")
    print(f"    Real Full-PVT In Training Loop: {agent.run_all_corners}")
    print(f"    Online PVT Proxy: {getattr(agent, 'pvt_outer_loop_enabled', False)}")
    print(f"    Effective VAE Switch: {agent.use_vae}")
    if getattr(agent, "pvt_outer_loop_enabled", False):
        print(
            "    Real PVT Verify Budget Per Step: "
            f"top-{getattr(agent, 'pvt_verify_topk_per_step', 0)} + "
            f"uncertain-{getattr(agent, 'pvt_verify_uncertain_per_step', 0)}"
        )
        print(
            "    Warmup Verify Budget Per Step: "
            f"top-{getattr(agent, 'pvt_warmup_verify_topk_per_step', 0)} + "
            f"diverse-{getattr(agent, 'pvt_warmup_diverse_per_step', 0)}"
        )
        print(f"    PVT Warmup Steps: {getattr(agent, 'pvt_warmup_steps', 'N/A')}")
        print(f"    VAE Warmup Samples: {getattr(agent, 'vae_min_samples', 'N/A')}")
        print(
            "    VAE Offline/Block Fine-Tune: "
            f"{getattr(agent, 'pvt_offline_epochs', 'N/A')} epochs + "
            f"every {getattr(agent, 'pvt_proxy_finetune_interval', 'N/A')} steps for "
            f"{getattr(agent, 'pvt_proxy_finetune_epochs', 'N/A')} epochs"
        )
        print(
            "    Adaptive PVT Summary: "
            f"{getattr(agent, '_adaptive_pvt_schedule_summary', {})}"
        )
        print("    Final Evaluation Uses Real PVT: True")
    print(f"  Multi-Objective Keys: {agent.objective_keys}")
    print(f"  Objective Weights: {agent.objective_weights}")
    print("  Utility Mode: PM-first smooth Tchebycheff")
    print(f"  Advantage Mode: {getattr(agent, 'advantage_mode', 'scalar_utility')}")
    print(f"  Objective Normalization: {getattr(agent, 'objective_normalization_mode', 'legacy')}")
    print(f"  Covariance Whitening Enabled: {getattr(agent, 'covariance_whitening_enabled', False)}")

    dummy_state_tensor = torch.FloatTensor(dummy_state).unsqueeze(0).to(config["device"])
    with torch.no_grad():
        _ = policy_network(dummy_state_tensor)
    print(f"  Total Parameters: {sum(p.numel() for p in policy_network.parameters()):,}")

    print("\n[Step 4] Starting training...")
    agent.train(
        num_steps=config["num_steps"],
        num_circuits_per_step=config["num_circuits_per_step"],
        eval_interval=config["eval_interval"],
        save_interval=config["save_interval"],
    )
    print("\n[OK] Training complete!")

    print("\n[Step 5] Running final evaluation...")
    final_eval_num_designs = max(1, int(config.get("final_eval_num_designs", 20)))
    recommended_num_designs = max(1, int(config.get("recommended_num_designs", 5)))
    print(f"\nGenerating final TT designs ({final_eval_num_designs})...")
    final_episodes: List[Episode] = []

    for i in range(final_eval_num_designs):
        state, _ = env.reset()
        action = agent.select_action(state)
        _, rewards_eval, _, _, infos_eval = env.parallel_step([action], enable_pvt=agent.run_all_corners)
        info = infos_eval[0] if len(infos_eval) > 0 else {}

        final_episode = agent._build_episode_from_perf(
            circuit_spec=f"final_{i}",
            state=state,
            action=action,
            perf_info=info,
            design_idx=i,
            old_log_prob=0.0,
            verbose=False,
        )
        final_episodes.append(final_episode)

        print(
            f"  Design {i+1}/{final_eval_num_designs}: training_reward={final_episode.reward:.4f}, "
            f"pm_feasible={final_episode.pm_feasible}, pm_violation={final_episode.pm_violation:.4f}"
        )

    tt_ranked_final_episodes = _rank_episodes(agent, final_episodes)
    tt_pareto_episodes = _pareto_front_episodes(agent, tt_ranked_final_episodes)
    tt_best_episode = tt_pareto_episodes[0] if tt_pareto_episodes else tt_ranked_final_episodes[0]

    historical_tt_records: List[Dict[str, Any]] = []
    best_performance_record = getattr(agent, "best_performance_record", None)
    if isinstance(best_performance_record, dict):
        historical_tt_records.append({**best_performance_record, "candidate_source": "historical_best"})
    for record in list(getattr(agent, "best_pareto_records", []) or []):
        if isinstance(record, dict):
            historical_tt_records.append({**record, "candidate_source": "historical_archive"})
    for record in list((getattr(agent, "best_objective_records", {}) or {}).values()):
        if isinstance(record, dict):
            historical_tt_records.append({**record, "candidate_source": "historical_objective_best"})
    final_tt_records = [
        {**_episode_record(agent, env, episode, rank=rank), "candidate_source": "final_test"}
        for rank, episode in enumerate(tt_pareto_episodes, start=1)
    ]
    recommended_tt_candidates = _merge_recommendation_records(
        agent,
        historical_tt_records + final_tt_records,
        candidate_limit=recommended_num_designs,
    )

    verified_final_episodes: List[Episode] = []
    verified_pvt_pareto_episodes: List[Episode] = []
    verified_pvt_best_episode: Episode = None
    recommended_verified_pvt_candidates: List[Dict[str, Any]] = []
    if getattr(agent, "pvt_outer_loop_enabled", False):
        print("\nVerifying final designs under real full-corner PVT...")
        verified_final_episodes = agent._verify_pvt_episodes(
            final_episodes,
            step=max(1, int(getattr(agent, "total_steps", 0))),
            update_dataset=False,
            update_archive=False,
        )
        if verified_final_episodes:
            verified_ranked_final_episodes = _rank_episodes(agent, verified_final_episodes)
            verified_pvt_pareto_episodes = _pareto_front_episodes(agent, verified_ranked_final_episodes)
            verified_pvt_best_episode = (
                verified_pvt_pareto_episodes[0]
                if verified_pvt_pareto_episodes
                else verified_ranked_final_episodes[0]
            )
            print(f"  Verified {len(verified_final_episodes)}/{len(final_episodes)} final designs with real PVT.")
        else:
            print("  Final real-PVT verification failed; verified-PVT candidate list is empty.")

        historical_verified_pvt_records = [
            {**record, "candidate_source": "historical_archive"}
            for record in list(getattr(agent, "verified_pvt_pareto_records", []) or [])
            if isinstance(record, dict)
        ]
        final_verified_pvt_records = [
            {**_episode_record(agent, env, episode, rank=rank), "candidate_source": "final_test"}
            for rank, episode in enumerate(verified_pvt_pareto_episodes, start=1)
        ]
        recommended_verified_pvt_candidates = _merge_recommendation_records(
            agent,
            historical_verified_pvt_records + final_verified_pvt_records,
            candidate_limit=recommended_num_designs,
        )

    if tt_pareto_episodes:
        print(
            f"\nFinal TT Pareto Candidate Designs "
            f"({len(tt_pareto_episodes)} PM-feasible non-dominated solutions):"
        )
        for rank, episode in enumerate(tt_pareto_episodes, start=1):
            record = _episode_record(agent, env, episode, rank=rank)
            print(f"  #{rank}")
            for line in _record_detail_lines(record, prefix="    "):
                print(line)
    else:
        print("\nNo PM-feasible TT Pareto candidate found in the final evaluation set.")

    if recommended_tt_candidates:
        print(
            f"\nRecommended TT Candidates "
            f"(historical bests + final-test merged, top {len(recommended_tt_candidates)}):"
        )
        for rank, record in enumerate(recommended_tt_candidates, start=1):
            print(f"  #{rank} source={record.get('candidate_source')}")
            for line in _record_detail_lines(record, prefix="    "):
                print(line)
    else:
        print("\nNo PM-feasible TT recommendation candidate is available.")

    if verified_pvt_pareto_episodes:
        print(
            f"\nFinal Verified PVT Pareto Candidate Designs "
            f"({len(verified_pvt_pareto_episodes)} PM-feasible non-dominated solutions):"
        )
        for rank, episode in enumerate(verified_pvt_pareto_episodes, start=1):
            record = _episode_record(agent, env, episode, rank=rank)
            print(f"  #{rank}")
            for line in _record_detail_lines(record, prefix="    "):
                print(line)
    elif getattr(agent, "pvt_outer_loop_enabled", False):
        print("\nNo PM-feasible verified-PVT Pareto candidate found in the final verification set.")

    if getattr(agent, "pvt_outer_loop_enabled", False):
        if recommended_verified_pvt_candidates:
            print(
                f"\nRecommended Verified-PVT Candidates "
                f"(historical verified archive + final-test merged, top {len(recommended_verified_pvt_candidates)}):"
            )
            for rank, record in enumerate(recommended_verified_pvt_candidates, start=1):
                print(f"  #{rank} source={record.get('candidate_source')}")
                for line in _record_detail_lines(record, prefix="    "):
                    print(line)
        else:
            print("\nNo PM-feasible verified-PVT recommendation candidate is available.")

    print(f"\n{'=' * 80}")
    print("Representative TT Design Found:")
    print(f"{'=' * 80}")
    print(f"  Training Reward: {tt_best_episode.reward:.4f}")
    print(f"  PM Feasible: {tt_best_episode.pm_feasible}")
    print(f"  PM Violation: {tt_best_episode.pm_violation:.4f}")
    print("  Raw Performance Metrics:")
    for key, value in iter_reporting_metric_items(tt_best_episode.performance):
        print(f"    {key:20s}: {format_reporting_value(value)}")
    print(f"{'=' * 80}")

    if verified_pvt_best_episode is not None:
        print(f"\n{'=' * 80}")
        print("Representative Verified-PVT Design Found:")
        print(f"{'=' * 80}")
        print(f"  Training Reward: {verified_pvt_best_episode.reward:.4f}")
        print(f"  PM Feasible: {verified_pvt_best_episode.pm_feasible}")
        print(f"  PM Violation: {verified_pvt_best_episode.pm_violation:.4f}")
        print("  Raw Performance Metrics:")
        for key, value in iter_reporting_metric_items(verified_pvt_best_episode.performance):
            print(f"    {key:20s}: {format_reporting_value(value)}")
        print(f"{'=' * 80}")

    print("\nRe-running representative TT design for verification...")
    _, reward_verify_arr, _, _, info_verify_arr = env.parallel_step(
        [tt_best_episode.action],
        enable_pvt=False,
    )
    reward_verify = float(reward_verify_arr[0])
    info_verify = info_verify_arr[0] if len(info_verify_arr) > 0 else {}
    verify_episode = agent._build_episode_from_perf(
        circuit_spec="verify_tt",
        state=tt_best_episode.state,
        action=tt_best_episode.action,
        perf_info=info_verify,
        design_idx=0,
        old_log_prob=0.0,
        verbose=False,
    )
    print(
        f"  TT verification env_reward: {reward_verify:.4f}, "
        f"training_reward={verify_episode.reward:.4f}, "
        f"pm_feasible={verify_episode.pm_feasible}, pm_violation={verify_episode.pm_violation:.4f}"
    )

    if verified_pvt_best_episode is not None:
        print("\nRe-running representative verified-PVT design for verification...")
        _, reward_verify_pvt_arr, _, _, info_verify_pvt_arr = env.parallel_step(
            [verified_pvt_best_episode.action],
            enable_pvt=True,
        )
        reward_verify_pvt = float(reward_verify_pvt_arr[0])
        info_verify_pvt = info_verify_pvt_arr[0] if len(info_verify_pvt_arr) > 0 else {}
        verify_pvt_episode = agent._build_episode_from_perf(
            circuit_spec="verify_pvt",
            state=verified_pvt_best_episode.state,
            action=verified_pvt_best_episode.action,
            perf_info=info_verify_pvt,
            design_idx=0,
            old_log_prob=0.0,
            verbose=False,
        )
        print(
            f"  Verified-PVT env_reward: {reward_verify_pvt:.4f}, "
            f"training_reward={verify_pvt_episode.reward:.4f}, "
            f"pm_feasible={verify_pvt_episode.pm_feasible}, pm_violation={verify_pvt_episode.pm_violation:.4f}"
        )

    print("\n[Step 6] Saving results...")
    tt_top_design_records = _save_top_designs(
        agent,
        env,
        CIRCUIT_NAME,
        tt_pareto_episodes,
        subdir="top_designs_tt",
        title=f"Final TT Pareto candidates ({len(tt_pareto_episodes)} PM-feasible designs) for {CIRCUIT_NAME}",
    )
    verified_pvt_top_design_records: List[Dict[str, Any]] = []
    if getattr(agent, "pvt_outer_loop_enabled", False):
        verified_pvt_top_design_records = _save_top_designs(
            agent,
            env,
            CIRCUIT_NAME,
            verified_pvt_pareto_episodes,
            subdir="top_designs_verified_pvt",
            title=f"Final Verified-PVT Pareto candidates ({len(verified_pvt_pareto_episodes)} PM-feasible designs) for {CIRCUIT_NAME}",
        )
    recommended_tt_records = _save_recommended_records(
        agent,
        recommended_tt_candidates,
        subdir="recommended_candidates_tt",
        title=(
            f"Recommended TT candidates for {CIRCUIT_NAME} "
            f"(merged from historical best records and final TT test candidates)"
        ),
    )
    recommended_verified_pvt_records: List[Dict[str, Any]] = []
    if getattr(agent, "pvt_outer_loop_enabled", False):
        recommended_verified_pvt_records = _save_recommended_records(
            agent,
            recommended_verified_pvt_candidates,
            subdir="recommended_candidates_verified_pvt",
            title=(
                f"Recommended Verified-PVT candidates for {CIRCUIT_NAME} "
                f"(merged from historical verified archive and final verified-PVT candidates)"
            ),
        )

    policy_save_path = (
        f"{agent._training_saves_dir}/Policy_GRPO_{CIRCUIT_NAME}_{date}"
        f"_tt_reward={tt_best_episode.reward:.2f}_{GNN().__class__.__name__}.pth"
    )
    torch.save(agent.policy.state_dict(), policy_save_path)
    print(f"[OK] Policy saved to: {policy_save_path}")

    agent_save_path = f"{agent._training_saves_dir}/GRPOAgent_{date}.pkl"
    try:
        light = (
            agent.to_lightweight()
            if hasattr(agent, "to_lightweight")
            else {
                "policy_state_dict": agent.policy.state_dict(),
                "optimizer_state_dict": agent.optimizer.state_dict(),
            }
        )
        with open(agent_save_path, "wb") as f:
            pickle.dump(light, f)
        print(f"[OK] Agent (lightweight) saved to: {agent_save_path}")
    except Exception as e:
        print(f"[warn] Failed to save lightweight agent: {e}")

    history = {
        "reward_history": agent.reward_history,
        "training_reward_history": agent.reward_history,
        "loss_history": agent.loss_history,
        "success_rate_history": agent.success_rate_history,
        "config": config,
        "reporting_schedule_summary": {
            "eval_interval": int(config["eval_interval"]),
            "save_interval": int(config["save_interval"]),
            "final_eval_num_designs": int(config["final_eval_num_designs"]),
            "recommended_num_designs": int(config["recommended_num_designs"]),
        },
        "best_reward": tt_best_episode.reward,
        "best_training_reward": tt_best_episode.reward,
        "best_pm_feasible": tt_best_episode.pm_feasible,
        "best_pm_violation": tt_best_episode.pm_violation,
        "best_performance": get_filtered_performance(tt_best_episode.performance),
        "best_objective_rewards": tt_best_episode.objective_rewards,
        "best_tt_training_reward": tt_best_episode.reward,
        "best_tt_pm_feasible": tt_best_episode.pm_feasible,
        "best_tt_pm_violation": tt_best_episode.pm_violation,
        "best_tt_performance": get_filtered_performance(tt_best_episode.performance),
        "best_tt_objective_rewards": tt_best_episode.objective_rewards,
        "best_verified_pvt_training_reward": None if verified_pvt_best_episode is None else verified_pvt_best_episode.reward,
        "best_verified_pvt_pm_feasible": None if verified_pvt_best_episode is None else verified_pvt_best_episode.pm_feasible,
        "best_verified_pvt_pm_violation": None if verified_pvt_best_episode is None else verified_pvt_best_episode.pm_violation,
        "best_verified_pvt_performance": None if verified_pvt_best_episode is None else get_filtered_performance(verified_pvt_best_episode.performance),
        "best_verified_pvt_objective_rewards": None if verified_pvt_best_episode is None else verified_pvt_best_episode.objective_rewards,
        "best_performance_record": getattr(agent, "best_performance_record", None),
        "best_pareto_records": getattr(agent, "best_pareto_records", []),
        "verified_pvt_pareto_records": getattr(agent, "verified_pvt_pareto_records", []),
        "best_objective_records": getattr(agent, "best_objective_records", {}),
        "objective_history": getattr(agent, "objective_history", {}),
        "objective_plot_history": getattr(agent, "objective_plot_history", {}),
        "verified_pvt_reward_history": getattr(agent, "verified_pvt_reward_history", []),
        "verified_pvt_pm_violation_history": getattr(agent, "verified_pvt_pm_violation_history", []),
        "verified_pvt_pm_feasible_rate_history": getattr(agent, "verified_pvt_pm_feasible_rate_history", []),
        "verified_pvt_objective_history": getattr(agent, "verified_pvt_objective_history", {}),
        "verified_pvt_objective_plot_history": getattr(agent, "verified_pvt_objective_plot_history", {}),
        "pvt_verified_count_history": getattr(agent, "pvt_verified_count_history", []),
        "pvt_verified_archive_size_history": getattr(agent, "pvt_verified_archive_size_history", []),
        "pvt_phase_history": getattr(agent, "pvt_phase_history", []),
        "top_designs": tt_top_design_records,
        "top_designs_tt": tt_top_design_records,
        "top_designs_verified_pvt": verified_pvt_top_design_records,
        "recommended_tt_candidates": recommended_tt_records,
        "recommended_verified_pvt_candidates": recommended_verified_pvt_records,
    }

    history_save_path = f"{agent._training_saves_dir}/training_history_GRPO_{date}.pkl"
    os.makedirs(agent._training_saves_dir, exist_ok=True)
    with open(history_save_path, "wb") as f:
        pickle.dump(history, f)
    print(f"[OK] Training history saved to: {history_save_path}")

    print("\n" + "=" * 80)
    print("Training Summary")
    print("=" * 80)
    print("  Algorithm: GRPO")
    print(f"  Total Steps: {config['num_steps']}")
    print(f"  Designs per Step: {config['num_designs_per_circuit']}")
    print(f"  Total Designs Evaluated: {config['num_steps'] * config['num_designs_per_circuit']}")
    print(f"  Best TT Training Reward: {tt_best_episode.reward:.4f}")
    print(f"  Best TT PM Feasible: {tt_best_episode.pm_feasible}")
    print(f"  Best TT PM Violation: {tt_best_episode.pm_violation:.4f}")
    print(f"  Recommended TT Candidates: {len(recommended_tt_records)}")
    if verified_pvt_best_episode is not None:
        print(f"  Best Verified-PVT Training Reward: {verified_pvt_best_episode.reward:.4f}")
        print(f"  Best Verified-PVT PM Feasible: {verified_pvt_best_episode.pm_feasible}")
        print(f"  Best Verified-PVT PM Violation: {verified_pvt_best_episode.pm_violation:.4f}")
        print(f"  Recommended Verified-PVT Candidates: {len(recommended_verified_pvt_records)}")
    print(f"  Final Mean Training Reward: {np.mean(agent.reward_history[-10:]):.4f}")
    print(f"  Final Success Rate: {np.mean(agent.success_rate_history[-10:]):.2%}")
    print("=" * 80)

    print("\n[OK] GRPO training script complete!")
    print(f"   Check saved results in: {agent._training_saves_dir}/")


if __name__ == "__main__":
    main()
