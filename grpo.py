"""GRPO agent and training utilities for analog circuit optimization."""

import numpy as np
import math
import copy
import json
from typing import Any, List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import os
from datetime import datetime

import torch
import torch.optim as optim

from IPython.display import clear_output
import matplotlib

# Use non-interactive backend to avoid blocking
try:
    if matplotlib.get_backend().lower() not in ("agg", "pdf", "svg"):
        matplotlib.use("Agg")
except Exception:
    pass

import matplotlib.pyplot as plt

try:
    from draw import (
        log_step as draw_log_step,
        save_plot_data as draw_save_plot_data,
        write_vae_predictions as draw_write_vae_predictions,
        plot_progress as draw_plot_progress,
    )
except Exception:
    import sys as _sys, os as _os

    _sys.path.append(_os.path.dirname(__file__))
    from draw import (
        log_step as draw_log_step,
        save_plot_data as draw_save_plot_data,
        write_vae_predictions as draw_write_vae_predictions,
        plot_progress as draw_plot_progress,
    )

from reward_adapter import RewardAdapter
from reporting_metrics import (
    format_reporting_value,
    get_filtered_performance,
    get_reporting_metrics,
    get_reporting_signals,
    get_reporting_scores,
    iter_reporting_metric_items,
)
from vae_model import (
    RL_OBJECTIVE_KEYS,
    VAE_CONDITION_PERFORMANCE_KEYS,
    VAE_OBJECTIVE_KEYS,
    VAETrainingData,
    WorstCornerVAE,
    build_target_objectives,
    build_vae_condition,
    compute_phase_margin_violation,
    objective_vector_to_dict,
    prepare_vae_training_data,
)


@dataclass
class Episode:
    """Store one sampled or evaluated circuit design."""

    circuit_spec: str = ""
    state: np.ndarray = None
    action: np.ndarray = None
    reward: float = 0.0
    performance: Dict = field(default_factory=dict)
    objective_rewards: Dict[str, float] = field(default_factory=dict)
    objective_advantages: Dict[str, float] = field(default_factory=dict)
    advantage: float = 0.0
    utility: float = 0.0
    pm_violation: float = 0.0
    pm_feasible: bool = False
    selected_corner_idx: int = -1
    design_idx: int = 0
    old_log_prob: float = 0.0  # Log prob under sampling policy for importance ratio
    evaluation_source: str = "unknown"

    def __post_init__(self):
        if self.state is not None and not isinstance(self.state, np.ndarray):
            self.state = np.array(self.state)
        if self.action is not None and not isinstance(self.action, np.ndarray):
            self.action = np.array(self.action)


class GRPOAgent:
    """
    GRPO Agent for Circuit Parameter Optimization (continuous actions).

    Improvements:
    - Gaussian policy std used in log_prob matches exploration noise_sigma.
    - Optional advantage shaping (tanh / clipping) to reduce the impact of outliers.
    - Group-level weighting: groups whose mean reward is closer to 0
      (i.e., better) get higher weight in the policy gradient.
    - Extensive NaN / Inf guards to stabilise training.
    """

    def __init__(
        self,
        env,
        policy_network,
        circuit_name: Optional[str] = None,
        num_designs_per_circuit: int = 8,
        learning_rate: float = 1e-4,
        max_grad_norm: float = 0.5,
        gamma: float = 0.99,
        device: str = "cpu",
        reward_adapter: Optional[RewardAdapter] = None,
        reward_strategy: str = "multi_level",
        run_all_corners: bool = False,
        pvt_num_processes: Optional[int] = None,
        initial_reset_state: Optional[np.ndarray] = None,
        initial_reset_info: Optional[Dict] = None,
        wandb_logger: Optional[object] = None,
        pvt_outer_loop_enabled: bool = False,
        pvt_verify_topk_per_step: int = 1,
        pvt_verify_uncertain_per_step: int = 1,
        pvt_proxy_beta: float = 1.0,
        pvt_proxy_prediction_samples: int = 20,
        pvt_proxy_online_update_interval: int = 5,
        pvt_proxy_online_updates_per_step: int = 2,
        pvt_proxy_online_batch_size: int = 32,
        pvt_verified_archive_capacity: int = 64,
        pvt_warmup_steps: int = 100,
        pvt_warmup_verify_topk_per_step: int = 1,
        pvt_warmup_diverse_per_step: int = 1,
        pvt_offline_epochs: int = 100,
        pvt_offline_patience: int = 12,
        pvt_proxy_finetune_interval: int = 10,
        pvt_proxy_finetune_epochs: int = 5,
        pvt_proxy_finetune_patience: int = 3,
        use_vae: Optional[bool] = None,
        vae_min_samples: Optional[int] = None,
        vae_reward_threshold: Optional[float] = None,
        vae_diagnostics_enabled: bool = True,
        vae_save_per_objective_plots: bool = True,
        use_kan: Optional[bool] = None,
        kan_min_samples: Optional[int] = None,
        kan_reward_threshold: Optional[float] = None,
        # --- PPO/GRPO hyper-parameters ---
        clip_epsilon: float = 0.2,          # PPO clip range
        ppo_epochs: int = 10,                # Number of epochs per batch
        entropy_coef: float = 0.01,         # Entropy bonus coefficient
        target_kl: float = 0.02,            # KL divergence threshold for early stopping
        # --- advantage shaping hyper-parameters ---
        adv_clip_range: float = 5.0,
        use_adv_tanh: bool = False,
        objective_keys: Optional[List[str]] = None,
        objective_weights: Optional[Dict[str, float]] = None,
        covariance_whitening_enabled: bool = True,
        covariance_whitening_warmup_steps: int = 100,
        covariance_whitening_min_samples: int = 128,
        covariance_whitening_shrinkage: float = 0.25,
        covariance_whitening_buffer_capacity: int = 4096,
        utility_temperature: float = 0.15,
        utility_baseline_temperature: float = 0.2,
        utility_reference_alpha: float = 0.1,
        utility_reference_update_interval: int = 10,
        utility_reference_quantile_high: float = 0.9,
        utility_reference_quantile_low: float = 0.1,
        utility_reference_clip_delta: float = 0.25,
        utility_archive_capacity: int = 512,
        utility_norm_clip: float = 2.0,
        constraint_base_penalty: float = 1.25,
        constraint_violation_scale: float = 2.0,
        constraint_infeasible_residual: float = 0.05,
    ):
        if use_vae is None:
            use_vae = bool(use_kan) if use_kan is not None else False
        if vae_min_samples is None:
            vae_min_samples = kan_min_samples if kan_min_samples is not None else 1600
        if vae_reward_threshold is None:
            vae_reward_threshold = (
                kan_reward_threshold if kan_reward_threshold is not None else -8.0
            )
        if not run_all_corners and not pvt_outer_loop_enabled:
            use_vae = False

        # Initialize training session directory
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        self._training_saves_dir = os.path.join(
            'training_saves',
            '_'.join(filter(None, [
                f'grpo_{circuit_name}',
                'pvt' if run_all_corners else ('pvtproxy' if pvt_outer_loop_enabled else None),
                'vae' if use_vae else None,
                timestamp
            ]))
        )
        self._logs_dir = os.path.join(self._training_saves_dir, 'logs')
        os.makedirs(self._training_saves_dir, exist_ok=True)
        os.makedirs(self._logs_dir, exist_ok=True)
        self.env = env
        self.policy = policy_network
        self.num_designs_per_circuit = num_designs_per_circuit
        self.use_vae = bool(use_vae)
        self.vae_min_samples = int(vae_min_samples)
        self.vae_reward_threshold = float(vae_reward_threshold)
        self.vae_diagnostics_enabled = bool(vae_diagnostics_enabled)
        self.vae_save_per_objective_plots = bool(vae_save_per_objective_plots)
        self.objective_keys = list(objective_keys) if objective_keys is not None else list(RL_OBJECTIVE_KEYS)
        self.objective_weights = {
            key: float((objective_weights or {}).get(key, 1.0))
            for key in self.objective_keys
        }
        self.covariance_whitening_enabled = bool(covariance_whitening_enabled) and len(self.objective_keys) > 1
        self.covariance_whitening_warmup_steps = max(0, int(covariance_whitening_warmup_steps))
        self.covariance_whitening_min_samples = max(
            len(self.objective_keys) + 1,
            int(covariance_whitening_min_samples),
        )
        self.covariance_whitening_shrinkage = float(np.clip(covariance_whitening_shrinkage, 0.0, 1.0))
        self.covariance_whitening_buffer_capacity = max(
            self.covariance_whitening_min_samples,
            int(covariance_whitening_buffer_capacity),
        )
        self.vae_objective_keys = tuple(VAE_OBJECTIVE_KEYS)
        self.utility_temperature = max(1e-4, float(utility_temperature))
        self.utility_baseline_temperature = max(1e-4, float(utility_baseline_temperature))
        self.utility_reference_alpha = float(np.clip(utility_reference_alpha, 0.0, 1.0))
        self.utility_reference_update_interval = max(1, int(utility_reference_update_interval))
        self.utility_reference_quantile_high = float(np.clip(utility_reference_quantile_high, 0.5, 1.0))
        self.utility_reference_quantile_low = float(np.clip(utility_reference_quantile_low, 0.0, 0.5))
        if self.utility_reference_quantile_low > self.utility_reference_quantile_high:
            self.utility_reference_quantile_low = self.utility_reference_quantile_high
        self.utility_reference_clip_delta = max(1e-6, float(utility_reference_clip_delta))
        self.utility_archive_capacity = max(8, int(utility_archive_capacity))
        self.utility_norm_clip = max(0.25, float(utility_norm_clip))
        self.constraint_base_penalty = float(constraint_base_penalty)
        self.constraint_violation_scale = float(constraint_violation_scale)
        self.constraint_infeasible_residual = max(0.0, float(constraint_infeasible_residual))
        
        # Flag to track if templates have been copied (only need to copy once per agent instance)
        self._templates_copied = False

        # Hyperparameters
        self.lr = learning_rate
        self.max_grad_norm = max_grad_norm
        self.gamma = gamma

        # Reward adapter disabled (raw env reward is used)
        self.reward_adapter = None
        self.use_reward_transformation = False
        self.run_all_corners = bool(run_all_corners)
        self.pvt_outer_loop_enabled = bool(pvt_outer_loop_enabled) and not self.run_all_corners
        self.pvt_num_processes = pvt_num_processes
        self.pvt_verify_topk_per_step = max(0, int(pvt_verify_topk_per_step))
        self.pvt_verify_uncertain_per_step = max(0, int(pvt_verify_uncertain_per_step))
        self.pvt_proxy_beta = max(0.0, float(pvt_proxy_beta))
        self.pvt_proxy_prediction_samples = max(1, int(pvt_proxy_prediction_samples))
        self.pvt_proxy_online_update_interval = max(1, int(pvt_proxy_online_update_interval))
        self.pvt_proxy_online_updates_per_step = max(1, int(pvt_proxy_online_updates_per_step))
        self.pvt_proxy_online_batch_size = max(1, int(pvt_proxy_online_batch_size))
        self.pvt_verified_archive_capacity = max(8, int(pvt_verified_archive_capacity))
        self.pvt_warmup_steps = max(0, int(pvt_warmup_steps))
        self.pvt_warmup_verify_topk_per_step = max(0, int(pvt_warmup_verify_topk_per_step))
        self.pvt_warmup_diverse_per_step = max(0, int(pvt_warmup_diverse_per_step))
        self.pvt_offline_epochs = max(1, int(pvt_offline_epochs))
        self.pvt_offline_patience = max(1, int(pvt_offline_patience))
        self.pvt_proxy_finetune_interval = max(1, int(pvt_proxy_finetune_interval))
        self.pvt_proxy_finetune_epochs = max(1, int(pvt_proxy_finetune_epochs))
        self.pvt_proxy_finetune_patience = max(1, int(pvt_proxy_finetune_patience))

        # Device & policy
        self.device = torch.device(device)
        self.policy.to(self.device)

        # Optimizer
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)

        # Stats
        self.total_steps = 0
        self.episode_count = 0

        # --------------------
        # Global best record (reward is maximized; closer to 0 is better)
        # --------------------
        self.best_reward: float = -float("inf")
        self.best_record: Optional[Dict] = None
        self.best_step: Optional[int] = None
        self.best_performance_reward: float = -float("inf")
        self.best_performance_record: Optional[Dict] = None
        self.best_performance_step: Optional[int] = None
        self._best_rank_tuple: Optional[Tuple[float, ...]] = None
        self.best_pareto_records: List[Dict[str, Any]] = []
        self.verified_pvt_pareto_records: List[Dict[str, Any]] = []
        self.best_objective_records: Dict[str, Dict[str, Any]] = {}
        self._best_objective_values: Dict[str, float] = {}
        self.best_record_archive_capacity: int = 32
        self._latest_best_update_summary: Dict[str, Any] = {
            "global_rank_updated": False,
            "global_rank_step": None,
            "objective_updates": [],
            "pareto_archive_updated": False,
            "pareto_archive_size": 0,
        }
        self.objective_history: Dict[str, List[float]] = {key: [] for key in self.objective_keys}
        self.objective_plot_history: Dict[str, List[float]] = {
            "constraint_reward": [],
            "FOML": [],
            "FOMS": [],
            "Active Area": [],
        }
        self.pm_violation_history: List[float] = []
        self.pm_feasible_rate_history: List[float] = []
        self._utility_feasible_archive: List[np.ndarray] = []
        self._utility_ideal_ema: Optional[np.ndarray] = None
        self._utility_nadir_ema: Optional[np.ndarray] = None
        self._covariance_adv_buffer: List[np.ndarray] = []
        self._covariance_whitener_ready: bool = False
        self._covariance_freeze_step: Optional[int] = None
        self._covariance_aggregation_vector: Optional[np.ndarray] = None
        self._covariance_matrix: Optional[np.ndarray] = None
        self._covariance_whitening_transform: Optional[np.ndarray] = None
        self._covariance_condition_number: Optional[float] = None
        self._last_pvt_verified_count: int = 0
        self._last_pvt_proxy_count: int = 0
        self._last_pvt_proxy_ready: bool = False
        self._last_pvt_dataset_size: int = 0
        self.pvt_verified_count_history: List[int] = []
        self.pvt_verified_archive_size_history: List[int] = []
        self.pvt_phase_history: List[str] = []
        self.verified_pvt_reward_history: List[float] = []
        self.verified_pvt_pm_violation_history: List[float] = []
        self.verified_pvt_pm_feasible_rate_history: List[float] = []
        self.verified_pvt_objective_history: Dict[str, List[float]] = {key: [] for key in self.objective_keys}
        self.verified_pvt_objective_plot_history: Dict[str, List[float]] = {
            "constraint_reward": [],
            "FOML": [],
            "FOMS": [],
            "Active Area": [],
        }

        # Histories
        self.reward_history: List[float] = []
        self.loss_history: List[float] = []
        self.success_rate_history: List[float] = []
        self.per_step_design_rewards: List[List[float]] = []
        self.per_step_extra_corner_rewards: List[List[float]] = []
        self.per_step_verified_pvt_rewards: List[List[float]] = []
        self.advantage_min_history: List[float] = []
        self.advantage_max_history: List[float] = []
        self.covariance_condition_history: List[float] = []
        self.vae_total_history: List[float] = []
        self.vae_recon_history: List[float] = []
        self.vae_pred_worst_history: List[float] = []
        self.vae_actual_worst_history: List[float] = []
        self._vae_pred_records: List[Tuple[int, float, float, float]] = []
        self._vae_multi_pred_records: List[Dict[str, object]] = []
        self._last_verified_pvt_episodes: List[Episode] = []
        self._last_pvt_finetune_step: int = 0
        self._last_pvt_finetune_dataset_size: int = 0
        self._pvt_proxy_ready_step: Optional[int] = None
        self._adaptive_pvt_schedule_ready: bool = False
        self._adaptive_pvt_schedule_summary: Dict[str, Any] = {}
        self._base_vae_min_samples = int(self.vae_min_samples)
        self._base_pvt_warmup_steps = int(self.pvt_warmup_steps)
        self._base_pvt_warmup_verify_topk_per_step = int(self.pvt_warmup_verify_topk_per_step)
        self._base_pvt_warmup_diverse_per_step = int(self.pvt_warmup_diverse_per_step)
        self._base_pvt_verify_topk_per_step = int(self.pvt_verify_topk_per_step)
        self._base_pvt_verify_uncertain_per_step = int(self.pvt_verify_uncertain_per_step)
        self._base_pvt_proxy_prediction_samples = int(self.pvt_proxy_prediction_samples)
        self._base_pvt_verified_archive_capacity = int(self.pvt_verified_archive_capacity)
        self._base_pvt_proxy_finetune_interval = int(self.pvt_proxy_finetune_interval)
        self._base_pvt_offline_epochs = int(self.pvt_offline_epochs)
        self._base_pvt_offline_patience = int(self.pvt_offline_patience)
        self._base_pvt_proxy_finetune_epochs = int(self.pvt_proxy_finetune_epochs)
        self._base_pvt_proxy_finetune_patience = int(self.pvt_proxy_finetune_patience)
        self._pvt_target_new_samples_per_finetune: int = 0

        # Group-level statistics (per circuit_spec)
        self.group_mean_reward_history: List[Dict[str, float]] = []
        self.group_weight_history: List[Dict[str, float]] = []
        self.group_mean_objective_history: List[Dict[str, Dict[str, float]]] = []
        self._last_group_mean_rewards: Dict[str, float] = {}
        self._last_group_weights: Dict[str, float] = {}
        self._last_group_objective_means: Dict[str, Dict[str, float]] = {}

        # Initial state cache
        self.initial_reset_state = None if initial_reset_state is None else np.array(initial_reset_state, copy=True)
        self.initial_reset_info = None if initial_reset_info is None else copy.deepcopy(initial_reset_info)

        if self.initial_reset_state is not None:
            self._cached_initial_state = np.array(self.initial_reset_state, copy=True)
            self._cache_is_valid = True
        else:
            self._cached_initial_state = None
            self._cache_is_valid = False

        # WandB & logging helpers
        self.wandb = wandb_logger
        self.corner_selection_counter: Dict[str, int] = {}
        self._details_buffer: List[str] = []
        self._details_buffer_start_step: Optional[int] = None
        self._extra_details_buffer: List[str] = []
        self._extra_details_buffer_start_step: Optional[int] = None

        # Worst-corner VAE predictor
        if self.use_vae:
            action_dim = env.action_dim
            perf_keys_count = len(VAE_CONDITION_PERFORMANCE_KEYS)
            
            objective_dim = len(self.vae_objective_keys)
            vae_input_dim = action_dim + perf_keys_count + objective_dim
            
            self.vae = WorstCornerVAE(
                input_dim=vae_input_dim,
                output_dim=objective_dim,
                width=[vae_input_dim, 10, objective_dim],
                grid=3,
                k=5,
                seed=42,
                learning_rate=1e-3,
                device="cpu",
                recon_reward_weight=1.0,
            )
            self.vae.objective_keys = tuple(self.vae_objective_keys)
            self.vae_dataset: List[VAETrainingData] = []
            self.vae_loss_history: List[float] = []
            self._vae_training_active: bool = True
            self._vae_offline_trained: bool = False
            self._vae_offline_epochs: int = self.pvt_offline_epochs if self.pvt_outer_loop_enabled else 50
            print("[OK] VAE enabled for worst-corner prediction")
        else:
            self.vae = None
            self.vae_dataset = None
            self.vae_loss_history: List[float] = []
            self._vae_training_active: bool = False
            self._vae_offline_trained: bool = False
            self._vae_offline_epochs: int = 0
            print("[OK] VAE disabled")

        # Backward-compatible KAN aliases.
        self.use_kan = self.use_vae
        self.kan_min_samples = self.vae_min_samples
        self.kan_reward_threshold = self.vae_reward_threshold
        self.kan = self.vae
        self.kan_dataset = self.vae_dataset
        self.kan_loss_history = self.vae_loss_history
        self.kan_total_history = self.vae_total_history
        self.kan_recon_history = self.vae_recon_history
        self.kan_pred_worst_history = self.vae_pred_worst_history
        self.kan_actual_worst_history = self.vae_actual_worst_history
        self._kan_pred_records = self._vae_pred_records
        self._kan_multi_pred_records = self._vae_multi_pred_records
        self._kan_training_active = self._vae_training_active
        self._kan_offline_trained = self._vae_offline_trained
        self._kan_offline_epochs = self._vae_offline_epochs

        # Advantage shaping hyper-parameters
        self.adv_clip_range = float(adv_clip_range) if adv_clip_range is not None else None
        self.use_adv_tanh = bool(use_adv_tanh)
        self.advantage_mode = (
            "covariance_whitened_aggregation"
            if self.covariance_whitening_enabled
            else "objective_decoupled_normalization"
        )
        self.objective_normalization_mode = "per_objective_mean_std"

        # PPO/GRPO hyper-parameters
        self.clip_epsilon = float(clip_epsilon)
        self.ppo_epochs = int(ppo_epochs)
        self.entropy_coef = float(entropy_coef)
        self.target_kl = float(target_kl)

        # PPO tracking histories
        self.kl_history: List[float] = []
        self.clip_fraction_history: List[float] = []
        self.entropy_history: List[float] = []

    # --------------------
    # Serialization helpers
    # --------------------
    def __getstate__(self):
        """Remove heavy runtime-only fields when pickling."""
        state = self.__dict__.copy()
        state["env"] = None
        state["wandb"] = None
        return state

    def to_lightweight(self) -> Dict[str, object]:
        """Return a lightweight dict for persistence (preferred)."""
        result = {
            "policy_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "hyperparams": {
                "num_designs_per_circuit": self.num_designs_per_circuit,
                "lr": self.lr,
                "max_grad_norm": self.max_grad_norm,
                "run_all_corners": self.run_all_corners,
                "pvt_num_processes": self.pvt_num_processes,
                "use_vae": self.use_vae,
                "vae_diagnostics_enabled": self.vae_diagnostics_enabled,
                "vae_save_per_objective_plots": self.vae_save_per_objective_plots,
                "use_kan": self.use_kan,
                "adv_clip_range": self.adv_clip_range,
                "use_adv_tanh": self.use_adv_tanh,
                "advantage_mode": self.advantage_mode,
                "objective_normalization_mode": self.objective_normalization_mode,
                # PPO/GRPO hyperparameters
                "clip_epsilon": self.clip_epsilon,
                "ppo_epochs": self.ppo_epochs,
            "entropy_coef": self.entropy_coef,
            "target_kl": self.target_kl,
            "covariance_whitening_enabled": self.covariance_whitening_enabled,
            "covariance_whitening_warmup_steps": self.covariance_whitening_warmup_steps,
            "covariance_whitening_min_samples": self.covariance_whitening_min_samples,
            "covariance_whitening_shrinkage": self.covariance_whitening_shrinkage,
            "covariance_whitening_buffer_capacity": self.covariance_whitening_buffer_capacity,
            "objective_keys": self.objective_keys,
            "objective_weights": self.objective_weights,
                "covariance_whitening_enabled": self.covariance_whitening_enabled,
                "covariance_whitening_warmup_steps": self.covariance_whitening_warmup_steps,
                "covariance_whitening_min_samples": self.covariance_whitening_min_samples,
                "covariance_whitening_shrinkage": self.covariance_whitening_shrinkage,
                "covariance_whitening_buffer_capacity": self.covariance_whitening_buffer_capacity,
                "vae_objective_keys": list(self.vae_objective_keys),
                "pvt_warmup_steps": self.pvt_warmup_steps,
                "pvt_warmup_verify_topk_per_step": self.pvt_warmup_verify_topk_per_step,
                "pvt_warmup_diverse_per_step": self.pvt_warmup_diverse_per_step,
                "pvt_offline_epochs": self.pvt_offline_epochs,
                "pvt_offline_patience": self.pvt_offline_patience,
                "pvt_proxy_finetune_interval": self.pvt_proxy_finetune_interval,
                "pvt_proxy_finetune_epochs": self.pvt_proxy_finetune_epochs,
                "pvt_proxy_finetune_patience": self.pvt_proxy_finetune_patience,
                "adaptive_pvt_schedule_summary": dict(getattr(self, "_adaptive_pvt_schedule_summary", {})),
                "utility_temperature": self.utility_temperature,
                "utility_baseline_temperature": self.utility_baseline_temperature,
                "utility_reference_alpha": self.utility_reference_alpha,
                "utility_reference_update_interval": self.utility_reference_update_interval,
                "utility_reference_quantile_high": self.utility_reference_quantile_high,
                "utility_reference_quantile_low": self.utility_reference_quantile_low,
                "utility_reference_clip_delta": self.utility_reference_clip_delta,
                "utility_archive_capacity": self.utility_archive_capacity,
                "utility_norm_clip": self.utility_norm_clip,
                "constraint_base_penalty": self.constraint_base_penalty,
                "constraint_violation_scale": self.constraint_violation_scale,
                "constraint_infeasible_residual": self.constraint_infeasible_residual,
            },
            "histories": {
                "reward_history": self.reward_history,
                "loss_history": self.loss_history,
                "success_rate_history": self.success_rate_history,
                "raw_reward_worst_history": getattr(self, "raw_reward_worst_history", []),
                "group_reward_min_history": getattr(self, "group_reward_min_history", []),
                "group_reward_max_history": getattr(self, "group_reward_max_history", []),
                "grad_norm_history": getattr(self, "grad_norm_history", []),
                "vae_loss_history": getattr(self, "vae_loss_history", []),
                "kan_loss_history": getattr(self, "kan_loss_history", []),
                # PPO/GRPO tracking
                "kl_history": getattr(self, "kl_history", []),
                "clip_fraction_history": getattr(self, "clip_fraction_history", []),
                "entropy_history": getattr(self, "entropy_history", []),
                "objective_history": getattr(self, "objective_history", {}),
                "objective_plot_history": getattr(self, "objective_plot_history", {}),
                "verified_pvt_reward_history": getattr(self, "verified_pvt_reward_history", []),
                "verified_pvt_pm_violation_history": getattr(self, "verified_pvt_pm_violation_history", []),
                "verified_pvt_pm_feasible_rate_history": getattr(self, "verified_pvt_pm_feasible_rate_history", []),
                "verified_pvt_objective_history": getattr(self, "verified_pvt_objective_history", {}),
                "verified_pvt_objective_plot_history": getattr(self, "verified_pvt_objective_plot_history", {}),
                "pvt_phase_history": getattr(self, "pvt_phase_history", []),
                "pm_violation_history": getattr(self, "pm_violation_history", []),
                "pm_feasible_rate_history": getattr(self, "pm_feasible_rate_history", []),
                "covariance_condition_history": getattr(self, "covariance_condition_history", []),
            },
            "corner_selection_counter": self.corner_selection_counter,
            "total_steps": self.total_steps,
            "best_reward": self.best_reward,
            "best_step": self.best_step,
            "best_record": self.best_record,
            "best_performance_reward": self.best_performance_reward,
            "best_performance_step": self.best_performance_step,
            "best_performance_record": self.best_performance_record,
            "best_pareto_records": self.best_pareto_records,
            "best_objective_records": self.best_objective_records,
            "covariance_whitening_state": {
                "ready": self._covariance_whitener_ready,
                "freeze_step": self._covariance_freeze_step,
                "aggregation_vector": None
                if self._covariance_aggregation_vector is None
                else self._covariance_aggregation_vector.tolist(),
                "condition_number": self._covariance_condition_number,
            },
        }
        if self.use_vae and self.vae is not None:
            result["vae_state_dict"] = self.vae.state_dict()
            result["vae_optimizer_state_dict"] = self.vae.optimizer.state_dict()
            result["kan_state_dict"] = self.vae.state_dict()
            result["kan_optimizer_state_dict"] = self.vae.optimizer.state_dict()
        return result

    # --------------------
    # JSON helpers (for saving best record)
    # --------------------
    def _json_safe(self, obj):
        """Recursively convert obj into JSON-serializable types."""
        if obj is None:
            return None
        if isinstance(obj, (str, int, float, bool)):
            if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
                return None
            return obj
        if isinstance(obj, (np.floating, np.integer)):
            v = obj.item()
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                return None
            return v
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy().tolist()
        if isinstance(obj, dict):
            return {str(k): self._json_safe(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self._json_safe(x) for x in obj]
        try:
            json.dumps(obj)
            return obj
        except Exception:
            return str(obj)

    def _utility_weight_vector(self) -> np.ndarray:
        weights = np.asarray(
            [max(1e-8, float(self.objective_weights.get(key, 1.0))) for key in self.objective_keys],
            dtype=np.float32,
        )
        total = float(weights.sum())
        if not np.isfinite(total) or total <= 0.0:
            return np.full(len(self.objective_keys), 1.0 / max(1, len(self.objective_keys)), dtype=np.float32)
        return weights / total

    def _append_covariance_advantage_buffer(self, vectors: List[np.ndarray]) -> None:
        if not self.covariance_whitening_enabled or self._covariance_whitener_ready:
            return
        for vector in vectors:
            arr = np.asarray(vector, dtype=np.float64).reshape(-1)
            if arr.shape[0] != len(self.objective_keys) or not np.all(np.isfinite(arr)):
                continue
            self._covariance_adv_buffer.append(arr)
        overflow = len(self._covariance_adv_buffer) - self.covariance_whitening_buffer_capacity
        if overflow > 0:
            del self._covariance_adv_buffer[:overflow]

    def _fit_covariance_whitener(self) -> bool:
        if not self.covariance_whitening_enabled:
            return False
        if len(self._covariance_adv_buffer) < self.covariance_whitening_min_samples:
            return False
        values = np.stack(self._covariance_adv_buffer, axis=0)
        if values.ndim != 2 or values.shape[0] < 2:
            return False
        covariance = np.cov(values, rowvar=False, bias=False)
        covariance = np.asarray(covariance, dtype=np.float64)
        if covariance.ndim == 0:
            covariance = covariance.reshape(1, 1)
        covariance = np.nan_to_num(covariance, nan=0.0, posinf=0.0, neginf=0.0)
        covariance = 0.5 * (covariance + covariance.T)
        identity = np.eye(covariance.shape[0], dtype=np.float64)
        regularized = (
            (1.0 - self.covariance_whitening_shrinkage) * covariance
            + self.covariance_whitening_shrinkage * identity
        )
        eigenvalues, eigenvectors = np.linalg.eigh(regularized)
        eigenvalues = np.clip(eigenvalues, 1e-6, None)
        inv_eigenvalues = 1.0 / eigenvalues
        inv_sqrt_eigenvalues = 1.0 / np.sqrt(eigenvalues)
        covariance_inv = (eigenvectors * inv_eigenvalues[np.newaxis, :]) @ eigenvectors.T
        whitening_transform = (eigenvectors * inv_sqrt_eigenvalues[np.newaxis, :]) @ eigenvectors.T
        weight_vector = self._utility_weight_vector().astype(np.float64)
        denominator_sq = float(weight_vector @ covariance_inv @ weight_vector)
        if not np.isfinite(denominator_sq) or denominator_sq <= 1e-8:
            return False
        self._covariance_aggregation_vector = (covariance_inv @ weight_vector) / math.sqrt(denominator_sq)
        self._covariance_matrix = regularized
        self._covariance_whitening_transform = whitening_transform
        self._covariance_condition_number = float(np.max(eigenvalues) / np.min(eigenvalues))
        return True

    def _maybe_freeze_covariance_whitener(self, current_step: int) -> None:
        if not self.covariance_whitening_enabled or self._covariance_whitener_ready:
            return
        if current_step < self.covariance_whitening_warmup_steps:
            return
        if not self._fit_covariance_whitener():
            return
        self._covariance_whitener_ready = True
        self._covariance_freeze_step = int(current_step)
        self._covariance_adv_buffer = []

    def _aggregate_objective_advantages(self, objective_vector: np.ndarray) -> Tuple[float, str]:
        vec = np.asarray(objective_vector, dtype=np.float64).reshape(-1)
        weight_vector = self._utility_weight_vector().astype(np.float64)
        legacy_value = float(weight_vector @ vec)
        if not self.covariance_whitening_enabled or not self._covariance_whitener_ready:
            return legacy_value, "legacy"
        if self._covariance_aggregation_vector is None:
            return legacy_value, "legacy"
        aggregated = float(self._covariance_aggregation_vector @ vec)
        if not np.isfinite(aggregated):
            return legacy_value, "legacy"
        return aggregated, "covariance_whitened"

    @staticmethod
    def _dominates_vector(lhs: np.ndarray, rhs: np.ndarray, eps: float = 1e-8) -> bool:
        if lhs.shape != rhs.shape:
            return False
        return bool(np.all(lhs >= rhs - eps) and np.any(lhs > rhs + eps))

    @staticmethod
    def _crowding_distance(values: np.ndarray) -> np.ndarray:
        if values.ndim != 2 or values.shape[0] == 0:
            return np.zeros(0, dtype=np.float32)
        num_points, num_dims = values.shape
        if num_points <= 2:
            return np.full(num_points, np.inf, dtype=np.float32)
        distances = np.zeros(num_points, dtype=np.float32)
        for dim in range(num_dims):
            order = np.argsort(values[:, dim])
            sorted_vals = values[order, dim]
            span = float(sorted_vals[-1] - sorted_vals[0])
            distances[order[0]] = np.inf
            distances[order[-1]] = np.inf
            if span <= 1e-12:
                continue
            for pos in range(1, num_points - 1):
                if np.isinf(distances[order[pos]]):
                    continue
                distances[order[pos]] += float(sorted_vals[pos + 1] - sorted_vals[pos - 1]) / span
        return distances

    def _objective_vector(self, objectives: Dict[str, float]) -> np.ndarray:
        values = np.asarray(
            [float(objectives.get(key, -1e6)) for key in self.objective_keys],
            dtype=np.float64,
        )
        return np.where(np.isfinite(values), values, -1e6)

    def _register_feasible_objective_vector(self, vector: np.ndarray) -> None:
        vec = np.asarray(vector, dtype=np.float64).reshape(-1)
        if vec.shape[0] != len(self.objective_keys) or not np.all(np.isfinite(vec)):
            return
        keep_archive: List[np.ndarray] = []
        dominated = False
        for existing in self._utility_feasible_archive:
            existing_vec = np.asarray(existing, dtype=np.float64).reshape(-1)
            if self._dominates_vector(existing_vec, vec):
                dominated = True
                keep_archive.append(existing_vec)
                continue
            if self._dominates_vector(vec, existing_vec):
                continue
            keep_archive.append(existing_vec)
        if not dominated:
            keep_archive.append(vec)
        self._utility_feasible_archive = keep_archive
        if len(self._utility_feasible_archive) > self.utility_archive_capacity:
            archive_values = np.stack(self._utility_feasible_archive, axis=0)
            crowding = self._crowding_distance(archive_values)
            remove_idx = int(np.argmin(crowding))
            del self._utility_feasible_archive[remove_idx]

    def _reference_point_estimates(self, values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        arr = np.asarray(values, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[0] == 0:
            zeros = np.zeros(len(self.objective_keys), dtype=np.float64)
            return zeros.copy(), zeros.copy()
        ideal = np.max(arr, axis=0)
        nadir = np.min(arr, axis=0)
        ideal = np.where(np.isfinite(ideal), ideal, 0.0)
        nadir = np.where(np.isfinite(nadir), nadir, 0.0)
        ideal = np.maximum(ideal, nadir + 1e-6)
        return ideal.astype(np.float64), nadir.astype(np.float64)

    def _lagged_reference_update(self, current: np.ndarray, target: np.ndarray) -> np.ndarray:
        current_arr = np.asarray(current, dtype=np.float64)
        target_arr = np.asarray(target, dtype=np.float64)
        updated = (1.0 - self.utility_reference_alpha) * current_arr + self.utility_reference_alpha * target_arr
        return np.where(np.isfinite(updated), updated, current_arr)

    def _update_utility_reference_points(
        self,
        feasible_vectors: List[np.ndarray],
        step: Optional[int] = None,
    ) -> None:
        if not feasible_vectors:
            return
        for vector in feasible_vectors:
            self._register_feasible_objective_vector(vector)
        if self._utility_feasible_archive:
            archive_values = np.stack(self._utility_feasible_archive, axis=0)
        else:
            archive_values = np.stack(feasible_vectors, axis=0)
        current_ideal, current_nadir = self._reference_point_estimates(archive_values)
        if self._utility_ideal_ema is None or self._utility_nadir_ema is None:
            self._utility_ideal_ema = current_ideal.astype(np.float64)
            self._utility_nadir_ema = current_nadir.astype(np.float64)
            return
        self._utility_ideal_ema = self._lagged_reference_update(self._utility_ideal_ema, current_ideal)
        self._utility_nadir_ema = self._lagged_reference_update(self._utility_nadir_ema, current_nadir)
        self._utility_ideal_ema = np.maximum(self._utility_ideal_ema, self._utility_nadir_ema + 1e-6)

    def _get_utility_reference_points(
        self,
        fallback_vectors: Optional[List[np.ndarray]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self._utility_ideal_ema is not None and self._utility_nadir_ema is not None:
            return self._utility_ideal_ema.copy(), self._utility_nadir_ema.copy()
        if self._utility_feasible_archive:
            archive_values = np.stack(self._utility_feasible_archive, axis=0)
            return self._reference_point_estimates(archive_values)
        if fallback_vectors:
            values = np.stack([np.asarray(vec, dtype=np.float64).reshape(-1) for vec in fallback_vectors], axis=0)
            return self._reference_point_estimates(values)
        zeros = np.zeros(len(self.objective_keys), dtype=np.float64)
        return zeros.copy(), zeros.copy()

    def _normalize_objective_vector(
        self,
        vector: np.ndarray,
        fallback_vectors: Optional[List[np.ndarray]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        ideal, nadir = self._get_utility_reference_points(fallback_vectors=fallback_vectors)
        span = np.maximum(ideal - nadir, 1e-6)
        normalized = (vector - nadir) / span
        normalized = np.clip(normalized, -self.utility_norm_clip, 1.0)
        return normalized, ideal, nadir

    def _smooth_tchebycheff(
        self,
        objectives: Dict[str, float],
        fallback_vectors: Optional[List[np.ndarray]] = None,
    ) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        if not self.objective_keys:
            empty = np.zeros(0, dtype=np.float64)
            return 0.0, empty, empty, empty
        values = self._objective_vector(objectives)
        normalized, ideal, nadir = self._normalize_objective_vector(values, fallback_vectors=fallback_vectors)
        weights = self._utility_weight_vector().astype(np.float64)
        distances = weights * (1.0 - normalized)
        scaled = distances / self.utility_temperature
        max_scaled = float(np.max(scaled))
        scaled -= max_scaled
        soft_distance = self.utility_temperature * (
            math.log(float(np.sum(np.exp(scaled)))) + max_scaled
        )
        utility = self.utility_temperature * math.log(max(1, len(self.objective_keys))) - soft_distance
        return float(utility), normalized, ideal, nadir

    def _constrained_utility(self, utility: float, pm_violation: float) -> float:
        if pm_violation <= 1e-8:
            return float(utility)
        residual = self.constraint_infeasible_residual * float(utility)
        return float(-(self.constraint_base_penalty + self.constraint_violation_scale * pm_violation) + residual)

    def _utility_baseline(self, utilities: np.ndarray) -> float:
        finite = np.asarray(utilities, dtype=np.float64)
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            return 0.0
        if finite.size == 1:
            return float(finite[0])
        shifted = (finite - float(np.max(finite))) / self.utility_baseline_temperature
        weights = np.exp(shifted)
        weight_sum = float(weights.sum())
        if not np.isfinite(weight_sum) or weight_sum <= 1e-12:
            return float(np.mean(finite))
        weights /= weight_sum
        return float(np.sum(weights * finite))

    @staticmethod
    def _mad_scale(values: np.ndarray) -> float:
        arr = np.asarray(values, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if arr.size <= 1:
            return 1.0
        median = float(np.median(arr))
        mad = float(np.median(np.abs(arr - median)))
        return max(1e-6, 1.4826 * mad)

    @staticmethod
    def _std_scale(values: np.ndarray) -> float:
        arr = np.asarray(values, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if arr.size <= 1:
            return 1.0
        std = float(np.std(arr))
        return max(1e-6, std)

    def _decoupled_normalize_group_values(
        self,
        values: np.ndarray,
    ) -> Tuple[np.ndarray, float, float]:
        arr = np.asarray(values, dtype=np.float64)
        finite_mask = np.isfinite(arr)
        if not finite_mask.any():
            return np.zeros_like(arr, dtype=np.float64), 0.0, 1.0
        finite_values = arr[finite_mask]
        baseline = float(np.mean(finite_values))
        scale = self._std_scale(finite_values)
        normalized = np.zeros_like(arr, dtype=np.float64)
        normalized[finite_mask] = (arr[finite_mask] - baseline) / (scale + 1e-8)
        normalized = np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)
        return normalized, baseline, scale

    @staticmethod
    def _corner_metadata(corner_idx: int, corner_fallback: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        base = {"index": int(corner_idx)}
        if isinstance(corner_fallback, dict):
            for key in ("proc", "vdd", "temp", "folder"):
                if key in corner_fallback:
                    base[key] = copy.deepcopy(corner_fallback.get(key))
        return base

    def _extract_objective_rewards(self, perf_info: Optional[Dict[str, Any]]) -> Dict[str, float]:
        perf = perf_info or {}
        raw_objectives = perf.get("objective_rewards", {}) or {}
        target_objectives = build_target_objectives(
            objectives=raw_objectives,
            performance=perf,
            objective_keys=self.vae_objective_keys,
        )
        return {
            key: float(target_objectives.get(key, 0.0))
            for key in self.vae_objective_keys
        }

    def _summarize_worst_corner(self, perf_info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not isinstance(perf_info, dict):
            return None
        pvt_context = perf_info.get("pvt_context")
        per_corner = (pvt_context or {}).get("per_corner", {}) if isinstance(pvt_context, dict) else {}
        if isinstance(per_corner, dict) and per_corner:
            items = sorted(per_corner.items(), key=lambda kv: int(kv[0]))
        else:
            return None

        condition_map = {}
        selected_corner_meta = perf_info.get("pvt_selected_corner")
        if isinstance(selected_corner_meta, dict):
            condition_map[int(selected_corner_meta.get("index", 0))] = selected_corner_meta

        local_vectors: List[np.ndarray] = []
        for raw_idx, corner_info in items:
            if not isinstance(corner_info, dict):
                continue
            target_objectives = self._extract_objective_rewards(corner_info)
            local_vectors.append(
                self._objective_vector({key: float(target_objectives.get(key, 0.0)) for key in self.objective_keys})
            )

        corner_records: List[Dict[str, Any]] = []
        for raw_idx, corner_info in items:
            if not isinstance(corner_info, dict):
                continue
            corner_idx = int(raw_idx)
            target_objectives = self._extract_objective_rewards(corner_info)
            utility_objectives = {
                key: float(target_objectives.get(key, 0.0))
                for key in self.objective_keys
            }
            pm_violation = float(
                target_objectives.get(
                    "PM_violation",
                    compute_phase_margin_violation(corner_info.get("phase_margin (deg)", np.nan)),
                )
            )
            utility, normalized, ideal, nadir = self._smooth_tchebycheff(
                utility_objectives,
                fallback_vectors=local_vectors,
            )
            constrained_utility = self._constrained_utility(utility, pm_violation)
            corner_records.append(
                {
                    "index": corner_idx,
                    "metadata": self._corner_metadata(corner_idx, condition_map.get(corner_idx)),
                    "info": copy.deepcopy(corner_info),
                    "target_objectives": target_objectives,
                    "utility_objectives": utility_objectives,
                    "normalized_objectives": normalized,
                    "ideal_point": ideal,
                    "nadir_point": nadir,
                    "pm_violation": pm_violation,
                    "pm_feasible": bool(pm_violation <= 1e-8),
                    "utility": float(utility),
                    "constrained_utility": float(constrained_utility),
                }
            )

        if not corner_records:
            return None

        infeasible = [record for record in corner_records if not record["pm_feasible"]]
        if infeasible:
            selected = max(
                infeasible,
                key=lambda record: (float(record["pm_violation"]), -float(record["utility"])),
            )
        else:
            selected = min(corner_records, key=lambda record: float(record["utility"]))

        selected_info = copy.deepcopy(selected["info"])
        selected_info["objective_rewards"] = dict(selected["target_objectives"])
        selected_info["pm_violation"] = float(selected["pm_violation"])
        selected_info["pm_constraint_satisfied"] = bool(selected["pm_feasible"])
        selected_info["utility"] = float(selected["utility"])
        selected_info["constrained_utility"] = float(selected["constrained_utility"])
        selected_info["training_reward"] = float(selected["constrained_utility"])
        selected_info["worst_case_targets"] = copy.deepcopy(selected["target_objectives"])
        selected_info["worst_case_objectives"] = copy.deepcopy(selected["utility_objectives"])
        selected_info["worst_case_normalized_objectives"] = {
            key: float(selected["normalized_objectives"][idx])
            for idx, key in enumerate(self.objective_keys)
        }
        selected_info["utility_reference"] = {
            "ideal": {key: float(selected["ideal_point"][idx]) for idx, key in enumerate(self.objective_keys)},
            "nadir": {key: float(selected["nadir_point"][idx]) for idx, key in enumerate(self.objective_keys)},
        }
        selected_info["legacy_pvt_selected_corner"] = copy.deepcopy(perf_info.get("pvt_selected_corner"))
        selected_info["pvt_selected_corner"] = copy.deepcopy(selected["metadata"])
        for passthrough_key in ("extra_corner", "real_action", "pvt_sim_time_sec", "pvt_context"):
            if passthrough_key in perf_info and passthrough_key not in selected_info:
                selected_info[passthrough_key] = copy.deepcopy(perf_info.get(passthrough_key))
        selected_info["worst_case_selection"] = {
            "mode": "pm_constraint_first_smooth_tchebycheff",
            "selected_corner_idx": int(selected["index"]),
            "selected_metadata": copy.deepcopy(selected["metadata"]),
            "selected_pm_violation": float(selected["pm_violation"]),
            "selected_utility": float(selected["utility"]),
            "selected_constrained_utility": float(selected["constrained_utility"]),
            "selected_targets": copy.deepcopy(selected["target_objectives"]),
        }
        return {
            "reward": float(selected["constrained_utility"]),
            "utility": float(selected["utility"]),
            "pm_violation": float(selected["pm_violation"]),
            "pm_feasible": bool(selected["pm_feasible"]),
            "objective_rewards": copy.deepcopy(selected["utility_objectives"]),
            "vae_targets": copy.deepcopy(selected["target_objectives"]),
            "selected_corner_idx": int(selected["index"]),
            "performance": selected_info,
        }

    def _build_episode_from_perf(
        self,
        circuit_spec: str,
        state: np.ndarray,
        action: np.ndarray,
        perf_info: Any,
        design_idx: int,
        old_log_prob: float,
        verbose: bool = True,
    ) -> Episode:
        source_perf = perf_info if isinstance(perf_info, dict) else {}
        selected_summary = self._summarize_worst_corner(source_perf)
        if selected_summary is not None:
            reward = float(selected_summary["reward"])
            perf_info = selected_summary["performance"]
            objective_rewards = dict(selected_summary["objective_rewards"])
            pm_violation = float(selected_summary["pm_violation"])
            pm_feasible = bool(selected_summary["pm_feasible"])
            utility = float(selected_summary["utility"])
            selected_corner_idx = int(selected_summary["selected_corner_idx"])
            evaluation_source = "full_pvt"
        else:
            perf_info = copy.deepcopy(source_perf)
            target_objectives = self._extract_objective_rewards(perf_info)
            objective_rewards = {
                key: float(target_objectives.get(key, 0.0))
                for key in self.objective_keys
            }
            pm_violation = float(target_objectives.get("PM_violation", compute_phase_margin_violation(perf_info.get("phase_margin (deg)", np.nan))))
            pm_feasible = bool(pm_violation <= 1e-8)
            utility, normalized, ideal, nadir = self._smooth_tchebycheff(objective_rewards)
            reward = float(self._constrained_utility(utility, pm_violation))
            perf_info["objective_rewards"] = dict(target_objectives)
            perf_info["pm_violation"] = float(pm_violation)
            perf_info["pm_constraint_satisfied"] = bool(pm_feasible)
            perf_info["utility"] = float(utility)
            perf_info["constrained_utility"] = float(reward)
            perf_info["training_reward"] = float(reward)
            perf_info["worst_case_targets"] = dict(target_objectives)
            perf_info["worst_case_objectives"] = dict(objective_rewards)
            perf_info["worst_case_normalized_objectives"] = {
                key: float(normalized[idx]) for idx, key in enumerate(self.objective_keys)
            }
            perf_info["utility_reference"] = {
                "ideal": {key: float(ideal[idx]) for idx, key in enumerate(self.objective_keys)},
                "nadir": {key: float(nadir[idx]) for idx, key in enumerate(self.objective_keys)},
            }
            selected_corner_idx = int((perf_info.get("pvt_selected_corner", {}) or {}).get("index", -1))
            evaluation_source = "full_pvt" if self.run_all_corners else "tt"

        ep = Episode(
            circuit_spec=circuit_spec,
            state=state.copy(),
            action=np.asarray(action, dtype=np.float32).copy(),
            reward=float(reward),
            utility=float(utility),
            pm_violation=float(pm_violation),
            pm_feasible=bool(pm_feasible),
            selected_corner_idx=int(selected_corner_idx),
            performance=perf_info if isinstance(perf_info, dict) else {},
            objective_rewards=objective_rewards,
            design_idx=int(design_idx),
            old_log_prob=float(old_log_prob),
            evaluation_source=evaluation_source,
        )

        if verbose:
            tt_r = None
            try:
                tt_r = (ep.performance.get("extra_corner") or {}).get("reward", None)
            except Exception:
                tt_r = None
            print(
                f"  Design {design_idx}/{self.num_designs_per_circuit-1}: "
                f"source={evaluation_source} worst_case_utility={reward:.4f} "
                f"PM_violation={pm_violation:.4f} TT={tt_r}"
            )
        return ep

    def _episode_real_action(self, ep: Episode) -> Optional[np.ndarray]:
        perf = ep.performance or {}
        real_action = perf.get("real_action")
        if real_action is not None:
            try:
                return np.asarray(real_action, dtype=np.float32).reshape(-1)
            except Exception:
                return None
        try:
            from utils import ActionNormalizer

            action = ActionNormalizer(
                self.env.action_space_low,
                self.env.action_space_high,
                self.env.action_space_step,
            ).action(np.array(ep.action, copy=True))
            return np.asarray(action, dtype=np.float32).reshape(-1)
        except Exception:
            return None

    def _build_pvt_proxy_condition(
        self,
        ep: Episode,
    ) -> Optional[Tuple[np.ndarray, Dict[str, float], np.ndarray]]:
        perf = ep.performance or {}
        tt_action = self._episode_real_action(ep)
        if tt_action is None:
            return None
        tt_objectives = build_target_objectives(
            objectives=perf.get("objective_rewards", {}) or ep.objective_rewards or {},
            performance=perf,
            objective_keys=self.vae_objective_keys,
        )
        condition = build_vae_condition(
            tt_action=np.asarray(tt_action, dtype=np.float32),
            tt_performance=perf,
            tt_objectives=tt_objectives,
            objective_keys=self.vae_objective_keys,
        )
        return condition, tt_objectives, np.asarray(tt_action, dtype=np.float32)

    def _conservative_pvt_proxy_targets(
        self,
        tt_targets: Dict[str, float],
        pred_mean: Dict[str, float],
        pred_std: Dict[str, float],
    ) -> Dict[str, float]:
        proxy: Dict[str, float] = {}
        for key in self.objective_keys:
            tt_value = float(tt_targets.get(key, 0.0))
            mean_value = float(pred_mean.get(key, tt_value))
            std_value = abs(float(pred_std.get(key, 0.0)))
            proxy[key] = float(min(tt_value, mean_value - self.pvt_proxy_beta * std_value))

        tt_pm = float(tt_targets.get("PM_violation", 1.0))
        pm_mean = float(pred_mean.get("PM_violation", tt_pm))
        pm_std = abs(float(pred_std.get("PM_violation", 0.0)))
        proxy["PM_violation"] = float(max(tt_pm, pm_mean + self.pvt_proxy_beta * pm_std))
        return proxy

    def _attach_pvt_proxy_predictions(self, episodes: List[Episode]) -> int:
        if not (self.pvt_outer_loop_enabled and self.use_vae and self.vae is not None and self._vae_offline_trained):
            self._last_pvt_proxy_ready = False
            self._last_pvt_proxy_count = 0
            return 0

        predicted = 0
        for ep in episodes:
            if not bool(getattr(ep, "pm_feasible", False)):
                continue
            bundle = self._build_pvt_proxy_condition(ep)
            if bundle is None:
                continue
            condition, tt_targets, _ = bundle
            try:
                pred_mean_dict, pred_std_dict = self.vae.predict_pvt_objectives(
                    condition,
                    num_samples=self.pvt_proxy_prediction_samples,
                )
            except Exception as exc:
                print(f"[PVT proxy] prediction failed: {exc}")
                continue

            proxy_targets = self._conservative_pvt_proxy_targets(tt_targets, pred_mean_dict, pred_std_dict)
            proxy_objectives = {
                key: float(proxy_targets.get(key, 0.0))
                for key in self.objective_keys
            }
            proxy_pm_violation = float(proxy_targets.get("PM_violation", 1.0))
            proxy_utility, _, _, _ = self._smooth_tchebycheff(proxy_objectives)
            proxy_training_reward = float(self._constrained_utility(proxy_utility, proxy_pm_violation))

            perf = ep.performance if isinstance(ep.performance, dict) else {}
            perf["pvt_proxy_ready"] = True
            perf["pvt_proxy_objectives_mean"] = dict(pred_mean_dict)
            perf["pvt_proxy_objectives_std"] = dict(pred_std_dict)
            perf["pvt_proxy_objectives_conservative"] = dict(proxy_targets)
            perf["pvt_proxy_utility"] = float(proxy_utility)
            perf["pvt_proxy_training_reward"] = float(proxy_training_reward)
            perf["pvt_proxy_pm_violation"] = float(proxy_pm_violation)
            perf["pvt_proxy_pm_feasible"] = bool(proxy_pm_violation <= 1e-8)
            ep.performance = perf
            predicted += 1

        self._last_pvt_proxy_ready = predicted > 0
        self._last_pvt_proxy_count = predicted
        return predicted

    def _pvt_proxy_rank_tuple(self, ep: Episode) -> Tuple[float, ...]:
        perf = ep.performance or {}
        proxy_targets = perf.get("pvt_proxy_objectives_conservative", {}) or {}
        proxy_constraint = float(proxy_targets.get("constraint_reward", -1e6))
        proxy_secondary = float(
            proxy_targets.get("FOML_score", 0.0)
            + proxy_targets.get("FOMS_score", 0.0)
            + proxy_targets.get("Active_Area_score", 0.0)
        )
        proxy_reward = float(perf.get("pvt_proxy_training_reward", -1e6))
        proxy_pm_violation = float(perf.get("pvt_proxy_pm_violation", np.inf))
        proxy_pm_feasible = 1.0 if bool(perf.get("pvt_proxy_pm_feasible", False)) else 0.0
        if not np.isfinite(proxy_reward):
            proxy_reward = -1e6
        if not np.isfinite(proxy_constraint):
            proxy_constraint = -1e6
        if not np.isfinite(proxy_secondary):
            proxy_secondary = -1e6
        if not np.isfinite(proxy_pm_violation):
            proxy_pm_violation = np.inf
        return (
            proxy_pm_feasible,
            proxy_reward,
            proxy_constraint,
            proxy_secondary,
            -proxy_pm_violation,
        )

    def _pvt_proxy_uncertainty(self, ep: Episode) -> float:
        perf = ep.performance or {}
        pred_std = perf.get("pvt_proxy_objectives_std", {}) or {}
        values = [
            float(pred_std.get(key, 0.0))
            for key in self.vae_objective_keys
            if np.isfinite(float(pred_std.get(key, 0.0)))
        ]
        if not values:
            return 0.0
        return float(np.linalg.norm(np.asarray(values, dtype=np.float64), ord=2))

    @staticmethod
    def _clamp_int(value: int, lower: int, upper: int) -> int:
        return int(max(lower, min(int(value), upper)))

    def configure_adaptive_pvt_schedule(self, total_train_steps: int) -> Dict[str, Any]:
        if not self.pvt_outer_loop_enabled:
            self._adaptive_pvt_schedule_ready = True
            self._adaptive_pvt_schedule_summary = {"enabled": False}
            return dict(self._adaptive_pvt_schedule_summary)

        total_steps = max(1, int(total_train_steps))
        min_proxy_steps = max(20, total_steps // 5)
        warmup_cap = max(10, total_steps - min_proxy_steps)
        warmup_floor = min(warmup_cap, max(40, total_steps // 4))
        adaptive_warmup = int(round(total_steps * 0.45))
        adaptive_warmup = self._clamp_int(adaptive_warmup, warmup_floor, warmup_cap)
        self.pvt_warmup_steps = int(max(self._base_pvt_warmup_steps, adaptive_warmup))
        self.pvt_warmup_steps = min(self.pvt_warmup_steps, warmup_cap)

        designs_per_step = max(1, int(self.num_designs_per_circuit))
        adaptive_warmup_budget = 1 if designs_per_step <= 4 else 2
        if total_steps >= 400 and designs_per_step >= 8:
            adaptive_warmup_budget = 3
        adaptive_warmup_budget = min(adaptive_warmup_budget, designs_per_step)
        self.pvt_warmup_verify_topk_per_step = max(1, self._base_pvt_warmup_verify_topk_per_step)
        self.pvt_warmup_verify_topk_per_step = min(self.pvt_warmup_verify_topk_per_step, adaptive_warmup_budget)
        self.pvt_warmup_diverse_per_step = max(
            self._base_pvt_warmup_diverse_per_step,
            max(0, adaptive_warmup_budget - self.pvt_warmup_verify_topk_per_step),
        )
        warmup_total_budget = min(
            designs_per_step,
            self.pvt_warmup_verify_topk_per_step + self.pvt_warmup_diverse_per_step,
        )

        adaptive_proxy_budget = 2 if designs_per_step >= 6 else 1
        if total_steps >= 400 and designs_per_step >= 8:
            adaptive_proxy_budget = 3
        adaptive_proxy_budget = min(adaptive_proxy_budget, designs_per_step)
        self.pvt_verify_topk_per_step = max(1, self._base_pvt_verify_topk_per_step)
        self.pvt_verify_topk_per_step = min(self.pvt_verify_topk_per_step, adaptive_proxy_budget)
        self.pvt_verify_uncertain_per_step = max(
            self._base_pvt_verify_uncertain_per_step,
            max(0, adaptive_proxy_budget - self.pvt_verify_topk_per_step),
        )

        expected_warmup_samples = max(1, self.pvt_warmup_steps * warmup_total_budget)
        adaptive_min_samples = int(round(expected_warmup_samples * 0.25))
        adaptive_min_samples = self._clamp_int(
            adaptive_min_samples,
            max(64, self._base_vae_min_samples),
            max(max(64, self._base_vae_min_samples), expected_warmup_samples),
        )
        self.vae_min_samples = int(max(self._base_vae_min_samples, adaptive_min_samples))

        adaptive_proxy_prediction_samples = self._clamp_int(
            int(round(12 + total_steps / 20.0)),
            max(12, self._base_pvt_proxy_prediction_samples),
            64,
        )
        self.pvt_proxy_prediction_samples = int(
            max(self._base_pvt_proxy_prediction_samples, adaptive_proxy_prediction_samples)
        )

        adaptive_archive_capacity = self._clamp_int(
            int(round(max(32, total_steps * 0.3))),
            max(32, self._base_pvt_verified_archive_capacity),
            256,
        )
        self.pvt_verified_archive_capacity = int(
            max(self._base_pvt_verified_archive_capacity, adaptive_archive_capacity)
        )

        remaining_steps = max(20, total_steps - self.pvt_warmup_steps)
        adaptive_finetune_interval = self._clamp_int(int(round(remaining_steps / 8.0)), 8, 25)
        self.pvt_proxy_finetune_interval = int(
            max(self._base_pvt_proxy_finetune_interval, adaptive_finetune_interval)
        )
        self._pvt_target_new_samples_per_finetune = self._clamp_int(
            int(round(self.vae_min_samples * 0.12)),
            4,
            24,
        )

        self._adaptive_pvt_schedule_ready = True
        self._adaptive_pvt_schedule_summary = {
            "enabled": True,
            "total_train_steps": total_steps,
            "warmup_steps": int(self.pvt_warmup_steps),
            "warmup_verify_topk_per_step": int(self.pvt_warmup_verify_topk_per_step),
            "warmup_diverse_per_step": int(self.pvt_warmup_diverse_per_step),
            "proxy_verify_topk_per_step": int(self.pvt_verify_topk_per_step),
            "proxy_verify_uncertain_per_step": int(self.pvt_verify_uncertain_per_step),
            "vae_min_samples": int(self.vae_min_samples),
            "expected_warmup_samples": int(expected_warmup_samples),
            "proxy_prediction_samples": int(self.pvt_proxy_prediction_samples),
            "verified_archive_capacity": int(self.pvt_verified_archive_capacity),
            "proxy_finetune_interval": int(self.pvt_proxy_finetune_interval),
            "target_new_samples_per_finetune": int(self._pvt_target_new_samples_per_finetune),
            "min_proxy_steps_reserved": int(min_proxy_steps),
        }
        return dict(self._adaptive_pvt_schedule_summary)

    def _adaptive_offline_training_budget(self, sample_count: int) -> Tuple[int, int]:
        epochs = self._clamp_int(
            int(round(24 + 0.25 * max(0, sample_count))),
            max(30, self._base_pvt_offline_epochs),
            50,
        )
        patience = max(
            self._base_pvt_offline_patience,
            min(12, max(5, epochs // 5)),
        )
        return int(epochs), int(patience)

    def _adaptive_finetune_training_budget(self, sample_count: int) -> Tuple[int, int]:
        epochs = self._clamp_int(
            int(round(6 + sample_count / 80.0)),
            max(6, self._base_pvt_proxy_finetune_epochs),
            24,
        )
        patience = max(
            self._base_pvt_proxy_finetune_patience,
            min(8, max(3, epochs // 3)),
        )
        return int(epochs), int(patience)

    def _pvt_phase(self, step: int) -> str:
        if not self.pvt_outer_loop_enabled:
            return "disabled"
        if not self._vae_offline_trained:
            return "warmup_collection"
        if step <= self.pvt_warmup_steps:
            return "warmup_frozen"
        return "proxy_block_frozen"

    @staticmethod
    def _episode_selection_key(ep: Episode) -> Tuple[str, int]:
        return (str(ep.circuit_spec), int(ep.design_idx))

    @staticmethod
    def _episode_action_distance(ep_a: Episode, ep_b: Episode) -> float:
        try:
            action_a = np.asarray(ep_a.action, dtype=np.float64).reshape(-1)
            action_b = np.asarray(ep_b.action, dtype=np.float64).reshape(-1)
        except Exception:
            return 0.0
        if action_a.shape != action_b.shape or action_a.size == 0:
            return 0.0
        diff = action_a - action_b
        if not np.all(np.isfinite(diff)):
            return 0.0
        return float(np.linalg.norm(diff, ord=2))

    def _select_diverse_candidates(
        self,
        candidates: List[Episode],
        seed_selected: List[Episode],
        count: int,
    ) -> List[Episode]:
        if count <= 0 or not candidates:
            return []

        remaining = list(candidates)
        selected = list(seed_selected)
        chosen: List[Episode] = []
        while remaining and len(chosen) < count:
            if not selected:
                next_ep = max(remaining, key=lambda ep: self._episode_rank_tuple(ep))
            else:
                next_ep = max(
                    remaining,
                    key=lambda ep: (
                        min(self._episode_action_distance(ep, ref) for ref in selected),
                        self._episode_rank_tuple(ep),
                    ),
                )
            chosen.append(next_ep)
            selected.append(next_ep)
            next_key = self._episode_selection_key(next_ep)
            remaining = [ep for ep in remaining if self._episode_selection_key(ep) != next_key]
        return chosen

    def _select_pvt_verification_candidates(self, episodes: List[Episode]) -> List[Episode]:
        if not (self.pvt_outer_loop_enabled and episodes):
            return []

        feasible_episodes = [ep for ep in episodes if bool(getattr(ep, "pm_feasible", False))]
        if not feasible_episodes:
            return []

        if not self._vae_offline_trained:
            total_budget = self.pvt_warmup_verify_topk_per_step + self.pvt_warmup_diverse_per_step
        else:
            total_budget = self.pvt_verify_topk_per_step + self.pvt_verify_uncertain_per_step
        if total_budget <= 0:
            return []

        if not self._vae_offline_trained:
            ranked = sorted(
                feasible_episodes,
                key=lambda ep: self._episode_rank_tuple(ep),
                reverse=True,
            )
            selected: List[Episode] = ranked[: self.pvt_warmup_verify_topk_per_step]
            selected_keys = {self._episode_selection_key(ep) for ep in selected}
            remaining = [ep for ep in ranked if self._episode_selection_key(ep) not in selected_keys]
            selected.extend(
                self._select_diverse_candidates(
                    remaining,
                    seed_selected=selected,
                    count=self.pvt_warmup_diverse_per_step,
                )
            )
            return selected[:total_budget]

        proxy_ready_eps = [
            ep
            for ep in feasible_episodes
            if bool((ep.performance or {}).get("pvt_proxy_ready", False))
        ]
        if not proxy_ready_eps:
            return []

        selected: List[Episode] = []
        ranked_proxy = sorted(proxy_ready_eps, key=self._pvt_proxy_rank_tuple, reverse=True)
        selected.extend(ranked_proxy[: self.pvt_verify_topk_per_step])

        selected_keys = {self._episode_selection_key(ep) for ep in selected}
        remaining = [ep for ep in proxy_ready_eps if self._episode_selection_key(ep) not in selected_keys]
        if self.pvt_verify_uncertain_per_step > 0 and remaining:
            uncertain = sorted(remaining, key=self._pvt_proxy_uncertainty, reverse=True)
            selected.extend(uncertain[: self.pvt_verify_uncertain_per_step])

        deduped: List[Episode] = []
        seen = set()
        for ep in selected:
            key = self._episode_selection_key(ep)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(ep)
        return deduped[:total_budget]

    def _update_verified_pvt_pareto_records(self, episodes: List[Episode], step: int) -> bool:
        archive = list(self.verified_pvt_pareto_records)
        original_size = len(archive)
        old_vectors = {
            tuple(np.asarray(record.get("pareto_objective_vector", []), dtype=np.float64).tolist())
            for record in archive
        }
        for ep in episodes:
            if not bool(getattr(ep, "pm_feasible", False)):
                continue
            candidate = self._build_best_record_payload(
                ep,
                step=step,
                selection_metric="verified_pvt_pareto",
                selection_value=float(ep.reward),
            )
            candidate_vector = np.asarray(candidate.get("pareto_objective_vector", []), dtype=np.float64)
            if candidate_vector.shape[0] != len(self.objective_keys) or not np.all(np.isfinite(candidate_vector)):
                continue

            keep_archive: List[Dict[str, Any]] = []
            dominated = False
            for existing in archive:
                existing_vector = np.asarray(existing.get("pareto_objective_vector", []), dtype=np.float64)
                if existing_vector.shape != candidate_vector.shape:
                    keep_archive.append(existing)
                    continue
                if np.allclose(existing_vector, candidate_vector, atol=1e-8, rtol=1e-8):
                    existing_rank = tuple(existing.get("rank_tuple", []))
                    if tuple(candidate.get("rank_tuple", [])) > existing_rank:
                        continue
                    dominated = True
                    keep_archive.append(existing)
                    continue
                if self._dominates_vector(existing_vector, candidate_vector):
                    dominated = True
                    keep_archive.append(existing)
                    continue
                if self._dominates_vector(candidate_vector, existing_vector):
                    continue
                keep_archive.append(existing)
            if not dominated:
                keep_archive.append(candidate)
            archive = keep_archive

        if len(archive) > self.pvt_verified_archive_capacity:
            archive_vectors = np.stack(
                [np.asarray(record["pareto_objective_vector"], dtype=np.float64) for record in archive],
                axis=0,
            )
            crowding = self._crowding_distance(archive_vectors)
            del archive[int(np.argmin(crowding))]

        archive.sort(key=lambda record: tuple(record.get("rank_tuple", [])), reverse=True)
        self.verified_pvt_pareto_records = archive
        if len(archive) != original_size:
            return True
        new_vectors = {
            tuple(np.asarray(record.get("pareto_objective_vector", []), dtype=np.float64).tolist())
            for record in archive
        }
        return old_vectors != new_vectors

    def _append_vae_training_sample(self, tt_episode: Episode, pvt_episode: Episode) -> bool:
        if not (self.use_vae and self.vae_dataset is not None):
            return False
        bundle = self._build_pvt_proxy_condition(tt_episode)
        if bundle is None:
            return False
        _, tt_objectives, tt_action = bundle
        pvt_perf = pvt_episode.performance or {}
        pvt_targets = build_target_objectives(
            objectives=pvt_perf.get("worst_case_targets", {}) or pvt_perf.get("objective_rewards", {}) or {},
            performance=pvt_perf,
            objective_keys=self.vae_objective_keys,
        )
        vae_data = prepare_vae_training_data(
            tt_action=np.asarray(tt_action, dtype=np.float32),
            tt_performance=tt_episode.performance or {},
            tt_reward=float(tt_episode.reward),
            pvt_worst_reward=float(pvt_episode.reward),
            pvt_worst_performance=pvt_perf,
            pvt_worst_corner_idx=int(getattr(pvt_episode, "selected_corner_idx", -1)),
            num_corners=20,
            tt_objectives=tt_objectives,
            pvt_worst_objectives=pvt_targets,
            objective_keys=self.vae_objective_keys,
            pm_feasible=bool(getattr(pvt_episode, "pm_feasible", False)),
        )
        if vae_data is None:
            return False
        self.vae_dataset.append(vae_data)
        return True

    def _record_vae_proxy_vs_actual(self, step: int, tt_episode: Episode, pvt_episode: Episode) -> None:
        tt_perf = tt_episode.performance or {}
        proxy_mean = tt_perf.get("pvt_proxy_objectives_mean", {}) or {}
        proxy_std = tt_perf.get("pvt_proxy_objectives_std", {}) or {}
        if not proxy_mean:
            return
        actual_targets = build_target_objectives(
            objectives=(pvt_episode.performance or {}).get("worst_case_targets", {}) or (pvt_episode.performance or {}).get("objective_rewards", {}) or {},
            performance=pvt_episode.performance or {},
            objective_keys=self.vae_objective_keys,
        )
        pred_mean_arr = np.asarray(
            [float(proxy_mean.get(key, 0.0)) for key in self.vae_objective_keys],
            dtype=np.float32,
        )
        pred_std_arr = np.asarray(
            [float(proxy_std.get(key, 0.0)) for key in self.vae_objective_keys],
            dtype=np.float32,
        )
        actual_arr = np.asarray(
            [float(actual_targets.get(key, 0.0)) for key in self.vae_objective_keys],
            dtype=np.float32,
        )
        tt_targets = build_target_objectives(
            objectives=tt_perf.get("objective_rewards", {}) or tt_episode.objective_rewards or {},
            performance=tt_perf,
            objective_keys=self.vae_objective_keys,
        )
        tt_constraint_reward = float(tt_targets.get("constraint_reward", 0.0))
        self._vae_pred_records.append(
            (
                int(step),
                tt_constraint_reward,
                float(actual_targets.get("constraint_reward", 0.0)),
                float(proxy_mean.get("constraint_reward", pred_mean_arr[0] if pred_mean_arr.size > 0 else 0.0)),
            )
        )
        self._vae_multi_pred_records.append(
            {
                "step": int(step),
                "tt_reward": tt_constraint_reward,
                "pred_mean": pred_mean_arr.tolist(),
                "pred_std": pred_std_arr.tolist(),
                "actual": actual_arr.tolist(),
            }
        )

    def _verify_pvt_episodes(
        self,
        episodes: List[Episode],
        step: int,
        update_dataset: bool = True,
        update_archive: bool = True,
    ) -> List[Episode]:
        if not episodes:
            return []

        actions = [np.asarray(ep.action, dtype=np.float32).copy() for ep in episodes]
        try:
            _, _, _, _, perf_infos = self.env.parallel_step(actions, enable_pvt=True)
        except Exception as exc:
            print(f"[PVT verify] failed: {exc}")
            return []

        verified_episodes: List[Episode] = []
        pred_actual_pairs: List[Tuple[float, float]] = []

        for idx, ep in enumerate(episodes):
            perf_info = perf_infos[idx] if idx < len(perf_infos) else {}
            verified_ep = self._build_episode_from_perf(
                circuit_spec=ep.circuit_spec,
                state=ep.state,
                action=ep.action,
                perf_info=copy.deepcopy(perf_info) if isinstance(perf_info, dict) else perf_info,
                design_idx=ep.design_idx,
                old_log_prob=ep.old_log_prob,
                verbose=False,
            )
            verified_episodes.append(verified_ep)

            perf = ep.performance if isinstance(ep.performance, dict) else {}
            perf["pvt_verified"] = True
            perf["pvt_verified_step"] = int(step)
            perf["pvt_verified_training_reward"] = float(verified_ep.reward)
            perf["pvt_verified_utility"] = float(getattr(verified_ep, "utility", verified_ep.reward))
            perf["pvt_verified_pm_violation"] = float(getattr(verified_ep, "pm_violation", np.inf))
            perf["pvt_verified_pm_feasible"] = bool(getattr(verified_ep, "pm_feasible", False))
            perf["pvt_verified_selected_corner_idx"] = int(getattr(verified_ep, "selected_corner_idx", -1))
            perf["pvt_verified_objectives"] = dict(verified_ep.objective_rewards or {})
            perf["pvt_verified_performance"] = copy.deepcopy(verified_ep.performance or {})
            ep.performance = perf

            if update_dataset:
                self._append_vae_training_sample(ep, verified_ep)
            if bool(perf.get("pvt_proxy_ready", False)):
                self._record_vae_proxy_vs_actual(step, ep, verified_ep)
                pred_actual_pairs.append(
                    (
                        float((perf.get("pvt_proxy_objectives_mean", {}) or {}).get("constraint_reward", 0.0)),
                        float((verified_ep.performance or {}).get("worst_case_targets", {}).get("constraint_reward", verified_ep.objective_rewards.get("constraint_reward", 0.0))),
                    )
                )

        if update_archive and verified_episodes:
            self._update_verified_pvt_pareto_records(verified_episodes, step)

        if pred_actual_pairs:
            worst_idx = min(range(len(pred_actual_pairs)), key=lambda i: float(pred_actual_pairs[i][1]))
            pred_reward, actual_reward = pred_actual_pairs[worst_idx]
            self.vae_pred_worst_history.append(float(pred_reward))
            self.vae_actual_worst_history.append(float(actual_reward))

        return verified_episodes

    def _maybe_train_pvt_proxy_model(self, step: int) -> None:
        if not (self.pvt_outer_loop_enabled and self.use_vae and self.vae is not None):
            return
        if not self._adaptive_pvt_schedule_ready:
            self.configure_adaptive_pvt_schedule(max(int(step), 1))
        sample_count = len(self.vae_dataset) if isinstance(self.vae_dataset, list) else 0
        self._last_pvt_dataset_size = sample_count
        if not self._vae_offline_trained:
            print(
                f"  PVT VAE phase={self._pvt_phase(step)} dataset: "
                f"{sample_count}/{self.vae_min_samples} verified samples"
            )
            if step <= self.pvt_warmup_steps:
                return
            if sample_count >= self.vae_min_samples:
                self.pvt_offline_epochs, self.pvt_offline_patience = self._adaptive_offline_training_budget(sample_count)
                self._adaptive_pvt_schedule_summary["resolved_offline_epochs"] = int(self.pvt_offline_epochs)
                self._adaptive_pvt_schedule_summary["resolved_offline_patience"] = int(self.pvt_offline_patience)
                self._adaptive_pvt_schedule_summary["offline_dataset_size"] = int(sample_count)
                self._run_vae_offline_training(sample_count)
                self._last_pvt_finetune_step = int(step)
                self._last_pvt_finetune_dataset_size = int(sample_count)
            return

        if self._last_pvt_verified_count <= 0:
            return
        if step <= self.pvt_warmup_steps:
            return
        steps_since_last = step - self._last_pvt_finetune_step
        new_samples_since_last = sample_count - self._last_pvt_finetune_dataset_size
        if (
            steps_since_last < self.pvt_proxy_finetune_interval
            and new_samples_since_last < self._pvt_target_new_samples_per_finetune
        ):
            return
        if sample_count <= 0:
            return

        self.pvt_proxy_finetune_epochs, self.pvt_proxy_finetune_patience = self._adaptive_finetune_training_budget(sample_count)
        self._adaptive_pvt_schedule_summary["resolved_finetune_epochs"] = int(self.pvt_proxy_finetune_epochs)
        self._adaptive_pvt_schedule_summary["resolved_finetune_patience"] = int(self.pvt_proxy_finetune_patience)
        self._adaptive_pvt_schedule_summary["latest_finetune_dataset_size"] = int(sample_count)
        losses = self._train_vae_dataset_with_patience(
            epochs=self.pvt_proxy_finetune_epochs,
            patience=self.pvt_proxy_finetune_patience,
            stage_label="block fine-tune",
        )
        if losses:
            self._last_pvt_finetune_step = int(step)
            self._last_pvt_finetune_dataset_size = int(sample_count)
            print(
                f"  PVT VAE block fine-tune: {len(losses)} epochs, "
                f"last_loss={losses[-1]:.4f}, dataset={sample_count}"
            )

    def _episode_rank_tuple(self, ep: Episode) -> Tuple[float, ...]:
        objective_rewards = ep.objective_rewards or {}
        constraint_reward = float(objective_rewards.get("constraint_reward", -1e6))
        secondary_sum = float(
            objective_rewards.get("FOML_score", 0.0)
            + objective_rewards.get("FOMS_score", 0.0)
            + objective_rewards.get("Active_Area_score", 0.0)
        )
        pm_feasible_flag = 1.0 if bool(getattr(ep, "pm_feasible", False)) else 0.0
        pm_violation = float(getattr(ep, "pm_violation", np.inf))
        training_reward = float(ep.reward)
        if not np.isfinite(training_reward):
            training_reward = -1e6
        if not np.isfinite(constraint_reward):
            constraint_reward = -1e6
        if not np.isfinite(secondary_sum):
            secondary_sum = -1e6
        if not np.isfinite(pm_violation):
            pm_violation = np.inf
        return (
            pm_feasible_flag,
            training_reward,
            constraint_reward,
            secondary_sum,
            -pm_violation,
        )

    def _build_best_record_payload(
        self,
        ep: Episode,
        step: int,
        selection_metric: str,
        selection_value: Optional[float] = None,
    ) -> Dict[str, Any]:
        perf = ep.performance or {}
        raw_r = perf.get("raw_reward", perf.get("reward", None))
        objective_rewards = dict(ep.objective_rewards or {})
        rank_tuple = self._episode_rank_tuple(ep)
        filtered_perf = get_filtered_performance(perf)
        raw_metric_summary = {
            key: self._json_safe(value)
            for key, value in get_reporting_metrics(filtered_perf).items()
        }
        payload = {
            "selection_metric": selection_metric,
            "selection_value": None if selection_value is None else float(selection_value),
            "rank_tuple": [float(x) for x in rank_tuple],
            "training_reward": float(ep.reward),
            "utility": float(getattr(ep, "utility", ep.reward)),
            "found_at_step": int(step),
            "circuit_spec": ep.circuit_spec,
            "design_idx": int(ep.design_idx),
            "evaluation_source": getattr(ep, "evaluation_source", "unknown"),
            "pm_feasible": bool(getattr(ep, "pm_feasible", False)),
            "pm_violation": float(getattr(ep, "pm_violation", np.inf)),
            "selected_corner_idx": int(getattr(ep, "selected_corner_idx", -1)),
            "raw_reward_in_perf": self._json_safe(raw_r),
            "objective_rewards": self._json_safe(objective_rewards),
            "raw_metric_summary": raw_metric_summary,
            "performance": self._json_safe(filtered_perf),
            "performance_signals": self._json_safe(get_reporting_signals(perf)),
        }
        if bool(getattr(ep, "pm_feasible", False)):
            payload["pareto_objective_vector"] = [
                float(objective_rewards.get(key, -1e6)) for key in self.objective_keys
            ]
        return payload

    def _metric_value_from_episode(self, ep: Episode, metric_key: str) -> Optional[float]:
        if metric_key == "training_reward":
            value = float(ep.reward)
            return value if np.isfinite(value) else None
        if metric_key == "utility":
            value = float(getattr(ep, "utility", np.nan))
            return value if np.isfinite(value) else None
        if metric_key == "pm_violation":
            value = float(getattr(ep, "pm_violation", np.nan))
            return value if np.isfinite(value) else None

        objective_rewards = ep.objective_rewards or {}
        if metric_key in objective_rewards:
            value = float(objective_rewards.get(metric_key, np.nan))
            return value if np.isfinite(value) else None

        perf = ep.performance or {}
        value = perf.get(metric_key, None)
        if value is None:
            return None
        try:
            numeric = float(value)
        except Exception:
            return None
        return numeric if np.isfinite(numeric) else None

    def _is_better_metric_value(
        self,
        new_value: Optional[float],
        current_value: Optional[float],
        mode: str,
    ) -> bool:
        if new_value is None:
            return False
        if current_value is None or not np.isfinite(current_value):
            return True
        tolerance = 1e-12
        if mode == "min":
            return new_value < current_value - tolerance
        return new_value > current_value + tolerance

    def _update_best_objective_records(self, episodes: List[Episode], step: int) -> List[str]:
        metric_specs = [
            ("constraint_reward", "max", True),
            ("FOML_score", "max", True),
            ("FOMS_score", "max", True),
            ("Active_Area_score", "max", True),
            ("FOML", "max", True),
            ("FOMS", "max", True),
            ("Active Area", "min", True),
            ("training_reward", "max", True),
            ("utility", "max", True),
        ]
        updated_metrics: List[str] = []

        for ep in episodes:
            for metric_key, mode, feasible_only in metric_specs:
                if feasible_only and not bool(getattr(ep, "pm_feasible", False)):
                    continue
                new_value = self._metric_value_from_episode(ep, metric_key)
                current_value = self._best_objective_values.get(metric_key)
                current_record = self.best_objective_records.get(metric_key)
                current_rank = tuple(current_record.get("rank_tuple", [])) if current_record else tuple()
                new_rank = self._episode_rank_tuple(ep)
                should_update = self._is_better_metric_value(new_value, current_value, mode)
                if (
                    not should_update
                    and new_value is not None
                    and current_value is not None
                    and abs(float(new_value) - float(current_value)) <= 1e-12
                    and new_rank > current_rank
                ):
                    should_update = True
                if not should_update:
                    continue
                payload = self._build_best_record_payload(
                    ep,
                    step=step,
                    selection_metric=metric_key,
                    selection_value=new_value,
                )
                payload["selection_mode"] = mode
                payload["feasible_only"] = bool(feasible_only)
                self._best_objective_values[metric_key] = float(new_value)
                self.best_objective_records[metric_key] = payload
                if metric_key not in updated_metrics:
                    updated_metrics.append(metric_key)
        return updated_metrics

    def _update_best_pareto_records(self, episodes: List[Episode], step: int) -> bool:
        archive = list(self.best_pareto_records)
        original_size = len(archive)
        old_vectors = {
            tuple(np.asarray(record.get("pareto_objective_vector", []), dtype=np.float64).tolist())
            for record in archive
        }
        for ep in episodes:
            if not bool(getattr(ep, "pm_feasible", False)):
                continue
            candidate = self._build_best_record_payload(
                ep,
                step=step,
                selection_metric="pareto_archive",
                selection_value=float(ep.reward),
            )
            candidate_vector = np.asarray(
                candidate.get("pareto_objective_vector", []),
                dtype=np.float64,
            )
            if candidate_vector.shape[0] != len(self.objective_keys) or not np.all(np.isfinite(candidate_vector)):
                continue

            keep_archive: List[Dict[str, Any]] = []
            dominated = False
            for existing in archive:
                existing_vector = np.asarray(
                    existing.get("pareto_objective_vector", []),
                    dtype=np.float64,
                )
                if existing_vector.shape != candidate_vector.shape:
                    keep_archive.append(existing)
                    continue
                if np.allclose(existing_vector, candidate_vector, atol=1e-8, rtol=1e-8):
                    existing_rank = tuple(existing.get("rank_tuple", []))
                    if tuple(candidate.get("rank_tuple", [])) > existing_rank:
                        continue
                    dominated = True
                    keep_archive.append(existing)
                    continue
                if self._dominates_vector(existing_vector, candidate_vector):
                    dominated = True
                    keep_archive.append(existing)
                    continue
                if self._dominates_vector(candidate_vector, existing_vector):
                    continue
                keep_archive.append(existing)
            if not dominated:
                keep_archive.append(candidate)
            archive = keep_archive

        if len(archive) > self.best_record_archive_capacity:
            archive_vectors = np.stack(
                [np.asarray(record["pareto_objective_vector"], dtype=np.float64) for record in archive],
                axis=0,
            )
            crowding = self._crowding_distance(archive_vectors)
            remove_idx = int(np.argmin(crowding))
            del archive[remove_idx]

        archive.sort(key=lambda record: tuple(record.get("rank_tuple", [])), reverse=True)
        self.best_pareto_records = archive
        if len(archive) != original_size:
            return True
        new_vectors = {
            tuple(np.asarray(record.get("pareto_objective_vector", []), dtype=np.float64).tolist())
            for record in archive
        }
        return old_vectors != new_vectors

    def _maybe_update_best(self, episodes: List[Episode], step: int):
        """Update global best-performance record using PM-feasible-first ranking."""
        if not episodes:
            return
        objective_updates = self._update_best_objective_records(episodes, step)
        pareto_archive_updated = self._update_best_pareto_records(episodes, step)
        global_rank_updated = False
        for ep in episodes:
            rank_tuple = self._episode_rank_tuple(ep)
            if self._best_rank_tuple is not None and rank_tuple <= self._best_rank_tuple:
                continue
            performance_record = self._build_best_record_payload(
                ep,
                step=step,
                selection_metric="global_rank",
                selection_value=float(ep.reward),
            )
            performance_record["selection_rule"] = {
                "primary": "pm_feasible",
                "secondary": "training_reward",
                "tertiary": "constraint_reward",
                "quaternary": "FOML+FOMS+Active_Area",
            }
            performance_record["best_reward"] = float(ep.reward)
            performance_record["best_performance_reward"] = float(ep.reward)
            performance_record["best_utility"] = float(getattr(ep, "utility", ep.reward))
            self._best_rank_tuple = rank_tuple
            self.best_performance_reward = float(ep.reward)
            self.best_performance_step = int(step)
            self.best_performance_record = performance_record
            self.best_reward = self.best_performance_reward
            self.best_step = self.best_performance_step
            self.best_record = copy.deepcopy(performance_record)
            global_rank_updated = True
        self._latest_best_update_summary = {
            "global_rank_updated": bool(global_rank_updated),
            "global_rank_step": int(self.best_performance_step) if self.best_performance_step is not None else None,
            "objective_updates": list(objective_updates),
            "pareto_archive_updated": bool(pareto_archive_updated),
            "pareto_archive_size": int(len(self.best_pareto_records)),
        }

    def _record_detail_lines(self, record: Dict[str, Any], prefix: str = "  ") -> List[str]:
        lines = [
            (
                f"{prefix}step={record.get('found_at_step')} design_idx={record.get('design_idx')} "
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

    def _save_best_record_txt(self, step: int, log_dir: str = None):
        if log_dir is None:
            log_dir = self._logs_dir
        """Dump current global/multi-objective best records under logs/ with step in filename."""
        if self.best_performance_record is None:
            return
        os.makedirs(log_dir, exist_ok=True)
        pareto_payload = json.dumps(self.best_pareto_records, ensure_ascii=False, indent=2)
        pareto_out_paths = [
            os.path.join(log_dir, f"best_pareto_archive_step_{step}.json"),
            os.path.join(log_dir, "best_pareto_archive_latest.json"),
        ]
        for out_path in pareto_out_paths:
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(pareto_payload)

        verified_pvt_payload = json.dumps(self.verified_pvt_pareto_records, ensure_ascii=False, indent=2)
        verified_pvt_out_paths = [
            os.path.join(log_dir, f"verified_pvt_pareto_archive_step_{step}.json"),
            os.path.join(log_dir, "verified_pvt_pareto_archive_latest.json"),
        ]
        for out_path in verified_pvt_out_paths:
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(verified_pvt_payload)

        feasible_objective_records = {
            key: record
            for key, record in self.best_objective_records.items()
            if bool(record.get("pm_feasible", False))
        }
        objective_payload = json.dumps(feasible_objective_records, ensure_ascii=False, indent=2)
        objective_out_paths = [
            os.path.join(log_dir, f"best_objective_records_step_{step}.json"),
            os.path.join(log_dir, "best_objective_records_latest.json"),
        ]
        for out_path in objective_out_paths:
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(objective_payload)

        summary_lines = [
            "Multi-Objective Best Summary",
            "=" * 80,
            "note: these records are for monitoring/documentation only and do not affect policy update.",
            f"tt_pareto_archive_size: {len(self.best_pareto_records)}",
            f"verified_pvt_pareto_archive_size: {len(self.verified_pvt_pareto_records)}",
            "objective_bests:",
        ]
        if feasible_objective_records:
            for metric_key, record in sorted(feasible_objective_records.items()):
                summary_lines.append(f"  {metric_key}:")
                summary_lines.extend(self._record_detail_lines(record, prefix="    "))
        else:
            summary_lines.append("  none")
        summary_lines.append("top_tt_pareto_designs:")
        top_pareto_records = self.best_pareto_records[: min(5, len(self.best_pareto_records))]
        if top_pareto_records:
            for idx, record in enumerate(top_pareto_records, start=1):
                summary_lines.append(f"  #{idx}")
                summary_lines.extend(self._record_detail_lines(record, prefix="    "))
        else:
            summary_lines.append("  none")
        summary_lines.append("verified_pvt_top_pareto_designs:")
        top_verified_pvt_records = self.verified_pvt_pareto_records[: min(5, len(self.verified_pvt_pareto_records))]
        if top_verified_pvt_records:
            for idx, record in enumerate(top_verified_pvt_records, start=1):
                summary_lines.append(f"  #{idx}")
                summary_lines.extend(self._record_detail_lines(record, prefix="    "))
        else:
            summary_lines.append("  none")
        summary_text = "\n".join(summary_lines) + "\n"
        summary_out_paths = [
            os.path.join(log_dir, f"best_multiobjective_summary_step_{step}.txt"),
            os.path.join(log_dir, "best_multiobjective_summary_latest.txt"),
        ]
        for out_path in summary_out_paths:
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(summary_text)
                f.write("\n")

    # --------------------
    # WandB helpers
    # --------------------
    def _to_native(self, v):
        """Convert numpy/tensor types to Python native for logging."""
        try:
            import torch as _torch
        except Exception:
            _torch = None

        if _torch is not None and isinstance(v, _torch.Tensor):
            v = v.detach().cpu().numpy()
        if isinstance(v, np.ndarray):
            if v.size == 1:
                v = v.item()
            else:
                return None
        if isinstance(v, (np.floating, np.integer)):
            v = v.item()
        if isinstance(v, (float, int)):
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                return None
        return v

    def _wandb_log(self, data: Dict, step: int):
        if self.wandb is None:
            return
        try:
            safe = {k: self._to_native(v) for k, v in data.items()}
            self.wandb.log(safe, step=step)
        except Exception:
            pass

    # --------------------
    # Advantage shaping
    # --------------------
    def _shape_advantage(self, adv: float) -> float:
        """
        Optionally clip or squash the raw advantage to stabilise training.
        """
        if self.use_adv_tanh:
            return float(np.tanh(adv))
        if self.adv_clip_range is not None and self.adv_clip_range > 0:
            return float(np.clip(adv, -self.adv_clip_range, self.adv_clip_range))
        return float(adv)

    # --------------------
    # Policy interface
    # --------------------
    def select_action(self, state: np.ndarray) -> np.ndarray:
        """ 
        Sample an action from the policy using model's mean and variance.
        """ 
        state = np.asarray(state, dtype=np.float32)
        if not np.all(np.isfinite(state)):
            print("[WARN] Non-finite state in select_action, sanitising to 0.")
            state = np.nan_to_num(state, nan=0.0, posinf=0.0, neginf=0.0)

        state_tensor = torch.from_numpy(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            mean, std = self.policy(state_tensor)
            dist = torch.distributions.Normal(mean, std)
            action = dist.sample()

        action = action.cpu().numpy()[0]
        action = np.clip(action, -1.0, 1.0)
        return action

    # --------------------
    # Rollout
    # --------------------
    def rollout(self, num_circuits: int = 1, verbose: bool = True, step: int = 1) -> List[Episode]:
        """
        Generate multiple designs for circuit specifications.
        """
        episodes: List[Episode] = []
        self._last_pvt_verified_count = 0
        self._last_pvt_proxy_count = 0
        self._last_pvt_proxy_ready = False

        for circuit_idx in range(num_circuits):
            # initial state: use cached if available
            if self._cache_is_valid and self._cached_initial_state is not None:
                state = self._cached_initial_state.copy()
                if not np.all(np.isfinite(state)):
                    print("[WARN] Non-finite cached initial state, sanitising to 0.")
                    state = np.nan_to_num(state, nan=0.0, posinf=0.0, neginf=0.0)
                info = {} if self.initial_reset_info is None else copy.deepcopy(self.initial_reset_info)
                if verbose:
                    print("[INFO] Using reset_state,info (no SPICE reset)")
            else:
                state, info = self.env.reset()
                state = np.asarray(state, dtype=np.float32)
                if not np.all(np.isfinite(state)):
                    print("[WARN] Non-finite state from env.reset, sanitising to 0.")
                    state = np.nan_to_num(state, nan=0.0, posinf=0.0, neginf=0.0)
                self._cached_initial_state = state.copy()
                self._cache_is_valid = True
                if verbose:
                    print("[INFO] Initial state reset (SPICE simulation performed)")

            circuit_spec = f"circuit_{circuit_idx}"
            if verbose:
                print(f"\nRollout for {circuit_spec}:")

            # multiple designs per circuit - batch processing for efficiency
            # Prepare batch of states
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

            # Batch compute actions and log_probs
            with torch.no_grad():
                mean, std = self.policy(state_tensor)
                dist = torch.distributions.Normal(mean, std)
                actions = dist.sample((self.num_designs_per_circuit,))
                actions = actions.squeeze(1)
                actions = torch.clamp(actions, -1, 1)
                old_log_probs = dist.log_prob(actions).sum(dim=1)
            # Convert to numpy
            actions = actions.cpu().numpy()
            old_log_probs = old_log_probs.cpu().numpy()
            
            
            next_states, rewards, terminateds, truncateds, perf_infos = self.env.parallel_step(
                actions, 
                enable_pvt=self.run_all_corners
            )

            
            for i, result in enumerate(zip(next_states, rewards, terminateds, truncateds, perf_infos)):
                next_state, reward, terminated, truncated, perf_info = result
                reward = float(reward)
                
                # reward NaN / Inf guard
                if not np.isfinite(reward):
                    if verbose:
                        print(
                            f"[WARN] Non-finite reward detected: {reward}, "
                            f"circuit={circuit_spec}, design_idx={i}"
                        )
                    # since "closer to 0 is better", use a very bad value
                    reward = -1e6
                
                ep = self._build_episode_from_perf(
                    circuit_spec=circuit_spec,
                    state=state,
                    action=actions[i],
                    perf_info=copy.deepcopy(perf_info) if isinstance(perf_info, dict) else perf_info,
                    design_idx=i,
                    old_log_prob=float(old_log_probs[i]),
                    verbose=verbose,
                )
                episodes.append(ep)

                # corner selection stats
                if self.run_all_corners and isinstance(ep.performance, dict):
                    corner_val = ep.performance.get("pvt_selected_corner")
                    key = None
                    if isinstance(corner_val, dict):
                        proc = corner_val.get("proc")
                        temp = corner_val.get("temp")
                        vdd = corner_val.get("vdd")
                        if proc is not None and temp is not None and vdd is not None:
                            key = f"{str(proc).upper()} {str(temp)} {str(vdd)}"
                        else:
                            idx = corner_val.get("index")
                            key = f"IDX_{idx}" if idx is not None else None
                    elif isinstance(corner_val, (str, int, float)):
                        key = str(corner_val)
                    if key:
                        self.corner_selection_counter[key] = self.corner_selection_counter.get(key, 0) + 1
                
                # collect VAE training data
                if (
                    self.use_vae
                    and self._vae_training_active
                    and self.vae_dataset is not None
                    and self.run_all_corners
                    and isinstance(ep.performance, dict)
                ):
                    extra_corner = ep.performance.get("extra_corner")
                    if isinstance(extra_corner, dict):
                        perf_extra = extra_corner.get("performance", {})
                        tt_reward = extra_corner.get("reward", None)
                        sel_corner = ep.performance.get("pvt_selected_corner", {})
                        pvt_worst_idx = (sel_corner or {}).get("index", None)
                        pvt_worst_reward = ep.reward

                        if (
                            pvt_worst_reward is not None
                            and pvt_worst_idx is not None
                        ):
                            tt_action = perf_extra.get("real_action")
                            if tt_action is None:
                                try:
                                    from utils import ActionNormalizer

                                    tt_action = ActionNormalizer(
                                        self.env.action_space_low, self.env.action_space_high
                                    ).action(np.array(actions[i], copy=True))
                                except Exception:
                                    tt_action = None

                            if tt_action is not None:
                                tt_objectives = build_target_objectives(
                                    objectives=perf_extra.get("objective_rewards", {}) or {},
                                    performance=perf_extra,
                                    objective_keys=self.vae_objective_keys,
                                )
                                pvt_worst_objectives = build_target_objectives(
                                    objectives=ep.performance.get("worst_case_targets", {}) or ep.performance.get("objective_rewards", {}) or {},
                                    performance=ep.performance,
                                    objective_keys=self.vae_objective_keys,
                                )
                                vae_data = prepare_vae_training_data(
                                    tt_action=np.array(tt_action, dtype=np.float32),
                                    tt_performance=perf_extra,
                                    tt_reward=None if tt_reward is None else float(tt_reward),
                                    pvt_worst_reward=float(pvt_worst_reward),
                                    pvt_worst_performance=ep.performance,
                                    pvt_worst_corner_idx=int(pvt_worst_idx),
                                    num_corners=20,
                                    tt_objectives=tt_objectives,
                                    pvt_worst_objectives=pvt_worst_objectives,
                                    objective_keys=self.vae_objective_keys,
                                    pm_feasible=bool(ep.pm_feasible),
                                )
                                if vae_data is not None and float(pvt_worst_reward) > self.vae_reward_threshold:
                                    self.vae_dataset.append(vae_data)
                
                # reset to cached initial state for next design
                state = self._cached_initial_state.copy()
                if not np.all(np.isfinite(state)):
                    print("[WARN] Non-finite cached initial state (loop), sanitising to 0.")
                    state = np.nan_to_num(state, nan=0.0, posinf=0.0, neginf=0.0)

        feasible_vectors: List[np.ndarray] = []
        for ep in episodes:
            if not getattr(ep, "pm_feasible", False):
                continue
            feasible_vectors.append(
                self._objective_vector(
                    {key: float(ep.objective_rewards.get(key, 0.0)) for key in self.objective_keys}
                )
            )
        self._update_utility_reference_points(feasible_vectors)

        if self.pvt_outer_loop_enabled:
            self._attach_pvt_proxy_predictions(episodes)
            verify_candidates = self._select_pvt_verification_candidates(episodes)
            verified_episodes = self._verify_pvt_episodes(
                verify_candidates,
                step=step,
                update_dataset=True,
                update_archive=True,
            )
            self._last_verified_pvt_episodes = list(verified_episodes)
            self._last_pvt_verified_count = len(verified_episodes)
        else:
            self._last_verified_pvt_episodes = []
            self._last_pvt_verified_count = 0

        return episodes

    # --------------------
    # Group-relative advantages + group weights
    # --------------------
    def compute_group_relative_advantages(self, episodes: List[Episode]) -> List[Episode]:
        """Compute per-objective group-relative advantages and aggregate in whitened objective geometry."""
        groups: Dict[str, List[Episode]] = defaultdict(list)
        for ep in episodes:
            groups[ep.circuit_spec].append(ep)

        group_mean_rewards: Dict[str, float] = {}
        group_objective_means: Dict[str, Dict[str, float]] = {}
        batch_objective_vectors: List[np.ndarray] = []

        for circuit_spec, group_eps in groups.items():
            utilities = np.array([ep.reward for ep in group_eps], dtype=float)
            finite_utilities = utilities[np.isfinite(utilities)]
            group_mean_rewards[circuit_spec] = float(np.mean(finite_utilities)) if finite_utilities.size > 0 else 0.0
            group_objective_means[circuit_spec] = {}

            for objective_key in self.objective_keys:
                objective_values = np.array(
                    [ep.objective_rewards.get(objective_key, 0.0) for ep in group_eps],
                    dtype=float,
                )
                if not np.all(np.isfinite(objective_values)):
                    bad_mask = ~np.isfinite(objective_values)
                    print(
                        f"[WARN] Non-finite values in objective {objective_key} for group {circuit_spec}: "
                        f"indices={np.where(bad_mask)[0]}"
                    )
                    objective_values[bad_mask] = -1e6
                mean_value = float(objective_values.mean())
                group_objective_means[circuit_spec][objective_key] = mean_value
                normalized_values, objective_baseline, objective_scale = self._decoupled_normalize_group_values(
                    objective_values
                )
                for idx, ep in enumerate(group_eps):
                    ep.objective_advantages[objective_key] = float(normalized_values[idx])
                    ep.objective_advantages[f"{objective_key}_raw"] = float(objective_values[idx])
                    ep.objective_advantages[f"{objective_key}_mean"] = float(mean_value)
                    ep.objective_advantages[f"{objective_key}_baseline"] = float(objective_baseline)
                    ep.objective_advantages[f"{objective_key}_scale"] = float(objective_scale)

            for idx, ep in enumerate(group_eps):
                objective_vector = np.asarray(
                    [ep.objective_advantages.get(objective_key, 0.0) for objective_key in self.objective_keys],
                    dtype=np.float64,
                )
                batch_objective_vectors.append(objective_vector)
                aggregated_advantage, aggregation_mode = self._aggregate_objective_advantages(objective_vector)
                ep.advantage = float(aggregated_advantage)
                ep.objective_advantages["aggregated_decoupled_advantage"] = float(
                    self._utility_weight_vector().astype(np.float64) @ objective_vector
                )
                ep.objective_advantages["covariance_whitened_advantage"] = float(ep.advantage)
                ep.objective_advantages["advantage_aggregation_mode"] = aggregation_mode
                if self._covariance_whitening_transform is not None:
                    whitened_vector = self._covariance_whitening_transform @ objective_vector
                    for objective_idx, objective_key in enumerate(self.objective_keys):
                        ep.objective_advantages[f"{objective_key}_whitened"] = float(whitened_vector[objective_idx])

        group_weights: Dict[str, float] = {spec: 1.0 for spec in group_mean_rewards.keys()}

        current_step = int(self.total_steps) + 1
        self._append_covariance_advantage_buffer(batch_objective_vectors)
        self._maybe_freeze_covariance_whitener(current_step)

        self._last_group_mean_rewards = group_mean_rewards
        self._last_group_weights = group_weights
        self._last_group_objective_means = group_objective_means

        return episodes

    # --------------------
    # Policy update
    # --------------------
    def update_policy(self, episodes: List[Episode], verbose: bool = True) -> Tuple[float, float]:
        """
        PPO/GRPO-style policy update with clipped surrogate objective.

        Key changes from REINFORCE:
        1. Uses importance sampling ratio: ratio = exp(new_log_prob - old_log_prob)
        2. Clipped surrogate objective: min(ratio * adv, clip(ratio) * adv)
        3. Multiple epochs per batch (controlled by ppo_epochs)
        4. Entropy bonus for exploration
        5. Early stopping based on KL divergence threshold
        """
        num_episodes = len(episodes)
        if num_episodes == 0:
            return 0.0, 0.0

        # Filter valid episodes
        valid_episodes = []
        for ep in episodes:
            if not np.isfinite(ep.reward) or not np.isfinite(ep.advantage):
                continue
            if not np.all(np.isfinite(ep.state)) or not np.all(np.isfinite(ep.action)):
                continue
            if not np.isfinite(ep.old_log_prob):
                continue
            valid_episodes.append(ep)

        if not valid_episodes:
            print("[WARN] No valid episodes in update_policy (all skipped).")
            return 0.0, 0.0

        use_covariance_whitened_update = bool(
            self.covariance_whitening_enabled
            and self._covariance_whitener_ready
            and self._covariance_freeze_step is not None
            and (int(self.total_steps) + 1) > int(self._covariance_freeze_step)
        )

        # Convert to tensors for batch processing
        states = torch.FloatTensor(np.array([ep.state for ep in valid_episodes])).to(self.device)
        actions = torch.FloatTensor(np.array([ep.action for ep in valid_episodes])).to(self.device)
        old_log_probs = torch.FloatTensor([ep.old_log_prob for ep in valid_episodes]).to(self.device)
        if use_covariance_whitened_update:
            scalar_advantages = torch.FloatTensor(
                [self._shape_advantage(ep.advantage) for ep in valid_episodes]
            ).to(self.device)
        else:
            objective_advantages = torch.FloatTensor(
                np.array(
                    [
                        [
                            self._shape_advantage(ep.objective_advantages.get(objective_key, 0.0))
                            for objective_key in self.objective_keys
                        ]
                        for ep in valid_episodes
                    ],
                    dtype=np.float32,
                )
            ).to(self.device)
            objective_weight_tensor = torch.FloatTensor(self._utility_weight_vector()).to(self.device)

        # Group weights
        group_weights_dict = getattr(self, "_last_group_weights", None)
        if isinstance(group_weights_dict, dict):
            gw_list = [group_weights_dict.get(ep.circuit_spec, 1.0) for ep in valid_episodes]
            group_weights = torch.FloatTensor([gw if np.isfinite(gw) else 1.0 for gw in gw_list]).to(self.device)
        else:
            group_weights = torch.ones(len(valid_episodes), device=self.device)

        # PPO training loop
        total_loss = 0.0
        total_grad_norm = 0.0
        total_kl = 0.0
        total_clip_fraction = 0.0
        num_updates = 0

        for epoch in range(self.ppo_epochs):
            self.optimizer.zero_grad()

            # Compute new log probabilities under current policy
            mu, std = self.policy(states)
            dist = torch.distributions.Normal(mu, std)
            new_log_probs = dist.log_prob(actions).sum(dim=-1)

            # Importance sampling ratio
            log_ratio = new_log_probs - old_log_probs
            log_ratio = torch.clamp(log_ratio, -20.0, 20.0)  # Prevent overflow
            ratio = torch.exp(log_ratio)
            clipped_ratio = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon)

            if use_covariance_whitened_update:
                surr1 = ratio * scalar_advantages * group_weights
                surr2 = clipped_ratio * scalar_advantages * group_weights
                policy_loss = -torch.min(surr1, surr2).mean()
            else:
                ratio_expanded = ratio.unsqueeze(-1)
                clipped_ratio_expanded = clipped_ratio.unsqueeze(-1)
                weighted_advantages = objective_advantages * objective_weight_tensor.unsqueeze(0)
                surr1 = ratio_expanded * weighted_advantages
                surr2 = clipped_ratio_expanded * weighted_advantages
                per_episode_surrogate = torch.min(surr1, surr2).sum(dim=-1) * group_weights
                policy_loss = -per_episode_surrogate.mean()

            # Entropy bonus (encourages exploration)
            entropy = dist.entropy().sum(dim=-1).mean()
            entropy_loss = -self.entropy_coef * entropy

            # Total loss
            loss = policy_loss + entropy_loss
            # loss = policy_loss

            # Backward pass
            loss.backward()

            # Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=self.max_grad_norm)

            # Optimizer step
            self.optimizer.step()

            # Compute approximate KL divergence for monitoring
            with torch.no_grad():
                ratio1 = torch.exp(-log_ratio)
                approx_kl = ((ratio1 - 1) + log_ratio).mean().item()
                clip_fraction = ((ratio - 1.0).abs() > self.clip_epsilon).float().mean().item()

            total_loss += float(loss.item())
            total_grad_norm += float(grad_norm.item()) if isinstance(grad_norm, torch.Tensor) else float(grad_norm)
            total_kl += approx_kl
            total_clip_fraction += clip_fraction
            num_updates += 1

            # Early stopping if KL divergence is too large
            if approx_kl > self.target_kl:
                if verbose:
                    print(f"  Early stopping at epoch {epoch+1}/{self.ppo_epochs}: KL={approx_kl:.4f} > target={self.target_kl}")
                break

        # Average metrics
        avg_loss = total_loss / max(num_updates, 1)
        avg_grad_norm = total_grad_norm / max(num_updates, 1)
        avg_kl = total_kl / max(num_updates, 1)
        avg_clip_fraction = total_clip_fraction / max(num_updates, 1)

        # Store metrics for tracking
        self.kl_history.append(avg_kl)
        self.clip_fraction_history.append(avg_clip_fraction)
        self.entropy_history.append(entropy)

        if verbose:
            print(f"  Loss: {avg_loss:.4f}, entropy: {entropy:.4f}, Grad Norm: {avg_grad_norm:.4f}, "
                  f"KL: {avg_kl:.4f}, Clip Frac: {avg_clip_fraction:.2%}, Epochs: {num_updates}")

        return avg_loss, avg_grad_norm

    # --------------------
    # Training loop
    # --------------------
    def train(
        self,
        num_steps: int,
        num_circuits_per_step: int = 1,
        eval_interval: int = 50,
        save_interval: int = 100,
    ):
        print("=" * 60)
        print("Starting GRPO Training")
        print("=" * 60)
        print(f"Total steps: {num_steps}")
        print(f"Designs per circuit: {self.num_designs_per_circuit}")
        print(f"Circuits per step: {num_circuits_per_step}")
        print(f"Total designs per step: {self.num_designs_per_circuit * num_circuits_per_step}")
        print("=" * 60)

        if self.pvt_outer_loop_enabled:
            schedule_summary = self.configure_adaptive_pvt_schedule(num_steps)
            print("Adaptive PVT Schedule:")
            for key, value in schedule_summary.items():
                print(f"  {key}: {value}")
            print("=" * 60)

        if not hasattr(self, "grad_norm_history"):
            self.grad_norm_history: List[float] = []
        if not hasattr(self, "raw_reward_worst_history"):
            self.raw_reward_worst_history: List[float] = []
        if not hasattr(self, "group_reward_min_history"):
            self.group_reward_min_history: List[float] = []
        if not hasattr(self, "group_reward_max_history"):
            self.group_reward_max_history: List[float] = []

        for step in range(1, num_steps + 1):
            print(f"\n{'='*60}")
            print(f"Step {step}/{num_steps}")
            print(f"{'='*60}")

            # rollout
            episodes = self.rollout(num_circuits=num_circuits_per_step, verbose=True, step=step)
            verified_pvt_episodes = list(getattr(self, "_last_verified_pvt_episodes", []) or [])

            # group-relative advantages + group mean / weights
            episodes = self.compute_group_relative_advantages(episodes)

            # policy update
            loss, grad_norm_value = self.update_policy(episodes, verbose=True)

            # update global best reward + full performance (GLOBAL)
            self._maybe_update_best(episodes, step)

            # VAE handling
            if self.use_vae:
                if self.pvt_outer_loop_enabled:
                    self._maybe_train_pvt_proxy_model(step)
                elif self._vae_training_active and not self._vae_offline_trained:
                    sample_count = len(self.vae_dataset) if isinstance(self.vae_dataset, list) else 0
                    print(
                        f"  VAE collecting offline dataset (samples: {sample_count}/{self.vae_min_samples}, "
                        f"reward > {self.vae_reward_threshold})"
                    )
                    if sample_count >= self.vae_min_samples:
                        self._run_vae_offline_training(sample_count)
                elif self._vae_offline_trained:
                    self._vae_predict_step(step, episodes)

            
            rewards_step = np.array([ep.reward for ep in episodes], dtype=float)
            if not np.all(np.isfinite(rewards_step)):
                bad_mask = ~np.isfinite(rewards_step)
                print(f"[WARN] Non-finite rewards in this step stats, indices={np.where(bad_mask)[0]}")
                mean_reward = float(np.nanmean(rewards_step))
                std_reward = float(np.nanstd(rewards_step))
            else:
                mean_reward = float(np.mean(rewards_step))
                std_reward = float(np.std(rewards_step))

            finite_mask = np.isfinite(rewards_step)
            if finite_mask.any():
                success_rate = float(np.mean(rewards_step[finite_mask] > 0))
            else:
                success_rate = 0.0

            advs = np.array([ep.advantage for ep in episodes], dtype=float)
            if not np.all(np.isfinite(advs)):
                print("[WARN] Non-finite advantages in this step stats.")
                adv_min = float(np.nanmin(advs))
                adv_max = float(np.nanmax(advs))
                mean_advantage = float(np.nanmean(advs))
            else:
                adv_min = float(advs.min())
                adv_max = float(advs.max())
                mean_advantage = float(advs.mean())

            objective_means: Dict[str, float] = {}
            for objective_key in self.objective_keys:
                objective_values = np.array(
                    [ep.objective_rewards.get(objective_key, np.nan) for ep in episodes],
                    dtype=float,
                )
                if np.all(~np.isfinite(objective_values)):
                    objective_means[objective_key] = float("nan")
                else:
                    objective_means[objective_key] = float(np.nanmean(objective_values))
            objective_plot_means: Dict[str, float] = {
                "constraint_reward": objective_means.get("constraint_reward", float("nan"))
            }
            for history_key, perf_key in (
                ("FOML", "FOML"),
                ("FOMS", "FOMS"),
                ("Active Area", "Active Area"),
            ):
                perf_values = np.array(
                    [(ep.performance or {}).get(perf_key, np.nan) for ep in episodes],
                    dtype=float,
                )
                if np.all(~np.isfinite(perf_values)):
                    objective_plot_means[history_key] = float("nan")
                else:
                    objective_plot_means[history_key] = float(np.nanmean(perf_values))
            pm_values = np.array([ep.pm_violation for ep in episodes], dtype=float)
            if np.all(~np.isfinite(pm_values)):
                mean_pm_violation = float("nan")
            else:
                mean_pm_violation = float(np.nanmean(pm_values))
            pm_feasible_rate = float(np.mean([1.0 if ep.pm_feasible else 0.0 for ep in episodes])) if episodes else 0.0

            self.per_step_design_rewards.append(rewards_step.tolist())
            verified_rewards_step = np.array([ep.reward for ep in verified_pvt_episodes], dtype=float)
            self.per_step_verified_pvt_rewards.append(verified_rewards_step.tolist())

            # group mean / weight history
            group_mean_rewards = getattr(self, "_last_group_mean_rewards", {})
            group_weights = getattr(self, "_last_group_weights", {})
            if isinstance(group_mean_rewards, dict):
                self.group_mean_reward_history.append(group_mean_rewards.copy())
            else:
                self.group_mean_reward_history.append({})
            if isinstance(group_weights, dict):
                self.group_weight_history.append(group_weights.copy())
            else:
                self.group_weight_history.append({})
            group_objective_means = getattr(self, "_last_group_objective_means", {})
            if isinstance(group_objective_means, dict):
                self.group_mean_objective_history.append(copy.deepcopy(group_objective_means))
            else:
                self.group_mean_objective_history.append({})

            tt_rewards_step: List[float] = []
            for ep in episodes:
                try:
                    r_tt = ((ep.performance or {}).get("extra_corner") or {}).get("reward", None)
                except Exception:
                    r_tt = None
                if r_tt is not None and np.isfinite(r_tt):
                    tt_rewards_step.append(float(r_tt))
            self.per_step_extra_corner_rewards.append(tt_rewards_step)

            self.advantage_min_history.append(adv_min)
            self.advantage_max_history.append(adv_max)
            self.pm_violation_history.append(mean_pm_violation)
            self.pm_feasible_rate_history.append(pm_feasible_rate)
            self.pvt_verified_count_history.append(int(self._last_pvt_verified_count))
            self.pvt_verified_archive_size_history.append(int(len(self.verified_pvt_pareto_records)))
            self.pvt_phase_history.append(self._pvt_phase(step))
            self.covariance_condition_history.append(
                float(self._covariance_condition_number)
                if self._covariance_condition_number is not None
                else float("nan")
            )

            if verified_pvt_episodes:
                if np.all(~np.isfinite(verified_rewards_step)):
                    mean_verified_reward = float("nan")
                else:
                    mean_verified_reward = float(np.nanmean(verified_rewards_step))
                verified_pm_values = np.array([ep.pm_violation for ep in verified_pvt_episodes], dtype=float)
                if np.all(~np.isfinite(verified_pm_values)):
                    mean_verified_pm_violation = float("nan")
                else:
                    mean_verified_pm_violation = float(np.nanmean(verified_pm_values))
                verified_pm_feasible_rate = float(
                    np.mean([1.0 if ep.pm_feasible else 0.0 for ep in verified_pvt_episodes])
                )
            else:
                mean_verified_reward = float("nan")
                mean_verified_pm_violation = float("nan")
                verified_pm_feasible_rate = float("nan")

            self.verified_pvt_reward_history.append(mean_verified_reward)
            self.verified_pvt_pm_violation_history.append(mean_verified_pm_violation)
            self.verified_pvt_pm_feasible_rate_history.append(verified_pm_feasible_rate)
            for objective_key in self.objective_keys:
                if verified_pvt_episodes:
                    objective_values = np.array(
                        [ep.objective_rewards.get(objective_key, np.nan) for ep in verified_pvt_episodes],
                        dtype=float,
                    )
                    objective_mean = float(np.nanmean(objective_values)) if not np.all(~np.isfinite(objective_values)) else float("nan")
                else:
                    objective_mean = float("nan")
                self.verified_pvt_objective_history.setdefault(objective_key, []).append(objective_mean)

            for history_key, perf_key in (
                ("constraint_reward", None),
                ("FOML", "FOML"),
                ("FOMS", "FOMS"),
                ("Active Area", "Active Area"),
            ):
                if not verified_pvt_episodes:
                    history_mean = float("nan")
                elif perf_key is None:
                    history_mean = float(
                        np.nanmean(
                            np.asarray(
                                [ep.objective_rewards.get("constraint_reward", np.nan) for ep in verified_pvt_episodes],
                                dtype=float,
                            )
                        )
                    )
                else:
                    perf_values = np.array(
                        [(ep.performance or {}).get(perf_key, np.nan) for ep in verified_pvt_episodes],
                        dtype=float,
                    )
                    history_mean = float(np.nanmean(perf_values)) if not np.all(~np.isfinite(perf_values)) else float("nan")
                self.verified_pvt_objective_plot_history.setdefault(history_key, []).append(history_mean)

            raw_rewards_step: List[float] = []
            for ep in episodes:
                rr = ep.performance.get("raw_reward", None)
                if rr is not None:
                    raw_rewards_step.append(float(rr))
            if raw_rewards_step:
                rr_arr = np.array(raw_rewards_step, dtype=float)
                if not np.all(np.isfinite(rr_arr)):
                    worst_raw = float(np.nanmin(rr_arr))
                else:
                    worst_raw = float(rr_arr.min())
            else:
                worst_raw = float("nan")
            self.raw_reward_worst_history.append(worst_raw)

            if not np.all(np.isfinite(rewards_step)):
                self.group_reward_min_history.append(float(np.nanmin(rewards_step)))
                self.group_reward_max_history.append(float(np.nanmax(rewards_step)))
            else:
                self.group_reward_min_history.append(float(np.min(rewards_step)))
                self.group_reward_max_history.append(float(np.max(rewards_step)))

            self.grad_norm_history.append(grad_norm_value)

            self.reward_history.append(mean_reward)
            self.loss_history.append(loss)
            self.success_rate_history.append(success_rate)
            for objective_key, objective_mean in objective_means.items():
                self.objective_history.setdefault(objective_key, []).append(objective_mean)
            for history_key, objective_mean in objective_plot_means.items():
                self.objective_plot_history.setdefault(history_key, []).append(objective_mean)
            self.total_steps += 1

            
            try:
                block_lines: List[str] = []
                block_lines.append(f"===== Step {step} (designs={len(episodes)}) =====")
                for ep in episodes:
                    perf = ep.performance or {}
                    raw_r = perf.get("raw_reward", None)
                    if raw_r is None:
                        raw_r = perf.get("reward", None)
                    real_action = perf.get("real_action", None)
                    if real_action is None:
                        try:
                            from utils import ActionNormalizer

                            real_action = ActionNormalizer(
                                self.env.action_space_low, self.env.action_space_high
                            ).action(np.array(ep.action, copy=True)).tolist()
                        except Exception:
                            real_action = None

                    block_lines.append(
                        f"-- design_idx={ep.design_idx} worst_reward={raw_r if raw_r is not None else 'N/A'}"
                    )
                    sim_t = perf.get("pvt_sim_time_sec", None)
                    if sim_t is not None:
                        block_lines.append(f"sim_time_sec: {sim_t}")
                    if real_action is not None:
                        block_lines.append(f"real_action: {real_action}")

                    metrics = get_reporting_metrics(perf)
                    scores = get_reporting_scores(perf)

                    if metrics:
                        block_lines.append("metrics:")
                        pm_key = "phase_margin (deg)"
                        if pm_key in metrics:
                            block_lines.append(f"  {pm_key}: {metrics[pm_key]}")
                        for k in sorted(metrics.keys()):
                            if k == pm_key:
                                continue
                            block_lines.append(f"  {k}: {metrics[k]}")
                    if scores:
                        block_lines.append("scores:")
                        for k in sorted(scores.keys()):
                            block_lines.append(f"  {k}: {scores[k]}")
                    block_lines.append("=" * 80)

                block_text = "\n".join(block_lines) + "\n"
                if self._details_buffer_start_step is None:
                    self._details_buffer_start_step = step
                self._details_buffer.append(block_text)

                if step % 10 == 0:
                    os.makedirs(self._logs_dir, exist_ok=True)
                    start_s = self._details_buffer_start_step
                    end_s = step
                    out_path = os.path.join(self._logs_dir, f"worst_raw_details_steps_{start_s}-{end_s}.txt")
                    with open(out_path, "w", encoding="utf-8") as f:
                        f.write("".join(self._details_buffer))
                    self._details_buffer = []
                    self._details_buffer_start_step = None
            except Exception:
                pass

            # step summary
            print(f"\n{'='*60}")
            print(f"Step {step} Summary:")
            print(f"  Mean Training Reward: {mean_reward:.4f} +/- {std_reward:.4f}")
            print(f"  Mean Advantage: {mean_advantage:.4f} (min={adv_min:.4f}, max={adv_max:.4f})")
            gm = getattr(self, "_last_group_mean_rewards", {})
            if isinstance(gm, dict) and gm:
                print("  Group mean training rewards (closer to 0 is better):")
                for spec, val in gm.items():
                    print(f"    {spec}: {val:.4f}")
            print(f"  Mean PM Violation: {mean_pm_violation:.4f}")
            print(f"  PM Feasible Rate: {pm_feasible_rate:.2%}")
            if self.covariance_whitening_enabled:
                whitening_state = "frozen" if self._covariance_whitener_ready else "warmup"
                freeze_info = (
                    f", freeze_step={self._covariance_freeze_step}"
                    if self._covariance_freeze_step is not None
                    else ""
                )
                cond_info = (
                    f", cond={self._covariance_condition_number:.4f}"
                    if self._covariance_condition_number is not None and np.isfinite(self._covariance_condition_number)
                    else ""
                )
                print(f"  Covariance Whitening: {whitening_state}{freeze_info}{cond_info}")
            for objective_key, objective_mean in objective_means.items():
                print(f"  Mean Objective {objective_key}: {objective_mean:.4f}")
            print(f"  Loss: {loss:.4f}")
            best_summary = getattr(self, "_latest_best_update_summary", {}) or {}
            print("  Best Monitor (logging only, does not affect policy update):")
            print(
                f"    Pareto archive size: {len(self.best_pareto_records)}"
                f"{' [updated]' if best_summary.get('pareto_archive_updated') else ''}"
            )
            objective_updates = list(best_summary.get("objective_updates", []) or [])
            if objective_updates:
                print(f"    Objective records updated this step: {', '.join(objective_updates)}")
            else:
                print("    Objective records updated this step: none")
            if self.pvt_outer_loop_enabled:
                dataset_size = len(self.vae_dataset) if isinstance(self.vae_dataset, list) else 0
                proxy_state = self._pvt_phase(step)
                print(
                    f"    PVT proxy: state={proxy_state}, "
                    f"proxy_predictions={self._last_pvt_proxy_count}, "
                    f"verified_this_step={self._last_pvt_verified_count}, "
                    f"verified_archive={len(self.verified_pvt_pareto_records)}, "
                    f"vae_dataset={dataset_size}"
                )
                if np.isfinite(mean_verified_reward):
                    print(
                        f"    Verified-PVT mean training reward: {mean_verified_reward:.4f}, "
                        f"pm_feasible_rate={verified_pm_feasible_rate:.2%}"
                    )
            objective_monitor_specs = [
                ("constraint_reward", "constraint_score_best"),
                ("FOML", "FOML_best"),
                ("FOMS", "FOMS_best"),
                ("Active Area", "Active_Area_best(min)"),
            ]
            for metric_key, label in objective_monitor_specs:
                record = self.best_objective_records.get(metric_key)
                if not record:
                    continue
                print(
                    f"    {label}: value={record.get('selection_value')} "
                    f"(step {record.get('found_at_step')}, design {record.get('design_idx')})"
                )
            print(f"{'='*60}")

            # delegating plotting & logging to draw.py
            draw_log_step(self, step, episodes, loss, grad_norm_value, mean_reward, std_reward, mean_advantage)

            if step % save_interval == 0:
                self._save_model(step)

            if step % 10 == 0:
                # save multi-objective monitoring snapshots every 10 steps
                self._save_best_record_txt(step)
                self._plot_progress()

        print("\n" + "=" * 60)
        print("Training Complete!")
        print("=" * 60)

        print("\nSaving final checkpoint...")
        self._save_model(num_steps)
        print("[OK] Training complete. All logs saved.")

        draw_write_vae_predictions(self)

        if self.wandb is not None:
            try:
                self.wandb.finish()
            except Exception:
                pass

    # --------------------
    # VAE helpers
    # --------------------
    def _snapshot_vae_state(self) -> Optional[Dict[str, Any]]:
        if not self.use_vae or self.vae is None:
            return None
        return {
            "net_state_dict": copy.deepcopy(self.vae.net.state_dict()),
            "optimizer_state_dict": copy.deepcopy(self.vae.optimizer.state_dict()),
            "std_c": copy.deepcopy(self.vae.std_c.state_dict()),
            "std_y": copy.deepcopy(self.vae.std_y.state_dict()),
            "train_losses": list(getattr(self.vae, "train_losses", [])),
            "training_step": int(getattr(self.vae, "training_step", 0)),
        }

    def _restore_vae_state(self, snapshot: Optional[Dict[str, Any]]) -> None:
        if not snapshot or not self.use_vae or self.vae is None:
            return
        self.vae.net.load_state_dict(snapshot.get("net_state_dict", {}))
        if snapshot.get("optimizer_state_dict"):
            self.vae.optimizer.load_state_dict(snapshot.get("optimizer_state_dict", {}))
        try:
            self.vae.std_c.load_state_dict(snapshot.get("std_c", {}))
            self.vae.std_y.load_state_dict(snapshot.get("std_y", {}))
        except Exception:
            self.vae.std_c.reset()
            self.vae.std_y.reset()
        self.vae.train_losses = list(snapshot.get("train_losses", []))
        self.vae.training_step = int(snapshot.get("training_step", 0))

    def _train_vae_dataset_with_patience(
        self,
        epochs: int,
        patience: int,
        stage_label: str,
    ) -> List[float]:
        if not self.use_vae or self.vae is None:
            return []
        sample_count = len(self.vae_dataset) if isinstance(self.vae_dataset, list) else 0
        if sample_count <= 0:
            return []

        batch_size = min(64, sample_count)
        best_snapshot = self._snapshot_vae_state()
        best_loss = float("inf")
        all_losses: List[float] = []
        stale_epochs = 0
        min_delta = 1e-4

        print(
            f"  VAE {stage_label}: dataset={sample_count}, "
            f"epochs={epochs}, patience={patience}, batch_size={batch_size}"
        )
        for epoch_idx in range(max(1, int(epochs))):
            epoch_losses = self.vae.offline_train(
                self.vae_dataset,
                epochs=1,
                batch_size=batch_size,
                shuffle=True,
            )
            if not epoch_losses:
                break

            epoch_loss = float(epoch_losses[-1])
            all_losses.append(epoch_loss)
            self.vae_total_history.append(epoch_loss)
            self.vae_recon_history.append(epoch_loss)
            self.vae_loss_history.append(epoch_loss)

            if epoch_loss < best_loss - min_delta:
                best_loss = epoch_loss
                best_snapshot = self._snapshot_vae_state()
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= patience:
                    print(
                        f"  VAE {stage_label}: early stop at epoch {epoch_idx + 1}/{epochs}, "
                        f"best_loss={best_loss:.4f}"
                    )
                    break

        self._restore_vae_state(best_snapshot)
        if all_losses:
            print(
                f"  VAE {stage_label} complete. "
                f"best_loss={best_loss:.4f}, last_loss={all_losses[-1]:.4f}"
            )
        else:
            print(f"  VAE {stage_label} produced no loss records")
        return all_losses

    def _run_vae_offline_training(self, sample_count: int) -> None:
        if not self.use_vae or self.vae is None:
            return

        if sample_count <= 0:
            print("  VAE offline training skipped: no samples collected")
            self._vae_training_active = False
            self._kan_training_active = self._vae_training_active
            return

        try:
            self._vae_offline_epochs = int(self.pvt_offline_epochs)
            losses = self._train_vae_dataset_with_patience(
                epochs=self.pvt_offline_epochs,
                patience=self.pvt_offline_patience,
                stage_label="offline warmup training",
            )
        except Exception as exc:
            print(f"[VAE offline] training failed: {exc}")
            self._vae_training_active = False
            self._kan_training_active = self._vae_training_active
            return

        self._vae_offline_trained = True
        self._vae_training_active = False
        self._kan_offline_trained = self._vae_offline_trained
        self._kan_training_active = self._vae_training_active
        self._pvt_proxy_ready_step = int(self.total_steps)

    def _vae_predict_step(self, step: int, episodes: List[Episode]) -> None:
        """
        After offline training, run VAE prediction per episode and record
        worst-case reward prediction vs actual.
        """
        if not self.use_vae or self.vae is None or not episodes:
            return

        print(f"  VAE inference-only mode (step {step}), predicting each action in the step")

        per_ep_preds: List[float] = []
        per_ep_actuals: List[float] = []

        for ep in episodes:
            try:
                perf_info = ep.performance or {}
                extra = perf_info.get("extra_corner")
                if not isinstance(extra, dict):
                    continue
                perf_extra = extra.get("performance", {}) or {}
                tt_action = perf_extra.get("real_action")
                tt_reward = extra.get("reward")
                if tt_action is None:
                    try:
                        from utils import ActionNormalizer

                        normalizer = ActionNormalizer(self.env.action_space_low, self.env.action_space_high)
                        tt_action = normalizer.action(np.array(ep.action, copy=True))
                    except Exception:
                        tt_action = None
                if tt_action is None or tt_reward is None:
                    continue

                tt_objectives = build_target_objectives(
                    objectives=perf_extra.get("objective_rewards", {}) or {},
                    performance=perf_extra,
                    objective_keys=self.vae_objective_keys,
                )
                tt_constraint_reward = float(tt_objectives.get("constraint_reward", 0.0))
                actual_targets = build_target_objectives(
                    objectives=perf_info.get("worst_case_targets", {}) or perf_info.get("objective_rewards", {}) or {},
                    performance=perf_info,
                    objective_keys=self.vae_objective_keys,
                )
                condition = build_vae_condition(
                    tt_action=np.array(tt_action, dtype=np.float32),
                    tt_performance=perf_extra,
                    tt_objectives=tt_objectives,
                    objective_keys=self.vae_objective_keys,
                )

                pred_mean_dict, pred_std_dict = self.vae.predict_pvt_objectives(condition, num_samples=50)
                pred_mean = np.asarray(
                    [float(pred_mean_dict.get(key, 0.0)) for key in self.vae_objective_keys],
                    dtype=np.float32,
                )
                pred_std = np.asarray(
                    [float(pred_std_dict.get(key, 0.0)) for key in self.vae_objective_keys],
                    dtype=np.float32,
                )
                pred_reward = float(pred_mean_dict.get("constraint_reward", pred_mean[0]))
                actual_reward = float(actual_targets.get("constraint_reward", 0.0))
                actual_objectives = [
                    float(actual_targets.get(key, 0.0))
                    for key in self.vae_objective_keys
                ]

                self._vae_pred_records.append((step, tt_constraint_reward, actual_reward, pred_reward))
                self._vae_multi_pred_records.append(
                    {
                        "step": int(step),
                        "tt_reward": tt_constraint_reward,
                        "pred_mean": pred_mean.astype(np.float32).tolist(),
                        "pred_std": pred_std.astype(np.float32).tolist(),
                        "actual": [float(x) for x in actual_objectives],
                    }
                )

                perf_info["vae_pred_pvt_worst_reward_mean"] = pred_reward
                perf_info["vae_pred_pvt_worst_reward_std"] = float(pred_std[0]) if len(pred_std) > 0 else 0.0
                perf_info["vae_pred_constraint_reward_mean"] = pred_reward
                perf_info["vae_pred_constraint_reward_std"] = float(pred_std[0]) if len(pred_std) > 0 else 0.0
                perf_info["vae_pred_multi_objective_mean"] = pred_mean.tolist()
                perf_info["vae_pred_multi_objective_std"] = pred_std.tolist()
                perf_info["vae_pred_objectives_mean"] = pred_mean_dict.copy()
                perf_info["vae_pred_objectives_std"] = pred_std_dict.copy()
                perf_info["vae_actual_multi_objective"] = actual_objectives
                perf_info["vae_actual_objectives"] = actual_targets.copy()
                perf_info["kan_pred_pvt_worst_reward_mean"] = pred_reward
                perf_info["kan_pred_pvt_worst_reward_std"] = float(pred_std[0]) if len(pred_std) > 0 else 0.0
                perf_info["kan_pred_multi_objective_mean"] = pred_mean.tolist()
                perf_info["kan_pred_multi_objective_std"] = pred_std.tolist()
                ep.performance = perf_info

                per_ep_preds.append(pred_reward)
                per_ep_actuals.append(actual_reward)
            except Exception as exc:
                print(f"[VAE inference/episode] error: {exc}")

        if per_ep_preds and per_ep_actuals:
            worst_idx = min(range(len(per_ep_actuals)), key=lambda i: float(per_ep_actuals[i]))
            actual_worst_reward = per_ep_actuals[worst_idx]
            pred_worst_reward = per_ep_preds[worst_idx]

            self.vae_actual_worst_history.append(float(actual_worst_reward))
            self.vae_pred_worst_history.append(float(pred_worst_reward))

    def _run_kan_offline_training(self, sample_count: int) -> None:
        self._run_vae_offline_training(sample_count)

    def _kan_predict_step(self, step: int, episodes: List[Episode]) -> None:
        self._vae_predict_step(step, episodes)


    def _save_model(self, step: int):
        """
        Save model checkpoint and training history.
        """
        import pickle
        from datetime import datetime

        os.makedirs(self._training_saves_dir, exist_ok=True)

        checkpoint = {
            "step": step,
            "policy_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "reward_history": self.reward_history,
            "loss_history": self.loss_history,
            "success_rate_history": self.success_rate_history,
            "total_steps": self.total_steps,
            "use_vae": self.use_vae,
            "vae_diagnostics_enabled": self.vae_diagnostics_enabled,
            "vae_save_per_objective_plots": self.vae_save_per_objective_plots,
            "use_kan": self.use_kan,
            "advantage_mode": self.advantage_mode,
            "objective_normalization_mode": self.objective_normalization_mode,
            "utility_temperature": self.utility_temperature,
            "utility_baseline_temperature": self.utility_baseline_temperature,
            "utility_reference_alpha": self.utility_reference_alpha,
            "utility_reference_update_interval": self.utility_reference_update_interval,
            "utility_reference_quantile_high": self.utility_reference_quantile_high,
            "utility_reference_quantile_low": self.utility_reference_quantile_low,
            "utility_reference_clip_delta": self.utility_reference_clip_delta,
            "best_reward": self.best_reward,
            "best_step": self.best_step,
            "best_record": self.best_record,
            "best_performance_reward": self.best_performance_reward,
            "best_performance_step": self.best_performance_step,
            "best_performance_record": self.best_performance_record,
            "best_pareto_records": self.best_pareto_records,
            "verified_pvt_pareto_records": self.verified_pvt_pareto_records,
            "best_objective_records": self.best_objective_records,
            "objective_plot_history": self.objective_plot_history,
            "verified_pvt_reward_history": self.verified_pvt_reward_history,
            "verified_pvt_pm_violation_history": self.verified_pvt_pm_violation_history,
            "verified_pvt_pm_feasible_rate_history": self.verified_pvt_pm_feasible_rate_history,
            "verified_pvt_objective_history": self.verified_pvt_objective_history,
            "verified_pvt_objective_plot_history": self.verified_pvt_objective_plot_history,
            "pvt_outer_loop_enabled": self.pvt_outer_loop_enabled,
            "pvt_verified_count_history": self.pvt_verified_count_history,
            "pvt_verified_archive_size_history": self.pvt_verified_archive_size_history,
            "pvt_phase_history": self.pvt_phase_history,
            "adaptive_pvt_schedule_summary": dict(getattr(self, "_adaptive_pvt_schedule_summary", {})),
            "covariance_whitening_state": {
                "ready": self._covariance_whitener_ready,
                "freeze_step": self._covariance_freeze_step,
                "aggregation_vector": None
                if self._covariance_aggregation_vector is None
                else self._covariance_aggregation_vector.tolist(),
                "condition_number": self._covariance_condition_number,
            },
        }
        if self.use_vae and self.vae is not None:
            checkpoint["vae_state_dict"] = self.vae.state_dict()
            checkpoint["vae_optimizer_state_dict"] = self.vae.optimizer.state_dict()
            checkpoint["vae_loss_history"] = getattr(
                self, "vae_loss_history", getattr(self, "vae_total_history", [])
            )
            checkpoint["kan_state_dict"] = self.vae.state_dict()
            checkpoint["kan_optimizer_state_dict"] = self.vae.optimizer.state_dict()
            checkpoint["kan_loss_history"] = checkpoint["vae_loss_history"]

        save_path = f"{self._training_saves_dir}/grpo_policy_step_{step}.pth"
        torch.save(checkpoint, save_path)
        print(f"\n[OK] Model saved to {save_path}")

        history_path = f"{self._training_saves_dir}/training_history_step_{step}.pkl"
        history_data = {
            "step": step,
            "reward_history": self.reward_history,
            "loss_history": self.loss_history,
            "success_rate_history": self.success_rate_history,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "config": {
                "num_designs_per_circuit": self.num_designs_per_circuit,
                "learning_rate": self.lr,
                "advantage_mode": self.advantage_mode,
                "objective_normalization_mode": self.objective_normalization_mode,
                "objective_keys": self.objective_keys,
                "vae_objective_keys": list(self.vae_objective_keys),
                "vae_diagnostics_enabled": self.vae_diagnostics_enabled,
                "vae_save_per_objective_plots": self.vae_save_per_objective_plots,
                "pvt_outer_loop_enabled": self.pvt_outer_loop_enabled,
                "pvt_verify_topk_per_step": self.pvt_verify_topk_per_step,
                "pvt_verify_uncertain_per_step": self.pvt_verify_uncertain_per_step,
                "pvt_warmup_steps": self.pvt_warmup_steps,
                "pvt_warmup_verify_topk_per_step": self.pvt_warmup_verify_topk_per_step,
                "pvt_warmup_diverse_per_step": self.pvt_warmup_diverse_per_step,
                "pvt_offline_epochs": self.pvt_offline_epochs,
                "pvt_offline_patience": self.pvt_offline_patience,
                "pvt_proxy_finetune_interval": self.pvt_proxy_finetune_interval,
                "pvt_proxy_finetune_epochs": self.pvt_proxy_finetune_epochs,
                "pvt_proxy_finetune_patience": self.pvt_proxy_finetune_patience,
                "adaptive_pvt_schedule_summary": dict(getattr(self, "_adaptive_pvt_schedule_summary", {})),
                "pvt_proxy_beta": self.pvt_proxy_beta,
                "pvt_proxy_prediction_samples": self.pvt_proxy_prediction_samples,
                "pvt_proxy_online_update_interval": self.pvt_proxy_online_update_interval,
                "pvt_proxy_online_updates_per_step": self.pvt_proxy_online_updates_per_step,
                "pvt_proxy_online_batch_size": self.pvt_proxy_online_batch_size,
                "pvt_verified_archive_capacity": self.pvt_verified_archive_capacity,
                "covariance_whitening_enabled": self.covariance_whitening_enabled,
                "covariance_whitening_warmup_steps": self.covariance_whitening_warmup_steps,
                "covariance_whitening_min_samples": self.covariance_whitening_min_samples,
                "covariance_whitening_shrinkage": self.covariance_whitening_shrinkage,
                "covariance_whitening_buffer_capacity": self.covariance_whitening_buffer_capacity,
                "utility_temperature": self.utility_temperature,
                "utility_baseline_temperature": self.utility_baseline_temperature,
                "utility_reference_alpha": self.utility_reference_alpha,
                "utility_reference_update_interval": self.utility_reference_update_interval,
                "utility_reference_quantile_high": self.utility_reference_quantile_high,
                "utility_reference_quantile_low": self.utility_reference_quantile_low,
                "utility_reference_clip_delta": self.utility_reference_clip_delta,
            },
            "best_reward": self.best_reward,
            "best_step": self.best_step,
            "best_record": self.best_record,
            "best_performance_reward": self.best_performance_reward,
            "best_performance_step": self.best_performance_step,
            "best_performance_record": self.best_performance_record,
            "best_pareto_records": self.best_pareto_records,
            "verified_pvt_pareto_records": self.verified_pvt_pareto_records,
            "best_objective_records": self.best_objective_records,
            "objective_history": self.objective_history,
            "objective_plot_history": self.objective_plot_history,
            "verified_pvt_reward_history": self.verified_pvt_reward_history,
            "verified_pvt_pm_violation_history": self.verified_pvt_pm_violation_history,
            "verified_pvt_pm_feasible_rate_history": self.verified_pvt_pm_feasible_rate_history,
            "verified_pvt_objective_history": self.verified_pvt_objective_history,
            "verified_pvt_objective_plot_history": self.verified_pvt_objective_plot_history,
            "pm_violation_history": self.pm_violation_history,
            "pm_feasible_rate_history": self.pm_feasible_rate_history,
            "pvt_verified_count_history": self.pvt_verified_count_history,
            "pvt_verified_archive_size_history": self.pvt_verified_archive_size_history,
            "pvt_phase_history": self.pvt_phase_history,
            "adaptive_pvt_schedule_summary": dict(getattr(self, "_adaptive_pvt_schedule_summary", {})),
            "covariance_condition_history": self.covariance_condition_history,
            "covariance_whitening_state": {
                "ready": self._covariance_whitener_ready,
                "freeze_step": self._covariance_freeze_step,
                "aggregation_vector": None
                if self._covariance_aggregation_vector is None
                else self._covariance_aggregation_vector.tolist(),
                "condition_number": self._covariance_condition_number,
            },
        }
        with open(history_path, "wb") as f:
            pickle.dump(history_data, f)
        print(f"[OK] Training history saved to {history_path}")

        draw_save_plot_data(self, snapshot_step=step)
        print(f"[OK] Plot data exported to {os.path.join(self._training_saves_dir, 'plot_data')}")
        self._save_best_record_txt(step, log_dir=self._logs_dir)
        print(f"[OK] Best-record documents exported to {self._logs_dir}")

        log_path = f"{self._training_saves_dir}/training_log.txt"
        with open(log_path, "a") as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"Checkpoint at Step {step}\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"{'='*80}\n")
            f.write(f"Mean Training Reward: {self.reward_history[-1]:.4f}\n")
            f.write(f"Mean Loss: {self.loss_history[-1]:.4f}\n")
            if len(self.success_rate_history) > 0:
                f.write(f"Success Rate: {self.success_rate_history[-1]:.2%}\n")
            else:
                f.write("Success Rate: N/A\n")
            f.write(f"Total Steps: {self.total_steps}\n")
            if len(self.reward_history) > 1:
                f.write(f"Best Training Reward (mean history): {max(self.reward_history):.4f}\n")
            f.write(f"Pareto Archive Size: {len(self.best_pareto_records)}\n")
            if self.pvt_outer_loop_enabled:
                f.write(f"Verified PVT Pareto Archive Size: {len(self.verified_pvt_pareto_records)}\n")
            f.write(f"{'='*80}\n")
        print(f"[OK] Log appended to {log_path}")

    def _plot_progress(self):
        draw_plot_progress(self)


# --------------------
# Standalone helpers
# --------------------
def compute_group_advantages(episodes: List[Episode]) -> List[Episode]:
    """
    Standalone function to compute group-relative advantages.
    """
    agent = GRPOAgent.__new__(GRPOAgent)
    # compute_group_relative_advantages will set _last_group_* on the fly
    return agent.compute_group_relative_advantages(episodes)


def rollout_designs(
    policy_network,
    env,
    num_designs: int,
    device: str = "cpu",
) -> List[Episode]:
    """
    Standalone rollout for evaluation (no GRPOAgent needed).
    """
    episodes: List[Episode] = []
    policy_network.eval()

    with torch.no_grad():
        for i in range(num_designs):
            state, _ = env.reset()
            state = np.asarray(state, dtype=np.float32)
            if not np.all(np.isfinite(state)):
                print("[WARN] Non-finite state from env.reset in rollout_designs, sanitising to 0.")
                state = np.nan_to_num(state, nan=0.0, posinf=0.0, neginf=0.0)
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            action = policy_network(state_tensor)
            if isinstance(action, tuple):
                action = action[0]
            action = action.cpu().numpy()[0]
            action = np.clip(action, -1.0, 1.0)

            _, reward, _, _, info = env.step(action)
            reward = float(reward)
            if not np.isfinite(reward):
                print(f"[WARN] Non-finite reward in rollout_designs, set to -1e6. reward={reward}")
                reward = -1e6

            ep = Episode(
                circuit_spec="eval",
                state=state,
                action=action,
                reward=reward,
                performance=info,
                design_idx=i,
            )
            episodes.append(ep)

    return episodes
