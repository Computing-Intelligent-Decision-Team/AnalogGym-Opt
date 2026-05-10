from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


VAE_CONDITION_PERFORMANCE_KEYS: Tuple[str, ...] = (
    "phase_margin (deg)",
    "dcgain",
    "PSRP",
    "PSRN",
    "PSRR",
    "cmrrdc",
    "TC",
    "setting_time",
    "FOML",
    "FOMS",
    "Active Area",
    "Power",
    "GBW",
    "sr",
)

RL_OBJECTIVE_KEYS: Tuple[str, ...] = (
    "constraint_reward",
    "FOML_score",
    "FOMS_score",
    "Active_Area_score",
)

VAE_OBJECTIVE_KEYS: Tuple[str, ...] = RL_OBJECTIVE_KEYS + ("PM_violation",)

PM_LOWER_BOUND = 45.0
PM_UPPER_BOUND = 90.0


# ============================================================

# ============================================================

@dataclass
class VAETrainingData:
    """Single training sample for the worst-corner VAE predictor."""
    condition: np.ndarray
    target_y: np.ndarray
    corner_label: int


class RunningStandardizer:
    """Online mean and variance estimator for tensor normalization."""

    def __init__(self, dim: int, device: str = "cpu", eps: float = 1e-6):
        self.dim = dim
        self.device = torch.device(device)
        self.eps = float(eps)
        self.reset()

    def reset(self) -> None:
        self.count = 0
        self.mean = torch.zeros(self.dim, device=self.device, dtype=torch.float32)
        self.M2 = torch.zeros(self.dim, device=self.device, dtype=torch.float32)

    def update(self, batch: torch.Tensor) -> None:
        if batch is None or batch.ndim != 2 or batch.size(1) != self.dim:
            return
        n = int(batch.size(0))
        if n == 0:
            return
        batch_mean = batch.mean(dim=0)
        batch_var = batch.var(dim=0, unbiased=True) if n > 1 else torch.zeros_like(batch_mean)
        total = self.count + n
        delta = batch_mean - self.mean
        new_mean = self.mean + delta * (n / float(total))
        M2_batch = batch_var * (n - 1)
        new_M2 = self.M2 + M2_batch + (delta * delta) * (self.count * n / float(total))
        self.mean = new_mean
        self.M2 = new_M2
        self.count = int(total)

    def std(self) -> torch.Tensor:
        if self.count <= 1:
            return torch.ones_like(self.mean)
        var = self.M2 / float(max(1, self.count - 1))
        return torch.sqrt(torch.clamp(var, min=self.eps))

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std()

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.std() + self.mean

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {
            "count": torch.tensor([self.count], dtype=torch.long, device=self.device),
            "mean": self.mean.detach().clone(),
            "M2": self.M2.detach().clone(),
        }

    def load_state_dict(self, state: Dict[str, torch.Tensor]) -> None:
        try:
            self.count = int(state.get("count", torch.tensor([0], device=self.device)).item())
            self.mean = state.get("mean", torch.zeros(self.dim, device=self.device))
            self.M2 = state.get("M2", torch.zeros(self.dim, device=self.device))
        except Exception:
            self.reset()


def _stack_training_data(
    batch: Sequence[VAETrainingData],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    conditions = np.stack([d.condition for d in batch], axis=0).astype(np.float32)
    targets = np.stack([d.target_y for d in batch], axis=0).astype(np.float32)
    c_tensor = torch.as_tensor(conditions, dtype=torch.float32, device=device)
    y_tensor = torch.as_tensor(targets, dtype=torch.float32, device=device)
    return c_tensor, y_tensor


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        value = float(value)
    except Exception:
        return float(default)
    if not np.isfinite(value):
        return float(default)
    return value


def compute_phase_margin_violation(
    phase_margin: Any,
    lower: float = PM_LOWER_BOUND,
    upper: float = PM_UPPER_BOUND,
) -> float:
    pm = _safe_float(phase_margin, default=float("nan"))
    if not np.isfinite(pm):
        return 1.0
    if lower < pm < upper:
        return 0.0
    if pm <= lower:
        return float(lower - pm) / max(abs(lower), 1e-8)
    return float(pm - upper) / max(abs(lower), 1e-8)


def extract_performance_vector(
    performance: Optional[Dict[str, Any]],
    performance_keys: Sequence[str] = VAE_CONDITION_PERFORMANCE_KEYS,
) -> np.ndarray:
    perf = performance or {}
    values = [_safe_float(perf.get(key, 0.0), default=0.0) for key in performance_keys]
    return np.asarray(values, dtype=np.float32)


def build_target_objectives(
    objectives: Optional[Dict[str, Any]] = None,
    performance: Optional[Dict[str, Any]] = None,
    objective_keys: Sequence[str] = VAE_OBJECTIVE_KEYS,
) -> Dict[str, float]:
    base = dict(objectives or {})
    perf = performance or {}

    if "PM_violation" in objective_keys and "PM_violation" not in base:
        base["PM_violation"] = compute_phase_margin_violation(perf.get("phase_margin (deg)", np.nan))

    return {
        str(key): (
            compute_phase_margin_violation(perf.get("phase_margin (deg)", np.nan))
            if key == "PM_violation"
            else _safe_float(base.get(key, perf.get(key, 0.0)), default=0.0)
        )
        for key in objective_keys
    }


def extract_objective_vector(
    objectives: Optional[Dict[str, Any]],
    objective_keys: Sequence[str] = VAE_OBJECTIVE_KEYS,
) -> np.ndarray:
    obj = objectives or {}
    values = [_safe_float(obj.get(key, 0.0), default=0.0) for key in objective_keys]
    return np.asarray(values, dtype=np.float32)


def objective_vector_to_dict(
    values: Sequence[float],
    objective_keys: Sequence[str] = VAE_OBJECTIVE_KEYS,
) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    return {
        str(key): _safe_float(arr[idx] if idx < arr.shape[0] else 0.0, default=0.0)
        for idx, key in enumerate(objective_keys)
    }


def build_vae_condition(
    tt_action: np.ndarray,
    tt_performance: Dict[str, Any],
    tt_objectives: Optional[Dict[str, Any]] = None,
    objective_keys: Sequence[str] = VAE_OBJECTIVE_KEYS,
    performance_keys: Sequence[str] = VAE_CONDITION_PERFORMANCE_KEYS,
    tt_reward: Optional[float] = None,
) -> np.ndarray:
    action_vec = np.asarray(tt_action, dtype=np.float32).reshape(-1)
    perf_vec = extract_performance_vector(tt_performance, performance_keys=performance_keys)

    if tt_objectives is not None:
        target_objectives = build_target_objectives(
            objectives=tt_objectives,
            performance=tt_performance,
            objective_keys=objective_keys,
        )
        objective_vec = extract_objective_vector(target_objectives, objective_keys=objective_keys)
        return np.concatenate([action_vec, perf_vec, objective_vec], axis=0).astype(np.float32)

    reward_vec = np.asarray([_safe_float(tt_reward, default=0.0)], dtype=np.float32)
    return np.concatenate([action_vec, perf_vec, reward_vec], axis=0).astype(np.float32)


def _extract_multi_objective_vector(
    performance: Optional[Dict],
    fallback_reward: float = 0.0,
) -> np.ndarray:
    performance = performance or {}
    reward_components = performance.get("reward_components") or {}
    reward_value = performance.get("reward", reward_components.get("reward", fallback_reward))
    foml_value = performance.get(
        "FOML_score",
        reward_components.get("FOML_score", performance.get("FOML", 0.0)),
    )
    foms_value = performance.get(
        "FOMS_score",
        reward_components.get("FOMS_score", performance.get("FOMS", 0.0)),
    )
    area_value = performance.get(
        "Active_Area_score",
        reward_components.get("Active_Area_score", performance.get("Active Area", 0.0)),
    )
    values = np.array([reward_value, foml_value, foms_value, area_value], dtype=np.float32)
    values[~np.isfinite(values)] = 0.0
    return values


# ============================================================

# ============================================================

def gaussian_log_prob(value: torch.Tensor,
                      mean: torch.Tensor,
                      logvar: torch.Tensor) -> torch.Tensor:
    """Return the diagonal-Gaussian log probability without the constant term."""
    var = logvar.exp()
    return -0.5 * (((value - mean) ** 2) / var + logvar).sum(dim=-1)


# ============================================================

# ============================================================

class PlanarFlow(nn.Module):
    """Planar normalizing-flow transform used by the conditional VAE."""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.u = nn.Parameter(torch.randn(1, dim) * 0.01)
        self.w = nn.Parameter(torch.randn(1, dim) * 0.01)
        self.b = nn.Parameter(torch.zeros(1))

    def _u_hat(self) -> torch.Tensor:
        wTu = torch.mm(self.w, self.u.t())        # [1,1]
        m = -1.0 + F.softplus(wTu)                # > -1
        w_norm_sq = torch.sum(self.w ** 2, dim=1, keepdim=True)  # [1,1]
        u_hat = self.u + (m - wTu) * self.w / (w_norm_sq + 1e-8)
        return u_hat  # [1, dim]

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply the flow and return the transformed latent plus log determinant."""
        u_hat = self._u_hat()             # [1, dim]
        linear = z @ self.w.t() + self.b  # [N,1]
        h = torch.tanh(linear)            # [N,1]
        h_prime = 1.0 - h ** 2            # [N,1]

        # z' = z + u_hat * h
        z_new = z + h @ u_hat             # [N,dim]

        # psi(z) = h'(w^T z + b) * w
        psi = h_prime * self.w            # [N,dim]

        # detJ = |1 + u_hat^T psi^T|
        det_jacobian = 1.0 + (psi @ u_hat.t())  # [N,1]
        log_det = torch.log(det_jacobian.squeeze(-1) + 1e-8)  # [N]

        return z_new, log_det


# ============================================================

# ============================================================

class FlowIWAEConditionalVAE(nn.Module):
    """Conditional VAE trained with an IWAE objective and planar flows."""
    def __init__(
        self,
        input_size: int,
        output_size: int,
        latent_dim: int = 8,
        hidden_dim: int = 64,
        n_flows: int = 4,
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.n_flows = n_flows

        # encoder: q0(z0 | x,y)
        enc_in_dim = input_size + output_size
        self.enc_fc1 = nn.Linear(enc_in_dim, hidden_dim)
        self.enc_mu = nn.Linear(hidden_dim, latent_dim)
        self.enc_logvar = nn.Linear(hidden_dim, latent_dim)

        # conditional prior: p(z|x)
        self.prior_fc1 = nn.Linear(input_size, hidden_dim)
        self.prior_mu = nn.Linear(hidden_dim, latent_dim)
        self.prior_logvar = nn.Linear(hidden_dim, latent_dim)

        # decoder: p(y|x,z)
        dec_in_dim = input_size + latent_dim
        self.dec_fc1 = nn.Linear(dec_in_dim, hidden_dim)
        self.dec_out = nn.Linear(hidden_dim, 2 * output_size)

        # flows
        self.flows = nn.ModuleList([PlanarFlow(latent_dim) for _ in range(n_flows)])

        self.act = nn.ReLU()

    # --- encoder / prior / decoder ---

    def encode(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        xy = torch.cat([x, y], dim=-1)
        h = self.act(self.enc_fc1(xy))
        mu_q = self.enc_mu(h)
        logvar_q = self.enc_logvar(h)
        return mu_q, logvar_q

    def prior(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.act(self.prior_fc1(x))
        mu_p = self.prior_mu(h)
        logvar_p = self.prior_logvar(h)
        return mu_p, logvar_p

    def decode(self, x: torch.Tensor, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        xz = torch.cat([x, z], dim=-1)
        h = self.act(self.dec_fc1(xz))
        out = self.dec_out(h)
        mu_y, logvar_y = out.chunk(2, dim=-1)
        logvar_y = torch.clamp(logvar_y, min=-10.0, max=10.0)
        return mu_y, logvar_y

    # --- flow posterior sampling q(z|x,y) ---

    def sample_z_and_logq(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        n_samples: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample latent variables from q(z|x,y) and return log q."""
        B = x.size(0)
        Dz = self.latent_dim
        K = n_samples

        mu_q, logvar_q = self.encode(x, y)             # [B,Dz]
        mu_q_exp = mu_q.unsqueeze(0).expand(K, -1, -1)      # [K,B,Dz]
        logvar_q_exp = logvar_q.unsqueeze(0).expand(K, -1, -1)

        eps = torch.randn(K, B, Dz, device=x.device)
        std_q_exp = torch.exp(0.5 * logvar_q_exp)
        z0 = mu_q_exp + std_q_exp * eps                     # [K,B,Dz]

        # log q0(z0 | x,y)
        log_q0 = gaussian_log_prob(z0, mu_q_exp, logvar_q_exp)  # [K,B]

        
        z = z0.reshape(K * B, Dz)                # [K*B,Dz]
        log_q = log_q0.reshape(K * B)            # [K*B]
        for flow in self.flows:
            z, log_det = flow(z)                 # log_det: [K*B]
            log_q = log_q - log_det              

        z_K = z.reshape(K, B, Dz)
        log_q = log_q.reshape(K, B)
        return z_K, log_q

    

    def iwae_loss(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        n_samples: int = 5,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute the negative IWAE bound and diagnostic statistics."""
        B = x.size(0)
        K = n_samples

        # z ~ q(z|x,y)
        z_K, log_q = self.sample_z_and_logq(x, y, n_samples=K)  # [K,B,Dz], [K,B]

        
        mu_p, logvar_p = self.prior(x)                   # [B,Dz]
        mu_p_exp = mu_p.unsqueeze(0).expand(K, -1, -1)   # [K,B,Dz]
        logvar_p_exp = logvar_p.unsqueeze(0).expand(K, -1, -1)

        log_p_z = gaussian_log_prob(z_K, mu_p_exp, logvar_p_exp)  # [K,B]

        
        x_exp = x.unsqueeze(0).expand(K, -1, -1)         # [K,B,Dx]
        y_exp = y.unsqueeze(0).expand(K, -1, -1)         # [K,B,Dy]

        x_flat = x_exp.reshape(K * B, -1)
        z_flat = z_K.reshape(K * B, -1)
        mu_y, logvar_y = self.decode(x_flat, z_flat)     # [K*B,Dy], [K*B,Dy]

        y_flat = y_exp.reshape(K * B, -1)
        log_p_y_flat = gaussian_log_prob(y_flat, mu_y, logvar_y)   # [K*B]
        log_p_y = log_p_y_flat.reshape(K, B)                        # [K,B]

        # log p(y,z|x)
        log_p_yz = log_p_y + log_p_z                # [K,B]

        # importance weights
        log_w = log_p_yz - log_q                    # [K,B]

        
        log_w_max, _ = torch.max(log_w, dim=0, keepdim=True)  # [1,B]
        log_mean_w = log_w_max.squeeze(0) + torch.log(
            torch.mean(torch.exp(log_w - log_w_max), dim=0)
        )  # [B]

        elbo = torch.mean(log_mean_w)           
        loss = -elbo                            

        stats = {
            "loss": loss.item(),
            "elbo": elbo.item(),
            "log_p_y": log_p_y.mean().item(),
            "log_p_z": log_p_z.mean().item(),
            "log_q": log_q.mean().item(),
        }
        return loss, stats

    @torch.no_grad()
    def predict(self, x: torch.Tensor, n_samples_z: int = 20) -> torch.Tensor:
        """Predict the expected output by sampling from p(z|x)."""
        self.eval()
        B = x.size(0)
        Dz = self.latent_dim
        Dy = self.output_size

        mu_p, logvar_p = self.prior(x)          # [B,Dz]
        mu_p_exp = mu_p.unsqueeze(0).expand(n_samples_z, -1, -1)      # [K,B,Dz]
        logvar_p_exp = logvar_p.unsqueeze(0).expand(n_samples_z, -1, -1)
        std_p_exp = torch.exp(0.5 * logvar_p_exp)

        eps = torch.randn(n_samples_z, B, Dz, device=x.device)
        z_samples = mu_p_exp + std_p_exp * eps  # [K,B,Dz]

        x_exp = x.unsqueeze(0).expand(n_samples_z, -1, -1)   # [K,B,Dx]
        x_flat = x_exp.reshape(n_samples_z * B, -1)
        z_flat = z_samples.reshape(n_samples_z * B, -1)

        mu_y, logvar_y = self.decode(x_flat, z_flat)         # [K*B,Dy]
        mu_y_samples = mu_y.reshape(n_samples_z, B, Dy)      # [K,B,Dy]

        y_pred_mean = mu_y_samples.mean(dim=0)               # [B,Dy]
        return y_pred_mean


# ============================================================
# 4. Worst-corner VAE wrapper
# ============================================================

class WorstCornerVAE(nn.Module):
    """Flow-IWAE-CVAE wrapper used to predict worst-corner multi-objective targets."""

    def __init__(
        self,
        input_dim: int = 41,
        output_dim: int = 1,   
        width: Optional[Sequence[int]] = None,  
        grid: int = 3,                          
        k: int = 5,                             
        seed: Optional[int] = 42,
        learning_rate: float = 1e-3,
        device: str = "cpu",
        recon_reward_weight: float = 1.0,
        recon_corner_weight: float = 1.5,
        latent_dim: int = 8,
        hidden_dim: int = 64,
        n_flows: int = 4,
        iwae_samples: int = 5,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.device = torch.device(device)

        if seed is not None:
            torch.manual_seed(int(seed))
            np.random.seed(int(seed))

        
        self.net = FlowIWAEConditionalVAE(
            input_size=self.input_dim,
            output_size=self.output_dim,
            latent_dim=int(latent_dim),
            hidden_dim=int(hidden_dim),
            n_flows=int(n_flows),
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=learning_rate)

        self.recon_reward_weight = float(recon_reward_weight)
        self.recon_corner_weight = float(recon_corner_weight)
        self.iwae_samples = int(iwae_samples)

        
        self.std_c = RunningStandardizer(dim=self.input_dim, device=str(self.device))
        self.std_y = RunningStandardizer(dim=self.output_dim, device=str(self.device))

        self.train_losses: List[float] = []
        self.training_step: int = 0
        if self.output_dim <= len(VAE_OBJECTIVE_KEYS):
            self.objective_keys = tuple(VAE_OBJECTIVE_KEYS[: self.output_dim])
        else:
            self.objective_keys = tuple(f"objective_{idx}" for idx in range(self.output_dim))

    def forward(self, conditioned: torch.Tensor) -> torch.Tensor:
        """Return a one-sample normalized prediction for compatibility."""
        if conditioned.ndim == 1:
            conditioned = conditioned.unsqueeze(0)
        return self.net.predict(conditioned, n_samples_z=1)

    def update(
        self,
        batch: Sequence[VAETrainingData],
        beta: float = 1.0,  
        update_stats: bool = True,
    ) -> Dict[str, float]:
        if not batch:
            return {"total_loss": 0.0, "recon_loss": 0.0, "kl_loss": 0.0, "kl_weight": 0.0}

        self.train()
        c_raw, y_raw = _stack_training_data(batch, self.device)

        
        valid_mask = torch.all(torch.isfinite(c_raw), dim=1) & torch.all(torch.isfinite(y_raw), dim=1)
        c_raw = c_raw[valid_mask]
        y_raw = y_raw[valid_mask]

        if c_raw.size(0) == 0:
            return {"total_loss": 0.0, "recon_loss": 0.0, "kl_loss": 0.0, "kl_weight": 0.0}

        if update_stats:
            
            self.std_c.update(c_raw)
            self.std_y.update(y_raw)

        c = self.std_c.transform(c_raw)
        y = self.std_y.transform(y_raw)

        loss, stats = self.net.iwae_loss(c, y, n_samples=self.iwae_samples)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.training_step += 1
        loss_value = float(loss.item())
        self.train_losses.append(loss_value)

        
        
        return {
            "total_loss": loss_value,
            "recon_loss": loss_value,
            "kl_loss": 0.0,
            "kl_weight": 0.0,
        }

    def offline_train(
        self,
        dataset: Sequence[VAETrainingData],
        epochs: int = 55,
        batch_size: int = 32,
        shuffle: bool = True,
    ) -> List[float]:
        if not dataset:
            return []

        c_raw, y_raw = _stack_training_data(dataset, self.device)

        
        valid_mask = torch.all(torch.isfinite(c_raw), dim=1) & torch.all(torch.isfinite(y_raw), dim=1)
        c_raw = c_raw[valid_mask]
        y_raw = y_raw[valid_mask]

        if c_raw.size(0) == 0:
            print("[VAE] No valid samples after NaN filtering")
            return []

        
        self.std_c.reset()
        self.std_y.reset()
        self.std_c.update(c_raw)
        self.std_y.update(y_raw)

        c_norm = self.std_c.transform(c_raw)
        y_norm = self.std_y.transform(y_raw)

        epoch_losses: List[float] = []
        num_samples = c_norm.size(0)
        bsz = max(1, int(batch_size))

        for epoch in range(max(1, int(epochs))):
            if shuffle:
                indices = torch.randperm(num_samples, device=self.device)
            else:
                indices = torch.arange(num_samples, device=self.device)

            total_loss = 0.0
            batches = 0

            for start in range(0, num_samples, bsz):
                idx = indices[start:start + bsz]
                c_batch = c_norm[idx]
                y_batch = y_norm[idx]

                loss, stats = self.net.iwae_loss(c_batch, y_batch, n_samples=self.iwae_samples)

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
                self.optimizer.step()

                batches += 1
                loss_item = float(loss.item())
                total_loss += loss_item
                self.training_step += 1
                self.train_losses.append(loss_item)

            epoch_loss = total_loss / max(1, batches)
            epoch_losses.append(epoch_loss)
            

        return epoch_losses

    @torch.no_grad()
    def predict_pvt_reward(
        self,
        condition: np.ndarray,
        num_samples: int = 20,
    ) -> Tuple[float, float]:
        """Predict the PVT worst reward mean and standard deviation."""
        pred_mean, pred_std = self.predict_multi_objective(condition, num_samples=num_samples)
        return float(pred_mean[0]), float(pred_std[0])

    @torch.no_grad()
    def predict_multi_objective(
        self,
        condition: np.ndarray,
        num_samples: int = 20,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predict the PVT worst-case multi-objective vector."""
        self.eval()
        c_raw = torch.as_tensor(condition, dtype=torch.float32, device=self.device).unsqueeze(0)
        c = self.std_c.transform(c_raw)
        sample_count = max(1, int(num_samples))
        pred_samples: List[np.ndarray] = []
        for _ in range(sample_count):
            y_norm = self.net.predict(c, n_samples_z=1)
            y = self.std_y.inverse(y_norm)
            pred_samples.append(y[0].detach().cpu().numpy().astype(np.float32))
        pred_array = np.stack(pred_samples, axis=0)
        pred_mean = pred_array.mean(axis=0).astype(np.float32)
        pred_std = pred_array.std(axis=0).astype(np.float32)
        return pred_mean, pred_std

    @torch.no_grad()
    def predict_pvt_objectives(
        self,
        condition: np.ndarray,
        num_samples: int = 20,
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        pred_mean, pred_std = self.predict_multi_objective(condition, num_samples=num_samples)
        return (
            objective_vector_to_dict(pred_mean, self.objective_keys),
            objective_vector_to_dict(pred_std, self.objective_keys),
        )

    def save(self, path: str) -> None:
        torch.save(
            {
                "state_dict": self.net.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "train_losses": self.train_losses,
                "training_step": self.training_step,
                "std_c": self.std_c.state_dict(),
                "std_y": self.std_y.state_dict(),
                "config": {
                    "input_dim": self.input_dim,
                    "output_dim": self.output_dim,
                },
            },
            path,
        )

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.net.load_state_dict(ckpt["state_dict"])
        if "optimizer_state_dict" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.train_losses = ckpt.get("train_losses", [])
        self.training_step = int(ckpt.get("training_step", 0))
        try:
            self.std_c.load_state_dict(ckpt.get("std_c", {}))
            self.std_y.load_state_dict(ckpt.get("std_y", {}))
        except Exception:
            self.std_c.reset()
            self.std_y.reset()


# ============================================================

# ============================================================

def _legacy_prepare_vae_training_data(
    tt_action: np.ndarray,
    tt_performance: Dict,
    tt_reward: float,
    pvt_worst_reward: float,
    pvt_worst_performance: Optional[Dict] = None,
    pvt_worst_corner_idx: int = None,  
    num_corners: int = 20,
) -> Optional[VAETrainingData]:
    try:
        perf_keys = [
            "phase_margin (deg)",
            "dcgain",
            "PSRP",
            "PSRN",
            "PSRR",
            "cmrrdc",
            # "vos",  # removed per requirement
            "TC",
            "setting_time",
            "FOML",
            "FOMS",
            "Active Area",
            "Power",
            "GBW",
            "sr",
            # "FOM_AMP",  # removed per requirement
        ]
        tt_perf = []
        for key in perf_keys:
            value = tt_performance.get(key, 0.0)
            if not np.isfinite(value):
                value = 0.0
            tt_perf.append(float(value))

        input_objectives = _extract_multi_objective_vector(
            tt_performance,
            fallback_reward=float(tt_reward),
        )
        condition = np.concatenate(
            [
                tt_action.astype(np.float32),
                np.asarray(tt_perf, dtype=np.float32),
                input_objectives,
            ],
            axis=0,
        ).astype(np.float32)

        target_y = _extract_multi_objective_vector(
            pvt_worst_performance,
            fallback_reward=float(pvt_worst_reward),
        )
        
        
        if not (np.all(np.isfinite(condition)) and np.all(np.isfinite(target_y))):
            return None
        
        
        return VAETrainingData(condition=condition, target_y=target_y, corner_label=-1)
    except Exception as exc:
        print(f"[VAE] prepare data failed: {exc}")
        return None


def prepare_vae_training_data(
    tt_action: np.ndarray,
    tt_performance: Dict,
    tt_reward: Optional[float] = None,
    pvt_worst_reward: Optional[float] = None,
    pvt_worst_performance: Optional[Dict] = None,
    pvt_worst_corner_idx: Optional[int] = None,
    num_corners: int = 20,
    tt_objectives: Optional[Dict[str, Any]] = None,
    pvt_worst_objectives: Optional[Dict[str, Any]] = None,
    objective_keys: Optional[Sequence[str]] = None,
    pm_feasible: Optional[bool] = None,
) -> Optional[VAETrainingData]:
    """Prepare one VAE sample using TT inputs and worst-corner multi-objective targets."""
    try:
        del num_corners
        target_keys = tuple(objective_keys) if objective_keys is not None else tuple(VAE_OBJECTIVE_KEYS)
        tt_perf_info = tt_performance or {}
        pvt_perf_info = pvt_worst_performance or {}
        if tt_objectives is None:
            raw_tt_objectives = tt_perf_info.get("objective_rewards", None)
            if isinstance(raw_tt_objectives, dict):
                tt_objectives = raw_tt_objectives
        if pvt_worst_objectives is None:
            raw_pvt_objectives = pvt_perf_info.get("objective_rewards", None)
            if isinstance(raw_pvt_objectives, dict):
                pvt_worst_objectives = raw_pvt_objectives

        if pvt_worst_objectives is not None or objective_keys is not None:
            tt_target_objectives = build_target_objectives(
                objectives=tt_objectives,
                performance=tt_perf_info,
                objective_keys=target_keys,
            )
            pvt_target_objectives = build_target_objectives(
                objectives=pvt_worst_objectives,
                performance=pvt_perf_info if pvt_perf_info else pvt_worst_objectives,
                objective_keys=target_keys,
            )
            condition = build_vae_condition(
                tt_action=np.asarray(tt_action, dtype=np.float32),
                tt_performance=tt_perf_info,
                tt_objectives=tt_target_objectives,
                objective_keys=target_keys,
            )
            target_y = extract_objective_vector(pvt_target_objectives, objective_keys=target_keys)
            if pm_feasible is None:
                pm_feasible = bool(_safe_float(pvt_target_objectives.get("PM_violation", 1.0), default=1.0) <= 1e-8)
        else:
            condition = build_vae_condition(
                tt_action=np.asarray(tt_action, dtype=np.float32),
                tt_performance=tt_perf_info,
                tt_objectives=None,
                tt_reward=_safe_float(tt_reward, default=0.0),
            )
            target_y = np.asarray([_safe_float(pvt_worst_reward, default=0.0)], dtype=np.float32)
            if pm_feasible is None:
                pm_feasible = False

        if not (np.all(np.isfinite(condition)) and np.all(np.isfinite(target_y))):
            return None

        corner_label = int(pvt_worst_corner_idx) if pvt_worst_corner_idx is not None else -1
        return VAETrainingData(condition=condition, target_y=target_y, corner_label=corner_label)
    except Exception as exc:
        print(f"[VAE] prepare data failed: {exc}")
        return None


# Backward-compatible aliases for the previous KAN naming.
KANTrainingData = VAETrainingData
KAN = WorstCornerVAE
prepare_kan_training_data = prepare_vae_training_data

__all__ = [
    "FlowIWAEConditionalVAE",
    "PlanarFlow",
    "RunningStandardizer",
    "VAE_CONDITION_PERFORMANCE_KEYS",
    "RL_OBJECTIVE_KEYS",
    "VAE_OBJECTIVE_KEYS",
    "VAETrainingData",
    "WorstCornerVAE",
    "build_target_objectives",
    "build_vae_condition",
    "compute_phase_margin_violation",
    "extract_objective_vector",
    "extract_performance_vector",
    "objective_vector_to_dict",
    "prepare_vae_training_data",
    "KANTrainingData",
    "KAN",
    "prepare_kan_training_data",
]
