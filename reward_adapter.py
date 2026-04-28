"""
用于 GRPO 的奖励适配器 - 阶段 3 实现

本模块提供不同的奖励转换策略，
以将原 DDPG 奖励函数转换为适合 GRPO 训练的形式。

原始 DDPG 奖励范围：[-11, 0]
- 11 个指标中的每一个贡献一个 [-1, 0] 之间的分数
- 总奖励 = 11 个独立分数的总和
- 奖励 = 0 意味着所有指标均满足要求
- 奖励 = -11 意味着所有指标均失败

适用于 GRPO 的奖励策略：
1. Binary (二值): {0, 1} - 简单的成功/失败
2. Multi-level (多层级): {0, 0.5, 1.0, 1.5, 2.0} - 阶梯式奖励
3. Normalized (连续归一化): [0, 2.0] - 平滑转换
4. Adaptive (自适应): 基于训练进度的动态缩放
"""

import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class RewardConfig:
    """Configuration for reward adapter"""
    strategy: str = 'multi_level'  # 'binary', 'multi_level', 'normalized', 'adaptive'

    # Multi-level thresholds (number of satisfied metrics)
    level_thresholds: Tuple[int, int, int, int] = (11, 8, 5, 2)  # Excellent, Good, Fair, Poor
    level_rewards: Tuple[float, float, float, float, float] = (2.0, 1.5, 1.0, 0.5, 0.0)

    # Normalized transformation parameters
    raw_min: float = -11.0  # Original min reward
    raw_max: float = 0.0    # Original max reward
    target_min: float = 0.0  # GRPO min reward
    target_max: float = 2.0  # GRPO max reward

    # Adaptive parameters
    adaptive_temperature: float = 1.0  # Controls reward sharpness
    adaptive_decay: float = 0.9995     # Temperature decay per step
    adaptive_min_temp: float = 0.1     # Minimum temperature


class RewardAdapter:
    """
    将 DDPG 奖励 [-11, 0] 转换为与 GRPO 兼容的奖励。

    支持多种策略：
    - Binary：简单的 0/1 分类
    - Multi-level：基于指标达标数量的阶梯式奖励
    - Normalized：映射到 [0, 2.0] 范围的线性转换
    - Adaptive：基于温度因子的 softmax 连续转换
    """

    def __init__(self, config: Optional[RewardConfig] = None):
        self.config = config if config is not None else RewardConfig()
        self.current_temperature = self.config.adaptive_temperature
        self.adaptation_count = 0

        # Statistics tracking
        self.reward_history = []
        self.satisfaction_history = []

    def transform(self, raw_reward: float, performance: Dict) -> Tuple[float, Dict]:
        """
        Transform raw DDPG reward to GRPO-compatible reward.

        Args:
            raw_reward: Original reward from environment [-11, 0]
            performance: Performance metrics dictionary

        Returns:
            transformed_reward: GRPO-compatible reward
            info: Additional information dictionary
        """
        strategy = self.config.strategy

        if strategy == 'binary':
            return self._binary_transform(raw_reward, performance)
        elif strategy == 'multi_level':
            return self._multi_level_transform(raw_reward, performance)
        elif strategy == 'normalized':
            return self._normalized_transform(raw_reward, performance)
        elif strategy == 'adaptive':
            return self._adaptive_transform(raw_reward, performance)
        elif strategy == 'ehvi':
            return self._ehvi_transform(raw_reward, performance)
        else:
            raise ValueError(f"Unknown reward strategy: {strategy}")

    def _binary_transform(self, raw_reward: float, performance: Dict) -> Tuple[float, Dict]:
        """
        Binary reward: 0 or 1

        Logic:
        - reward = 1.0 if raw_reward >= 0 (all metrics satisfied)
        - reward = 0.0 otherwise
        """
        transformed = 1.0 if raw_reward >= 0 else 0.0

        info = {
            'raw_reward': raw_reward,
            'transformed_reward': transformed,
            'strategy': 'binary',
            'success': raw_reward >= 0
        }

        return transformed, info

    def _multi_level_transform(self, raw_reward: float, performance: Dict) -> Tuple[float, Dict]:
        """
        Multi-level reward based on number of satisfied metrics.

        Logic:
        - Count how many metrics are satisfied (score = 0)
        - Assign reward based on satisfaction count:
          * 11 satisfied → 2.0 (Excellent)
          * 8-10 satisfied → 1.5 (Good)
          * 5-7 satisfied → 1.0 (Fair)
          * 2-4 satisfied → 0.5 (Poor)
          * 0-1 satisfied → 0.0 (Failed)
        """
        # Count satisfied metrics
        # In DDPG: each metric contributes score in [-1, 0]
        # score = 0 means satisfied
        # We approximate: num_satisfied ≈ 11 + raw_reward (since each failure is -1)
        num_satisfied = int(11 + raw_reward)
        num_satisfied = np.clip(num_satisfied, 0, 11)

        # Assign level-based reward
        thresholds = self.config.level_thresholds
        rewards = self.config.level_rewards

        if num_satisfied >= thresholds[0]:      # 11
            transformed = rewards[0]  # 2.0
            level = 'Excellent'
        elif num_satisfied >= thresholds[1]:    # 8
            transformed = rewards[1]  # 1.5
            level = 'Good'
        elif num_satisfied >= thresholds[2]:    # 5
            transformed = rewards[2]  # 1.0
            level = 'Fair'
        elif num_satisfied >= thresholds[3]:    # 2
            transformed = rewards[3]  # 0.5
            level = 'Poor'
        else:
            transformed = rewards[4]  # 0.0
            level = 'Failed'

        info = {
            'raw_reward': raw_reward,
            'transformed_reward': transformed,
            'strategy': 'multi_level',
            'num_satisfied': num_satisfied,
            'level': level,
            'success': num_satisfied == 11
        }

        self.satisfaction_history.append(num_satisfied)

        return transformed, info

    def _normalized_transform(self, raw_reward: float, performance: Dict) -> Tuple[float, Dict]:
        """
        Normalized continuous reward: [0, 2.0]

        Logic:
        - Linear transformation from [-11, 0] to [0, 2.0]
        - Formula: r_new = (r - r_min) / (r_max - r_min) * (target_max - target_min) + target_min
        """
        r_min = self.config.raw_min
        r_max = self.config.raw_max
        t_min = self.config.target_min
        t_max = self.config.target_max

        # Linear scaling
        transformed = (raw_reward - r_min) / (r_max - r_min) * (t_max - t_min) + t_min
        transformed = np.clip(transformed, t_min, t_max)

        info = {
            'raw_reward': raw_reward,
            'transformed_reward': transformed,
            'strategy': 'normalized',
            'success': raw_reward >= 0
        }

        return transformed, info

    def _adaptive_transform(self, raw_reward: float, performance: Dict) -> Tuple[float, Dict]:
        """
        Adaptive reward with temperature-based softmax.

        Logic:
        - Start with high temperature (soft rewards)
        - Gradually decrease temperature (sharper rewards)
        - Formula: r_new = 2.0 / (1 + exp(-raw_reward / temperature))

        This creates a sigmoid-like transformation that becomes
        more binary as temperature decreases.
        """
        # Sigmoid transformation with temperature
        # Map [-11, 0] → [0, 2.0] with temperature control
        transformed = self.config.target_max / (1 + np.exp(-raw_reward / self.current_temperature))

        # Decay temperature
        if self.current_temperature > self.config.adaptive_min_temp:
            self.current_temperature *= self.config.adaptive_decay

        self.adaptation_count += 1

        info = {
            'raw_reward': raw_reward,
            'transformed_reward': transformed,
            'strategy': 'adaptive',
            'temperature': self.current_temperature,
            'adaptation_step': self.adaptation_count,
            'success': raw_reward >= 0
        }

        return transformed, info

    # ------------------------------------------------------------------
    # EHVI-based transformation (direct formula over raw scores)
    # ------------------------------------------------------------------
    def _ehvi_transform(self, raw_reward: float, performance: Dict) -> Tuple[float, Dict]:
        """EHVI-like reward computed from raw per-metric scores using the new formula.

        New formula (alpha=2):
            r = -12 + 2 * exp( (1/M) * sum_{m=1..M} ln( 1 + 5 * exp(s_m / alpha) ) )

        Where s_m are the original metric scores (≤ 0, 0 means satisfied),
        and M is the count of included scores. This yields r ∈ [-10, 0].

        Notes:
        - We still filter out {TC_score, PSRP_score, PSRN_score} and never synthesize PSRR_score.
        - Non-finite scores are treated as very negative (≈ contribute ln(1)→0).
        - No piecewise/soft normalization; computed directly from raw scores.

        If no score keys are present, return a strong negative sentinel (e.g., -999)
        to indicate invalid transformation.
        """
        # Extract raw metric scores
        raw_score_keys = [k for k in performance.keys() if k.endswith('_score')]
        # 需要排除的原始键
        exclude_keys = {'TC_score', 'PSRP_score', 'PSRN_score'}
        # 不再合成 PSRR_score，完全依赖环境返回
        score_keys = [k for k in raw_score_keys if k not in exclude_keys]
        if not score_keys:
            # No *_score keys → treat as invalid and return very bad reward
            transformed = -999.0
            info = {
                'raw_reward': raw_reward,
                'transformed_reward': transformed,
                'strategy': 'ehvi',
                'note': 'no_score_keys_return_bad'
            }
            return transformed, info

        # Direct formula over raw scores
        alpha = 2.0
        scores = np.array([performance[k] for k in score_keys], dtype=float)
        scores = np.where(np.isfinite(scores), scores, -1e3)
        interior = 1.0 + 5.0 * np.exp(scores / alpha)  # each term in (1, 6]
        # Geometric mean via log; guard against any tiny numerical underflow
        log_g = float(np.mean(np.log(interior)))
        g = float(np.exp(log_g))
        # Reward scaling: r = -12 + 2g ∈ [-10, 0]
        transformed = float(-12.0 + 2.0 * g)

        info = {
            'raw_reward': raw_reward,
            'transformed_reward': transformed,
            'strategy': 'ehvi',
            'num_metrics': len(score_keys),
            'scores_raw': {k: performance[k] for k in score_keys},
            'alpha': alpha,
            'geom_mean_g': g,
            'log_g': log_g,
            'excluded_keys': list(sorted(exclude_keys))
        }

        return transformed, info

    def get_statistics(self) -> Dict:
        """Get reward transformation statistics"""
        if len(self.reward_history) == 0:
            return {}

        stats = {
            'num_transformations': len(self.reward_history),
            'mean_reward': np.mean(self.reward_history),
            'std_reward': np.std(self.reward_history),
            'max_reward': np.max(self.reward_history),
            'min_reward': np.min(self.reward_history)
        }

        if len(self.satisfaction_history) > 0:
            stats['mean_satisfaction'] = np.mean(self.satisfaction_history)
            stats['success_rate'] = np.mean([s == 11 for s in self.satisfaction_history])

        return stats

    def reset(self):
        """Reset adapter state"""
        self.current_temperature = self.config.adaptive_temperature
        self.adaptation_count = 0
        self.reward_history = []
        self.satisfaction_history = []


# ============================================================================
# Convenience Functions
# ============================================================================

def create_reward_adapter(strategy: str = 'multi_level', **kwargs) -> RewardAdapter:
    """
    Create a reward adapter with specified strategy.

    Args:
        strategy: 'binary', 'multi_level', 'normalized', 'adaptive'
        **kwargs: Additional configuration parameters

    Returns:
        RewardAdapter instance

    Examples:
        >>> adapter = create_reward_adapter('multi_level')
        >>> transformed, info = adapter.transform(-3.5, performance_dict)
    """
    config = RewardConfig(strategy=strategy, **kwargs)
    return RewardAdapter(config)


def compare_strategies(raw_rewards: np.ndarray) -> Dict:
    """
    Compare different reward strategies on a set of raw rewards.

    Args:
        raw_rewards: Array of raw DDPG rewards

    Returns:
        Dictionary with comparison results
    """
    strategies = ['binary', 'multi_level', 'normalized', 'adaptive']
    results = {}

    for strategy in strategies:
        adapter = create_reward_adapter(strategy)
        transformed_rewards = []

        for raw_reward in raw_rewards:
            transformed, _ = adapter.transform(raw_reward, {})
            transformed_rewards.append(transformed)

        results[strategy] = {
            'rewards': np.array(transformed_rewards),
            'mean': np.mean(transformed_rewards),
            'std': np.std(transformed_rewards),
            'range': (np.min(transformed_rewards), np.max(transformed_rewards))
        }

    return results


# ============================================================================
# Test and Visualization
# ============================================================================

if __name__ == '__main__':
    """Test reward adapter with sample rewards"""
    import matplotlib.pyplot as plt

    print("="*80)
    print("Reward Adapter Test")
    print("="*80)

    # Test rewards ranging from -11 (worst) to 0 (best)
    test_rewards = np.linspace(-11, 0, 50)

    # Test all strategies
    strategies = ['binary', 'multi_level', 'normalized', 'adaptive']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('GRPO Reward Transformation Strategies', fontsize=16, fontweight='bold')

    for idx, strategy in enumerate(strategies):
        ax = axes[idx // 2, idx % 2]

        adapter = create_reward_adapter(strategy)
        transformed_rewards = []

        for raw_reward in test_rewards:
            transformed, info = adapter.transform(raw_reward, {})
            transformed_rewards.append(transformed)

        # Plot
        ax.plot(test_rewards, transformed_rewards, linewidth=2, label=strategy)
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.3, label='Failure threshold')
        ax.axvline(x=0, color='green', linestyle='--', alpha=0.3, label='Success threshold')

        ax.set_xlabel('Raw DDPG Reward')
        ax.set_ylabel('Transformed GRPO Reward')
        ax.set_title(f'{strategy.upper()} Strategy')
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout()
    plt.savefig('reward_transformation_strategies.png', dpi=150, bbox_inches='tight')
    print("\n✓ Visualization saved to: reward_transformation_strategies.png")

    # Print sample transformations
    print("\n" + "="*80)
    print("Sample Transformations")
    print("="*80)

    sample_rewards = [-11, -8, -5, -2, 0]

    for raw_reward in sample_rewards:
        print(f"\nRaw Reward: {raw_reward:.1f}")
        print("-" * 40)

        for strategy in strategies:
            adapter = create_reward_adapter(strategy)
            transformed, info = adapter.transform(raw_reward, {})

            extra_info = ""
            if 'level' in info:
                extra_info = f" ({info['level']})"
            elif 'temperature' in info:
                extra_info = f" (T={info['temperature']:.3f})"

            print(f"  {strategy:15s}: {transformed:.4f}{extra_info}")

    print("\n" + "="*80)
    print("✓ Reward Adapter Test Complete")
    print("="*80)
