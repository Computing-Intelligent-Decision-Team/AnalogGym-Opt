import csv
import json
import os
import numpy as np
import matplotlib
# 强制使用非交互式后端，避免弹窗阻塞
try:
    if matplotlib.get_backend().lower() not in ('agg', 'pdf', 'svg'):
        matplotlib.use('Agg')
except Exception:
    pass
import matplotlib.pyplot as plt
from IPython.display import clear_output
from typing import Any, Dict, List, Optional

from reporting_metrics import get_reporting_metrics, get_reporting_scores


def _to_native(agent, v):
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
        if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
            return None
    return v


def _json_ready(v):
    if isinstance(v, dict):
        return {str(k): _json_ready(val) for k, val in v.items()}
    if isinstance(v, (list, tuple)):
        return [_json_ready(x) for x in v]
    return _to_native(None, v)


def _series_list(values):
    seq = list(values or [])
    return [_to_native(None, v) for v in seq]


def _series_at(values, idx):
    if idx >= len(values):
        return None
    return _to_native(None, values[idx])


def _min_max_log_series(history_lists):
    min_series = []
    max_series = []
    log_series = []
    for values in list(history_lists or []):
        arr = np.asarray(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            min_series.append(None)
            max_series.append(None)
            log_series.append(None)
            continue
        current_min = float(arr.min())
        current_max = float(arr.max())
        min_series.append(current_min)
        max_series.append(current_max)
        log_series.append(float(np.log10(max(1e-4, -current_min))))
    return min_series, max_series, log_series


def _plot_runtime_context(agent):
    run_all_corners = bool(getattr(agent, 'run_all_corners', False))
    pvt_outer_loop_enabled = bool(getattr(agent, 'pvt_outer_loop_enabled', False))

    if run_all_corners:
        return {
            'reward_source': 'full_pvt',
            'reward_source_label': 'Worst-PVT',
            'primary_history_lists': list(getattr(agent, 'per_step_design_rewards', []) or []),
            'primary_title': 'PVT Worst Training Reward',
            'primary_filename': 'pvt_worst',
            'comparison_history_lists': list(getattr(agent, 'per_step_extra_corner_rewards', []) or []),
            'comparison_title': 'TT Training Reward',
            'comparison_filename': 'tt_reference',
            'objective_title': 'Mean Constraint Score + Mean Raw Objective History (Worst-PVT)',
            'wandb_primary_key': 'wandb/pvt_worst',
            'wandb_comparison_key': 'wandb/tt_reference',
        }

    reward_source_label = 'TT Inner-Loop' if pvt_outer_loop_enabled else 'TT'
    return {
        'reward_source': 'tt',
        'reward_source_label': reward_source_label,
        'primary_history_lists': list(getattr(agent, 'per_step_design_rewards', []) or []),
        'primary_title': 'TT Training Reward',
        'primary_filename': 'tt_training_reward',
        'comparison_history_lists': [],
        'comparison_title': None,
        'comparison_filename': None,
        'objective_title': f'Mean Constraint Score + Mean Raw Objective History ({reward_source_label})',
        'wandb_primary_key': 'wandb/tt_training_reward',
        'wandb_comparison_key': None,
    }


def _plot_data_series(agent):
    objective_plot_history = getattr(agent, 'objective_plot_history', {}) or {}
    context = _plot_runtime_context(agent)
    primary_min, primary_max, primary_log = _min_max_log_series(context['primary_history_lists'])
    comparison_min, comparison_max, comparison_log = _min_max_log_series(context['comparison_history_lists'])
    verified_pvt_min, verified_pvt_max, verified_pvt_log = _min_max_log_series(
        getattr(agent, 'per_step_verified_pvt_rewards', [])
    )

    if context['reward_source'] == 'full_pvt':
        pvt_min, pvt_max, pvt_log = primary_min, primary_max, primary_log
        tt_min, tt_max, tt_log = comparison_min, comparison_max, comparison_log
    else:
        pvt_min, pvt_max, pvt_log = [], [], []
        tt_min, tt_max, tt_log = primary_min, primary_max, primary_log

    return {
        'mean_training_reward': _series_list(getattr(agent, 'reward_history', [])),
        'policy_loss': _series_list(getattr(agent, 'loss_history', [])),
        'success_rate': _series_list(getattr(agent, 'success_rate_history', [])),
        'grad_norm': _series_list(getattr(agent, 'grad_norm_history', [])),
        'kl': _series_list(getattr(agent, 'kl_history', [])),
        'clip_fraction': _series_list(getattr(agent, 'clip_fraction_history', [])),
        'entropy': _series_list(getattr(agent, 'entropy_history', [])),
        'advantage_min': _series_list(getattr(agent, 'advantage_min_history', [])),
        'advantage_max': _series_list(getattr(agent, 'advantage_max_history', [])),
        'raw_reward_worst': _series_list(getattr(agent, 'raw_reward_worst_history', [])),
        'group_reward_min': _series_list(getattr(agent, 'group_reward_min_history', [])),
        'group_reward_max': _series_list(getattr(agent, 'group_reward_max_history', [])),
        'pm_violation_mean': _series_list(getattr(agent, 'pm_violation_history', [])),
        'pm_feasible_rate': _series_list(getattr(agent, 'pm_feasible_rate_history', [])),
        'constraint_reward_score': _series_list(objective_plot_history.get('constraint_reward', [])),
        'FOML': _series_list(objective_plot_history.get('FOML', [])),
        'FOMS': _series_list(objective_plot_history.get('FOMS', [])),
        'Active_Area': _series_list(objective_plot_history.get('Active Area', [])),
        'training_reward_min': _series_list(primary_min),
        'training_reward_max': _series_list(primary_max),
        'training_reward_log10_neg_min': _series_list(primary_log),
        'pvt_worst_min': _series_list(pvt_min),
        'pvt_worst_max': _series_list(pvt_max),
        'pvt_worst_log10_neg_min': _series_list(pvt_log),
        'tt_min': _series_list(tt_min),
        'tt_max': _series_list(tt_max),
        'tt_log10_neg_min': _series_list(tt_log),
        'verified_pvt_training_reward_mean': _series_list(getattr(agent, 'verified_pvt_reward_history', [])),
        'verified_pvt_training_reward_min': _series_list(verified_pvt_min),
        'verified_pvt_training_reward_max': _series_list(verified_pvt_max),
        'verified_pvt_training_reward_log10_neg_min': _series_list(verified_pvt_log),
        'verified_pvt_pm_violation_mean': _series_list(getattr(agent, 'verified_pvt_pm_violation_history', [])),
        'verified_pvt_pm_feasible_rate': _series_list(getattr(agent, 'verified_pvt_pm_feasible_rate_history', [])),
        'verified_pvt_constraint_reward_score': _series_list(
            (getattr(agent, 'verified_pvt_objective_plot_history', {}) or {}).get('constraint_reward', [])
        ),
        'verified_pvt_FOML': _series_list(
            (getattr(agent, 'verified_pvt_objective_plot_history', {}) or {}).get('FOML', [])
        ),
        'verified_pvt_FOMS': _series_list(
            (getattr(agent, 'verified_pvt_objective_plot_history', {}) or {}).get('FOMS', [])
        ),
        'verified_pvt_Active_Area': _series_list(
            (getattr(agent, 'verified_pvt_objective_plot_history', {}) or {}).get('Active Area', [])
        ),
        'pvt_verified_count': _series_list(getattr(agent, 'pvt_verified_count_history', [])),
        'verified_pvt_archive_size': _series_list(getattr(agent, 'pvt_verified_archive_size_history', [])),
        'pvt_phase': _series_list(getattr(agent, 'pvt_phase_history', [])),
        'covariance_condition_number': _series_list(getattr(agent, 'covariance_condition_history', [])),
    }


def _write_rows_csv(path, rows, fieldnames):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _export_vae_plot_data(agent, out_dir):
    use_vae = getattr(agent, 'use_vae', getattr(agent, 'use_kan', False))
    diagnostics_enabled = getattr(agent, 'vae_diagnostics_enabled', True)
    vae_pred_records = getattr(agent, '_vae_pred_records', getattr(agent, '_kan_pred_records', []))
    vae_multi_pred_records = getattr(agent, '_vae_multi_pred_records', getattr(agent, '_kan_multi_pred_records', []))
    if not use_vae or not diagnostics_enabled:
        return

    if vae_pred_records:
        rows = []
        for idx, (step, tt_reward, actual_reward, pred_reward) in enumerate(vae_pred_records, start=1):
            rows.append({
                'index': idx,
                'step': int(step),
                'tt_constraint_reward': _to_native(None, tt_reward),
                'actual_constraint_reward': _to_native(None, actual_reward),
                'pred_constraint_reward': _to_native(None, pred_reward),
            })
        _write_rows_csv(
            os.path.join(out_dir, 'vae_pred_vs_actual_latest.csv'),
            rows,
            ['index', 'step', 'tt_constraint_reward', 'actual_constraint_reward', 'pred_constraint_reward'],
        )

    if vae_multi_pred_records:
        rows = []
        for idx, record in enumerate(vae_multi_pred_records, start=1):
            actual = np.asarray(record.get('actual', []), dtype=float)
            pred = np.asarray(record.get('pred_mean', []), dtype=float)
            pred_std = np.asarray(record.get('pred_std', []), dtype=float)
            row = {
                'index': idx,
                'step': int(record.get('step', 0)),
                'tt_constraint_reward': _to_native(None, record.get('tt_reward', None)),
                'actual_constraint_reward': float(actual[0]) if actual.size > 0 else None,
                'pred_constraint_reward': float(pred[0]) if pred.size > 0 else None,
                'pred_constraint_reward_std': float(pred_std[0]) if pred_std.size > 0 else None,
                'actual_foml_score': float(actual[1]) if actual.size > 1 else None,
                'pred_foml_score': float(pred[1]) if pred.size > 1 else None,
                'pred_foml_score_std': float(pred_std[1]) if pred_std.size > 1 else None,
                'actual_foms_score': float(actual[2]) if actual.size > 2 else None,
                'pred_foms_score': float(pred[2]) if pred.size > 2 else None,
                'pred_foms_score_std': float(pred_std[2]) if pred_std.size > 2 else None,
                'actual_active_area_score': float(actual[3]) if actual.size > 3 else None,
                'pred_active_area_score': float(pred[3]) if pred.size > 3 else None,
                'pred_active_area_score_std': float(pred_std[3]) if pred_std.size > 3 else None,
                'actual_pm_violation': float(actual[4]) if actual.size > 4 else None,
                'pred_pm_violation': float(pred[4]) if pred.size > 4 else None,
                'pred_pm_violation_std': float(pred_std[4]) if pred_std.size > 4 else None,
            }
            rows.append(row)
        _write_rows_csv(
            os.path.join(out_dir, 'vae_multi_pred_vs_actual_latest.csv'),
            rows,
            list(rows[0].keys()),
        )


def save_plot_data(agent, snapshot_step: Optional[int] = None):
    try:
        out_dir = os.path.join(agent._training_saves_dir, 'plot_data')
        os.makedirs(out_dir, exist_ok=True)

        series = _plot_data_series(agent)
        max_len = 0
        for values in series.values():
            max_len = max(max_len, len(values))
        rows = []
        fieldnames = ['step'] + list(series.keys())
        for idx in range(max_len):
            row = {'step': idx + 1}
            for key, values in series.items():
                row[key] = _series_at(values, idx)
            rows.append(row)

        latest_csv_path = os.path.join(out_dir, 'training_curves_latest.csv')
        latest_json_path = os.path.join(out_dir, 'training_curves_latest.json')
        _write_rows_csv(latest_csv_path, rows, fieldnames)

        payload = {
            'metadata': {
                'generated_at_step': int(snapshot_step if snapshot_step is not None else getattr(agent, 'total_steps', 0)),
                'total_steps': int(getattr(agent, 'total_steps', 0)),
                'fieldnames': fieldnames,
                'reward_history_source': _plot_runtime_context(agent)['reward_source'],
                'objective_history_source': _plot_runtime_context(agent)['reward_source'],
                'verified_pvt_history_source': 'verified_pvt',
                'adaptive_pvt_schedule_summary': _json_ready(
                    getattr(agent, '_adaptive_pvt_schedule_summary', {})
                ),
                'reporting_schedule_summary': _json_ready(
                    getattr(agent, 'reporting_schedule_summary', {})
                ),
                'pvt_outer_loop_enabled': bool(getattr(agent, 'pvt_outer_loop_enabled', False)),
                'run_all_corners': bool(getattr(agent, 'run_all_corners', False)),
            },
            'series': _json_ready(series),
            'group_mean_reward_history': _json_ready(getattr(agent, 'group_mean_reward_history', [])),
            'group_mean_objective_history': _json_ready(getattr(agent, 'group_mean_objective_history', [])),
            'group_weight_history': _json_ready(getattr(agent, 'group_weight_history', [])),
            'objective_history': _json_ready(getattr(agent, 'objective_history', {})),
            'objective_plot_history': _json_ready(getattr(agent, 'objective_plot_history', {})),
            'verified_pvt_reward_history': _json_ready(getattr(agent, 'verified_pvt_reward_history', [])),
            'verified_pvt_pm_violation_history': _json_ready(getattr(agent, 'verified_pvt_pm_violation_history', [])),
            'verified_pvt_pm_feasible_rate_history': _json_ready(getattr(agent, 'verified_pvt_pm_feasible_rate_history', [])),
            'verified_pvt_objective_history': _json_ready(getattr(agent, 'verified_pvt_objective_history', {})),
            'verified_pvt_objective_plot_history': _json_ready(getattr(agent, 'verified_pvt_objective_plot_history', {})),
            'pvt_phase_history': _json_ready(getattr(agent, 'pvt_phase_history', [])),
            'verified_pvt_pareto_records': _json_ready(getattr(agent, 'verified_pvt_pareto_records', [])),
            'covariance_whitening_state': {
                'ready': bool(getattr(agent, '_covariance_whitener_ready', False)),
                'freeze_step': _to_native(None, getattr(agent, '_covariance_freeze_step', None)),
                'condition_number': _to_native(None, getattr(agent, '_covariance_condition_number', None)),
                'aggregation_vector': _json_ready(getattr(agent, '_covariance_aggregation_vector', None)),
            },
        }
        with open(latest_json_path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        if snapshot_step is not None:
            snapshot_csv_path = os.path.join(out_dir, f'training_curves_step_{int(snapshot_step)}.csv')
            snapshot_json_path = os.path.join(out_dir, f'training_curves_step_{int(snapshot_step)}.json')
            _write_rows_csv(snapshot_csv_path, rows, fieldnames)
            with open(snapshot_json_path, 'w', encoding='utf-8') as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)

        _export_vae_plot_data(agent, out_dir)
    except Exception as exc:
        print(f"[plot_data] export failed: {exc}")


def _save_vae_vs_actual_plot(agent, outfile, title='VAE Prediction vs Actual Constraint Reward', dpi=150):
    use_vae = getattr(agent, 'use_vae', getattr(agent, 'use_kan', False))
    diagnostics_enabled = getattr(agent, 'vae_diagnostics_enabled', True)
    vae_pred_worst_history = getattr(agent, 'vae_pred_worst_history', getattr(agent, 'kan_pred_worst_history', []))
    vae_actual_worst_history = getattr(agent, 'vae_actual_worst_history', getattr(agent, 'kan_actual_worst_history', []))
    if not use_vae or not diagnostics_enabled or not vae_pred_worst_history or not vae_actual_worst_history:
        return False

    count = min(len(vae_pred_worst_history), len(vae_actual_worst_history))
    x_axis = range(1, count + 1)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x_axis, vae_actual_worst_history[:count], label='Actual Constraint Reward', color='tab:blue')
    ax.plot(x_axis, vae_pred_worst_history[:count], label='VAE Predicted Constraint Reward', color='tab:red')
    ax.set_xlabel('Inference Steps')
    ax.set_ylabel('Reward')
    ax.set_title(title)
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outfile, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    return True


def _vae_multi_objective_arrays(agent):
    use_vae = getattr(agent, 'use_vae', getattr(agent, 'use_kan', False))
    diagnostics_enabled = getattr(agent, 'vae_diagnostics_enabled', True)
    records = getattr(agent, '_vae_multi_pred_records', getattr(agent, '_kan_multi_pred_records', []))
    if not use_vae or not diagnostics_enabled or not records:
        return None, None, None, []

    actual_rows = []
    pred_rows = []
    pred_std_rows = []
    for record in records:
        actual = np.asarray(record.get('actual', []), dtype=np.float32)
        pred = np.asarray(record.get('pred_mean', []), dtype=np.float32)
        pred_std = np.asarray(record.get('pred_std', []), dtype=np.float32)
        target_dim = min(actual.size, pred.size)
        if target_dim == 0:
            continue
        actual_rows.append(actual[:target_dim])
        pred_rows.append(pred[:target_dim])
        if pred_std.size >= target_dim:
            pred_std_rows.append(pred_std[:target_dim])
        else:
            pred_std_rows.append(np.zeros(target_dim, dtype=np.float32))

    if not actual_rows or not pred_rows:
        return None, None, None, []

    labels = ['constraint_reward', 'FOML_score', 'FOMS_score', 'Active_Area_score', 'PM_violation']
    actual_array = np.vstack(actual_rows)
    pred_array = np.vstack(pred_rows)
    pred_std_array = np.vstack(pred_std_rows) if pred_std_rows else np.zeros_like(pred_array)
    dim = min(actual_array.shape[1], pred_array.shape[1], len(labels))
    return actual_array[:, :dim], pred_array[:, :dim], pred_std_array[:, :dim], labels[:dim]


def _save_vae_multi_objective_plot(agent, outfile, title='VAE Multi-Objective Prediction vs Actual', dpi=150):
    actual_array, pred_array, pred_std_array, labels = _vae_multi_objective_arrays(agent)
    if actual_array is None or pred_array is None or not labels:
        return False

    dim = len(labels)
    fig, axes = plt.subplots(dim, 1, figsize=(11, max(8, 2.4 * dim)), sharex=True)
    if dim == 1:
        axes = [axes]
    x_axis = range(1, actual_array.shape[0] + 1)

    for idx in range(dim):
        label = labels[idx]
        ax = axes[idx]
        ax.plot(x_axis, actual_array[:, idx], label='Actual', color='tab:blue')
        ax.plot(x_axis, pred_array[:, idx], label='VAE Predicted', color='tab:red')
        if pred_std_array is not None and pred_std_array.shape[1] > idx:
            lower = pred_array[:, idx] - pred_std_array[:, idx]
            upper = pred_array[:, idx] + pred_std_array[:, idx]
            ax.fill_between(x_axis, lower, upper, color='tab:red', alpha=0.18, label='Pred ±1 std')
        ax.set_title(label)
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.3)
        ax.legend()
    axes[-1].set_xlabel('Inference Samples')

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(outfile, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    return True


def _save_vae_per_objective_plots(agent, out_dir, prefix='vae_objective', dpi=150):
    if not getattr(agent, 'vae_save_per_objective_plots', True):
        return []
    actual_array, pred_array, pred_std_array, labels = _vae_multi_objective_arrays(agent)
    if actual_array is None or pred_array is None or not labels:
        return []

    os.makedirs(out_dir, exist_ok=True)
    x_axis = range(1, actual_array.shape[0] + 1)
    saved_paths = []
    for idx, label in enumerate(labels):
        fig, ax = plt.subplots(figsize=(9, 3.8))
        ax.plot(x_axis, actual_array[:, idx], label='Actual', color='tab:blue')
        ax.plot(x_axis, pred_array[:, idx], label='VAE Predicted', color='tab:red')
        if pred_std_array is not None and pred_std_array.shape[1] > idx:
            lower = pred_array[:, idx] - pred_std_array[:, idx]
            upper = pred_array[:, idx] + pred_std_array[:, idx]
            ax.fill_between(x_axis, lower, upper, color='tab:red', alpha=0.18, label='Pred ±1 std')
        ax.set_title(f'{label}: Actual vs VAE Predicted')
        ax.set_xlabel('Inference Samples')
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        path = os.path.join(out_dir, f'{prefix}_{label}.png')
        fig.savefig(path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        saved_paths.append(path)
    return saved_paths


def _objective_plot_specs_from_histories(objective_plot_history, objective_history):
    return [
        (
            'constraint_reward',
            'Mean Constraint Reward (score)',
            list(objective_plot_history.get('constraint_reward', []) or objective_history.get('constraint_reward', []) or []),
            'Mean Score',
        ),
        (
            'FOML',
            'Mean FOML',
            list(objective_plot_history.get('FOML', []) or []),
            'Mean Value',
        ),
        (
            'FOMS',
            'Mean FOMS',
            list(objective_plot_history.get('FOMS', []) or []),
            'Mean Value',
        ),
        (
            'Active Area',
            'Mean Active Area',
            list(objective_plot_history.get('Active Area', []) or []),
            'Mean Value',
        ),
    ]


def _objective_plot_specs(agent):
    return _objective_plot_specs_from_histories(
        getattr(agent, 'objective_plot_history', {}) or {},
        getattr(agent, 'objective_history', {}) or {},
    )


def _verified_pvt_objective_plot_specs(agent):
    return _objective_plot_specs_from_histories(
        getattr(agent, 'verified_pvt_objective_plot_history', {}) or {},
        getattr(agent, 'verified_pvt_objective_history', {}) or {},
    )


def _save_objective_history_plot_from_specs(specs, outfile, title='Objective History', dpi=150):
    if not any(len(values) > 0 for _, _, values, _ in specs):
        return False

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
    axes = axes.flatten()
    for idx, (_, title_text, values, ylabel) in enumerate(specs):
        ax = axes[idx]
        if values:
            x_axis = range(1, len(values) + 1)
            ax.plot(x_axis, values, color='tab:blue')
        ax.set_title(title_text)
        ax.set_xlabel('Step')
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(outfile, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    return True


def _save_objective_history_plot(agent, outfile, title='Objective History', dpi=150):
    return _save_objective_history_plot_from_specs(
        _objective_plot_specs(agent),
        outfile,
        title=title,
        dpi=dpi,
    )


def _save_reward_range_plot(history_lists, outfile, title, dpi=150, line_color='tab:green', log_color='tab:blue'):
    min_series = [min(r_list) if len(r_list) > 0 else np.nan for r_list in list(history_lists or [])]
    max_series = [max(r_list) if len(r_list) > 0 else np.nan for r_list in list(history_lists or [])]
    if not min_series:
        return False
    steps_axis = range(1, len(min_series) + 1)

    fig = plt.figure(figsize=(6, 4))
    plt.fill_between(steps_axis, min_series, max_series, color=line_color, alpha=0.15, label='range')
    plt.plot(steps_axis, min_series, color=line_color, linewidth=1.8, label='min')
    plt.title(title)
    plt.xlabel('Step')
    plt.ylabel('Training Reward')
    plt.ylim(-10, 0)
    plt.grid(alpha=0.3)
    plt.legend()
    fig.savefig(outfile, dpi=dpi, bbox_inches='tight')
    plt.close(fig)

    log_values = np.asarray(min_series, dtype=float)
    log_values = np.log10(np.maximum(1e-4, -log_values))
    fig_log = plt.figure(figsize=(6, 4))
    plt.plot(steps_axis, log_values, color=log_color, linewidth=1.8, label='log10(-min)')
    plt.title(f'{title} log10(-min)')
    plt.xlabel('Step')
    plt.ylabel('log10(-Training Reward)')
    plt.grid(alpha=0.3)
    plt.legend()
    root, ext = os.path.splitext(outfile)
    fig_log.savefig(f'{root}_log{ext}', dpi=dpi, bbox_inches='tight')
    plt.close(fig_log)
    return True


def _save_pm_history_plot_from_values(pm_violation_history, pm_feasible_rate_history, outfile, title='PM History', dpi=150):
    if not pm_violation_history and not pm_feasible_rate_history:
        return False

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=False)
    if pm_violation_history:
        x_axis = range(1, len(pm_violation_history) + 1)
        axes[0].plot(x_axis, pm_violation_history, color='tab:red')
    axes[0].set_title('Mean PM Violation')
    axes[0].set_xlabel('Step')
    axes[0].set_ylabel('Violation')
    axes[0].grid(True, alpha=0.3)

    if pm_feasible_rate_history:
        x_axis = range(1, len(pm_feasible_rate_history) + 1)
        axes[1].plot(x_axis, pm_feasible_rate_history, color='tab:green')
    axes[1].set_title('PM Feasible Rate')
    axes[1].set_xlabel('Step')
    axes[1].set_ylabel('Rate')
    axes[1].set_ylim(0, 1)
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(outfile, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    return True


def _save_pm_history_plot(agent, outfile, title='PM History', dpi=150):
    return _save_pm_history_plot_from_values(
        list(getattr(agent, 'pm_violation_history', []) or []),
        list(getattr(agent, 'pm_feasible_rate_history', []) or []),
        outfile,
        title=title,
        dpi=dpi,
    )


def log_step(agent, step, episodes, loss, grad_norm_value, mean_reward, std_reward, mean_advantage):
    """Handle per-step local plots, wandb logs, and details txt buffers.
    This mirrors the original code in grpo.py but centralized here.
    """
    # Worst-raw details buffer (flush every 10 steps)
    try:
        block_lines = []
        block_lines.append(f"===== Step {step} (worst corner raw details) =====")
        for ep in episodes:
            perf = (ep.performance or {})
            raw_r = perf.get('raw_reward', None)
            real_action = perf.get('real_action', None)
            if real_action is None:
                try:
                    from utils import ActionNormalizer
                    real_action = ActionNormalizer(agent.env.action_space_low, agent.env.action_space_high).action(np.array(ep.action, copy=True)).tolist()
                except Exception:
                    real_action = None
            block_lines.append(f"-- design_idx={ep.design_idx} worst_reward={raw_r if raw_r is not None else 'N/A'}")
            sim_t = perf.get('pvt_sim_time_sec', None)
            if sim_t is not None:
                block_lines.append(f"sim_time_sec: {sim_t}")
            if real_action is not None:
                block_lines.append(f"real_action: {real_action}")
            # 过滤掉不需要的键：raw_reward、extra_corner、pvt_context
            metrics = get_reporting_metrics(perf)
            scores = get_reporting_scores(perf)
            if metrics:
                block_lines.append("metrics:")
                pm_key = 'phase_margin (deg)'
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
        if agent._details_buffer_start_step is None:
            agent._details_buffer_start_step = step
        agent._details_buffer.append(block_text)
        if step % 10 == 0:
            os.makedirs(agent._logs_dir, exist_ok=True)
            start_s = agent._details_buffer_start_step
            end_s = step
            out_path = os.path.join(agent._logs_dir, f"worst_raw_details_steps_{start_s}-{end_s}.txt")
            with open(out_path, 'w', encoding='utf-8') as f:
                f.write("".join(agent._details_buffer))
            agent._details_buffer = []
            agent._details_buffer_start_step = None
    except Exception:
        pass

    # Local plots and wandb logs
    try:
        context = _plot_runtime_context(agent)
        steps_axis = list(range(1, agent.total_steps + 1))
        # Local plots every 25 steps
        if step % 25 == 0:
            os.makedirs(agent._training_saves_dir, exist_ok=True)
            primary_lists = context['primary_history_lists']
            _save_reward_range_plot(
                primary_lists,
                os.path.join(agent._training_saves_dir, f"{context['primary_filename']}.png"),
                title=context['primary_title'],
                dpi=150,
                line_color='tab:green',
                log_color='tab:blue',
            )

            comparison_lists = context['comparison_history_lists']
            if any(len(l) > 0 for l in comparison_lists):
                _save_reward_range_plot(
                    comparison_lists,
                    os.path.join(agent._training_saves_dir, f"{context['comparison_filename']}.png"),
                    title=context['comparison_title'],
                    dpi=150,
                    line_color='tab:red',
                    log_color='tab:purple',
                )

            verified_pvt_lists = list(getattr(agent, 'per_step_verified_pvt_rewards', []) or [])
            if getattr(agent, 'pvt_outer_loop_enabled', False) and any(len(l) > 0 for l in verified_pvt_lists):
                _save_reward_range_plot(
                    verified_pvt_lists,
                    os.path.join(agent._training_saves_dir, 'verified_pvt_training_reward.png'),
                    title='Verified PVT Training Reward',
                    dpi=150,
                    line_color='tab:orange',
                    log_color='tab:brown',
                )

            fig3 = plt.figure(figsize=(6, 4))
            plt.plot(steps_axis, agent.loss_history, color='tab:orange')
            plt.title('Policy Loss')
            plt.xlabel('Step'); plt.ylabel('Loss'); plt.grid(alpha=0.3)
            fig3.savefig(os.path.join(agent._training_saves_dir, 'policy_loss.png'), dpi=150, bbox_inches='tight')
            plt.close(fig3)

            use_vae = getattr(agent, 'use_vae', getattr(agent, 'use_kan', False))
            vae_total_history = getattr(agent, 'vae_total_history', getattr(agent, 'kan_total_history', []))
            vae_recon_history = getattr(agent, 'vae_recon_history', getattr(agent, 'kan_recon_history', []))
            vae_pred_worst_history = getattr(agent, 'vae_pred_worst_history', getattr(agent, 'kan_pred_worst_history', []))
            vae_actual_worst_history = getattr(agent, 'vae_actual_worst_history', getattr(agent, 'kan_actual_worst_history', []))

            if use_vae and (len(vae_total_history) > 0 or len(vae_recon_history) > 0):
                f_vae = plt.figure(figsize=(6, 4))
                if len(vae_total_history) > 0:
                    plt.plot(range(1, len(vae_total_history) + 1), vae_total_history, label='total', color='tab:blue')
                if len(vae_recon_history) > 0:
                    plt.plot(range(1, len(vae_recon_history) + 1), vae_recon_history, label='recon', color='tab:green')
                plt.title('VAE Losses')
                plt.xlabel('VAE Steps'); plt.ylabel('Loss'); plt.legend(); plt.grid(alpha=0.3)
                os.makedirs(agent._training_saves_dir, exist_ok=True)
                f_vae.savefig(os.path.join(agent._training_saves_dir, 'vae_losses.png'), dpi=150, bbox_inches='tight')
                plt.close(f_vae)

            if use_vae and getattr(agent, 'vae_diagnostics_enabled', True):
                _save_vae_multi_objective_plot(
                    agent,
                    os.path.join(agent._training_saves_dir, 'vae_multi_vs_actual.png'),
                    title='Verified-PVT 5-Objective: Actual vs VAE Predicted',
                    dpi=150,
                )
                _save_vae_per_objective_plots(
                    agent,
                    os.path.join(agent._training_saves_dir, 'vae_objective_plots'),
                    prefix='vae_vs_actual',
                    dpi=150,
                )

            # 移除 corner accuracy 绘图

            # fig4 = plt.figure(figsize=(6, 4))
            # plt.plot(steps_axis, agent.reward_history, label='Mean Training Reward', color='tab:purple')
            # if len(agent.advantage_min_history) == len(steps_axis):
            #     plt.fill_between(steps_axis, agent.advantage_min_history, agent.advantage_max_history, color='tab:purple', alpha=0.15, label='Advantage Range')
            # plt.xlabel('Step'); plt.ylabel('Value'); plt.title('Mean Training Reward & Advantage Range')
            # plt.legend(); plt.grid(alpha=0.3)
            # fig4.savefig(os.path.join(agent._training_saves_dir, 'mean_training_reward_advantage.png'), dpi=150, bbox_inches='tight')
            # plt.close(fig4)
            fig4 = plt.figure(figsize=(6, 4))
            plt.plot(steps_axis, agent.reward_history, label='Mean Training Reward', color='tab:purple')
            plt.xlabel('Step'); plt.ylabel('Training Reward'); plt.title('Mean Training Reward (Constrained Utility)')
            plt.legend(); plt.grid(alpha=0.3)
            fig4.savefig(os.path.join(agent._training_saves_dir, 'mean_training_reward.png'), dpi=150, bbox_inches='tight')
            plt.close(fig4)

            _save_objective_history_plot(
                agent,
                os.path.join(agent._training_saves_dir, 'objective_history.png'),
                title=context['objective_title'],
                dpi=150,
            )
            if getattr(agent, 'pvt_outer_loop_enabled', False):
                _save_objective_history_plot_from_specs(
                    _verified_pvt_objective_plot_specs(agent),
                    os.path.join(agent._training_saves_dir, 'verified_pvt_objective_history.png'),
                    title='Mean Constraint Score + Mean Raw Objective History (Verified-PVT)',
                    dpi=150,
                )
            _save_pm_history_plot(
                agent,
                os.path.join(agent._training_saves_dir, 'pm_history.png'),
                title='PM Constraint History',
                dpi=150,
            )
            if getattr(agent, 'pvt_outer_loop_enabled', False):
                _save_pm_history_plot_from_values(
                    list(getattr(agent, 'verified_pvt_pm_violation_history', []) or []),
                    list(getattr(agent, 'verified_pvt_pm_feasible_rate_history', []) or []),
                    os.path.join(agent._training_saves_dir, 'verified_pvt_pm_history.png'),
                    title='Verified-PVT PM History',
                    dpi=150,
                )
            save_plot_data(agent)

        # wandb plots and scalars each step
        if agent.wandb is not None:
            figs = {}
            steps_axis = list(range(1, agent.total_steps + 1))
            primary_lists = context['primary_history_lists']
            primary_min_series = [min(r_list) if len(r_list) > 0 else np.nan for r_list in primary_lists]
            primary_max_series = [max(r_list) if len(r_list) > 0 else np.nan for r_list in primary_lists]
            f_primary = plt.figure(figsize=(5, 3))
            plt.fill_between(steps_axis, primary_min_series, primary_max_series, color='tab:green', alpha=0.15, label='range')
            plt.plot(steps_axis, primary_min_series, color='tab:green', linewidth=1.4, label='min')
            plt.title(context['primary_title'])
            plt.xlabel('Step'); plt.ylabel('Training Reward'); plt.ylim(-10, 0); plt.grid(alpha=0.3); plt.legend(fontsize=8)
            figs[context['wandb_primary_key']] = f_primary

            f_primary_log = plt.figure(figsize=(5, 3))
            primary_log_series = np.log10(np.maximum(1e-4, -np.array(primary_min_series)))
            plt.plot(steps_axis, primary_log_series, color='tab:blue', linewidth=1.4, label='log10(-min)')
            plt.title(f"{context['primary_title']} log10(-min)")
            plt.xlabel('Step'); plt.ylabel('log10(-Training Reward)'); plt.grid(alpha=0.3); plt.legend(fontsize=8)
            figs[f"{context['wandb_primary_key']}_log"] = f_primary_log

            comparison_lists = context['comparison_history_lists']
            if context['wandb_comparison_key'] and any(len(l) > 0 for l in comparison_lists):
                comparison_min_series = [min(l) if len(l) > 0 else np.nan for l in comparison_lists]
                comparison_max_series = [max(l) if len(l) > 0 else np.nan for l in comparison_lists]
                f_comp = plt.figure(figsize=(5, 3))
                plt.fill_between(steps_axis, comparison_min_series, comparison_max_series, color='tab:red', alpha=0.15, label='range')
                plt.plot(steps_axis, comparison_min_series, color='tab:red', linewidth=1.4, label='min')
                plt.title(context['comparison_title'])
                plt.xlabel('Step'); plt.ylabel('Training Reward'); plt.ylim(-10, 0); plt.grid(alpha=0.3); plt.legend(fontsize=8)
                figs[context['wandb_comparison_key']] = f_comp

                f_comp_log = plt.figure(figsize=(5, 3))
                comparison_log_series = np.log10(np.maximum(1e-4, -np.array(comparison_min_series)))
                plt.plot(steps_axis, comparison_log_series, color='tab:purple', linewidth=1.4, label='log10(-min)')
                plt.title(f"{context['comparison_title']} log10(-min)")
                plt.xlabel('Step'); plt.ylabel('log10(-Training Reward)'); plt.grid(alpha=0.3); plt.legend(fontsize=8)
                figs[f"{context['wandb_comparison_key']}_log"] = f_comp_log

            verified_pvt_lists = list(getattr(agent, 'per_step_verified_pvt_rewards', []) or [])
            if getattr(agent, 'pvt_outer_loop_enabled', False) and any(len(l) > 0 for l in verified_pvt_lists):
                verified_min_series = [min(l) if len(l) > 0 else np.nan for l in verified_pvt_lists]
                verified_max_series = [max(l) if len(l) > 0 else np.nan for l in verified_pvt_lists]
                f_verified = plt.figure(figsize=(5, 3))
                plt.fill_between(steps_axis, verified_min_series, verified_max_series, color='tab:orange', alpha=0.15, label='range')
                plt.plot(steps_axis, verified_min_series, color='tab:orange', linewidth=1.4, label='min')
                plt.title('Verified PVT Training Reward')
                plt.xlabel('Step'); plt.ylabel('Training Reward'); plt.ylim(-10, 0); plt.grid(alpha=0.3); plt.legend(fontsize=8)
                figs['wandb/verified_pvt_training_reward'] = f_verified

                f_verified_log = plt.figure(figsize=(5, 3))
                verified_log_series = np.log10(np.maximum(1e-4, -np.array(verified_min_series)))
                plt.plot(steps_axis, verified_log_series, color='tab:brown', linewidth=1.4, label='log10(-min)')
                plt.title('Verified PVT Training Reward log10(-min)')
                plt.xlabel('Step'); plt.ylabel('log10(-Training Reward)'); plt.grid(alpha=0.3); plt.legend(fontsize=8)
                figs['wandb/verified_pvt_training_reward_log'] = f_verified_log

            f3 = plt.figure(figsize=(5, 3))
            plt.plot(steps_axis, agent.loss_history, color='tab:orange')
            plt.title('Policy Loss')
            plt.xlabel('Step'); plt.ylabel('Loss'); plt.grid(alpha=0.3)
            figs['wandb/policy_loss'] = f3

            f_mean_adv = plt.figure(figsize=(5, 3))
            plt.plot(steps_axis, agent.reward_history, label='mean_training_reward', color='tab:purple')
            if len(agent.advantage_min_history) == len(steps_axis):
                plt.fill_between(steps_axis, agent.advantage_min_history, agent.advantage_max_history, color='tab:purple', alpha=0.15, label='advantage_range')
            plt.title('mean—advantage')
            plt.title('Mean Training Reward and Advantage Range')
            plt.xlabel('Step'); plt.ylabel('Value'); plt.legend(fontsize=8); plt.grid(alpha=0.3)
            figs['wandb/mean_advantage'] = f_mean_adv

            objective_specs = _objective_plot_specs(agent)
            if any(len(values) > 0 for _, _, values, _ in objective_specs):
                f_obj, obj_axes = plt.subplots(2, 2, figsize=(9, 6), sharex=True)
                obj_axes = obj_axes.flatten()
                for idx, (_, title_text, values, ylabel) in enumerate(objective_specs):
                    if values:
                        x_axis = range(1, len(values) + 1)
                        obj_axes[idx].plot(x_axis, values, color='tab:blue')
                    obj_axes[idx].set_title(title_text)
                    obj_axes[idx].set_ylabel(ylabel)
                    obj_axes[idx].grid(alpha=0.3)
                f_obj.suptitle(context['objective_title'])
                f_obj.tight_layout()
                figs['wandb/objective_history'] = f_obj

            verified_objective_specs = _verified_pvt_objective_plot_specs(agent)
            if getattr(agent, 'pvt_outer_loop_enabled', False) and any(len(values) > 0 for _, _, values, _ in verified_objective_specs):
                f_obj_pvt, obj_axes_pvt = plt.subplots(2, 2, figsize=(9, 6), sharex=True)
                obj_axes_pvt = obj_axes_pvt.flatten()
                for idx, (_, title_text, values, ylabel) in enumerate(verified_objective_specs):
                    if values:
                        x_axis = range(1, len(values) + 1)
                        obj_axes_pvt[idx].plot(x_axis, values, color='tab:orange')
                    obj_axes_pvt[idx].set_title(title_text)
                    obj_axes_pvt[idx].set_ylabel(ylabel)
                    obj_axes_pvt[idx].grid(alpha=0.3)
                f_obj_pvt.suptitle('Mean Constraint Score + Mean Raw Objective History (Verified-PVT)')
                f_obj_pvt.tight_layout()
                figs['wandb/verified_pvt_objective_history'] = f_obj_pvt

            pm_violation_history = list(getattr(agent, 'pm_violation_history', []) or [])
            pm_feasible_rate_history = list(getattr(agent, 'pm_feasible_rate_history', []) or [])
            if pm_violation_history or pm_feasible_rate_history:
                f_pm, pm_axes = plt.subplots(1, 2, figsize=(8, 3))
                if pm_violation_history:
                    x_axis = range(1, len(pm_violation_history) + 1)
                    pm_axes[0].plot(x_axis, pm_violation_history, color='tab:red')
                pm_axes[0].set_title('Mean PM Violation')
                pm_axes[0].grid(alpha=0.3)
                if pm_feasible_rate_history:
                    x_axis = range(1, len(pm_feasible_rate_history) + 1)
                    pm_axes[1].plot(x_axis, pm_feasible_rate_history, color='tab:green')
                pm_axes[1].set_title('PM Feasible Rate')
                pm_axes[1].set_ylim(0, 1)
                pm_axes[1].grid(alpha=0.3)
                f_pm.tight_layout()
                figs['wandb/pm_history'] = f_pm

            verified_pm_violation_history = list(getattr(agent, 'verified_pvt_pm_violation_history', []) or [])
            verified_pm_feasible_rate_history = list(getattr(agent, 'verified_pvt_pm_feasible_rate_history', []) or [])
            if getattr(agent, 'pvt_outer_loop_enabled', False) and (verified_pm_violation_history or verified_pm_feasible_rate_history):
                f_pm_pvt, pm_axes_pvt = plt.subplots(1, 2, figsize=(8, 3))
                if verified_pm_violation_history:
                    x_axis = range(1, len(verified_pm_violation_history) + 1)
                    pm_axes_pvt[0].plot(x_axis, verified_pm_violation_history, color='tab:red')
                pm_axes_pvt[0].set_title('Verified-PVT PM Violation')
                pm_axes_pvt[0].grid(alpha=0.3)
                if verified_pm_feasible_rate_history:
                    x_axis = range(1, len(verified_pm_feasible_rate_history) + 1)
                    pm_axes_pvt[1].plot(x_axis, verified_pm_feasible_rate_history, color='tab:green')
                pm_axes_pvt[1].set_title('Verified-PVT PM Feasible Rate')
                pm_axes_pvt[1].set_ylim(0, 1)
                pm_axes_pvt[1].grid(alpha=0.3)
                f_pm_pvt.tight_layout()
                figs['wandb/verified_pvt_pm_history'] = f_pm_pvt

            use_vae = getattr(agent, 'use_vae', getattr(agent, 'use_kan', False))
            vae_total_history = getattr(agent, 'vae_total_history', getattr(agent, 'kan_total_history', []))
            vae_recon_history = getattr(agent, 'vae_recon_history', getattr(agent, 'kan_recon_history', []))
            vae_pred_worst_history = getattr(agent, 'vae_pred_worst_history', getattr(agent, 'kan_pred_worst_history', []))
            vae_actual_worst_history = getattr(agent, 'vae_actual_worst_history', getattr(agent, 'kan_actual_worst_history', []))

            if use_vae and getattr(agent, 'vae_diagnostics_enabled', True) and (len(vae_total_history) > 0 or len(vae_recon_history) > 0):
                f5 = plt.figure(figsize=(5, 3))
                if len(vae_total_history) > 0:
                    plt.plot(range(1, len(vae_total_history) + 1), vae_total_history, label='total', color='tab:blue')
                if len(vae_recon_history) > 0:
                    plt.plot(range(1, len(vae_recon_history) + 1), vae_recon_history, label='recon', color='tab:green')
                plt.title('VAE Losses'); plt.xlabel('VAE Steps'); plt.ylabel('Loss'); plt.legend(fontsize=8); plt.grid(alpha=0.3)
                figs['wandb/vae_losses'] = f5

            # 移除 wandb corner accuracy 绘图

            agent._wandb_log({k: agent.wandb.Image(v) for k, v in figs.items()}, step=step)
            for fig in figs.values():
                plt.close(fig)

            worst_min_current = primary_min_series[-1] if len(primary_min_series) > 0 else None
            scalars = {
                'training_reward/std': std_reward,
                'policy/loss': loss,
                'policy/grad_norm': grad_norm_value,
                'training_reward/mean': mean_reward,
            }
            for scalar_name, _, values, _ in _objective_plot_specs(agent):
                if values:
                    safe_name = scalar_name.replace(' ', '_')
                    scalars[f'objective/{safe_name}'] = float(values[-1])
            pm_violation_history = list(getattr(agent, 'pm_violation_history', []) or [])
            pm_feasible_rate_history = list(getattr(agent, 'pm_feasible_rate_history', []) or [])
            if pm_violation_history:
                scalars['constraint/pm_violation_mean'] = float(pm_violation_history[-1])
            if pm_feasible_rate_history:
                scalars['constraint/pm_feasible_rate'] = float(pm_feasible_rate_history[-1])
            if use_vae:
                if len(vae_total_history) > 0:
                    scalars['vae/total_loss'] = float(vae_total_history[-1])
                if len(vae_recon_history) > 0:
                    scalars['vae/recon_loss'] = float(vae_recon_history[-1])
                # 移除 corner accuracy 标量记录
            if worst_min_current is not None and np.isfinite(worst_min_current):
                scalars['reward/worst_min'] = float(worst_min_current)
            verified_pvt_reward_history = list(getattr(agent, 'verified_pvt_reward_history', []) or [])
            if verified_pvt_reward_history and np.isfinite(verified_pvt_reward_history[-1]):
                scalars['verified_pvt/training_reward_mean'] = float(verified_pvt_reward_history[-1])
            verified_pvt_pm_violation_history = list(getattr(agent, 'verified_pvt_pm_violation_history', []) or [])
            if verified_pvt_pm_violation_history and np.isfinite(verified_pvt_pm_violation_history[-1]):
                scalars['verified_pvt/pm_violation_mean'] = float(verified_pvt_pm_violation_history[-1])
            verified_pvt_pm_feasible_rate_history = list(getattr(agent, 'verified_pvt_pm_feasible_rate_history', []) or [])
            if verified_pvt_pm_feasible_rate_history and np.isfinite(verified_pvt_pm_feasible_rate_history[-1]):
                scalars['verified_pvt/pm_feasible_rate'] = float(verified_pvt_pm_feasible_rate_history[-1])
            if len(agent.per_step_extra_corner_rewards) > 0:
                extra_min_series = [min(l) if len(l) > 0 else np.nan for l in agent.per_step_extra_corner_rewards]
                extra_min_current = extra_min_series[-1] if len(extra_min_series) > 0 else None
                if extra_min_current is not None and np.isfinite(extra_min_current):
                    scalars['reward/tt_min'] = float(extra_min_current)
            agent._wandb_log({k: _to_native(agent, v) for k, v in scalars.items()}, step=step)
    except Exception as _e:
        print(f"[log/plot] error: {_e}")

    # Extra-corner details buffer (flush every 10 steps)
    try:
        extra_blocks = []
        extra_blocks.append(f"===== Step {step} (special corner: output_tt) =====")
        for ep in episodes:
            extra = (ep.performance or {}).get('extra_corner')
            if isinstance(extra, dict):
                perf_extra = extra.get('performance', {}) or {}
                reward_extra = extra.get('reward', None)
                extra_blocks.append(f"-- design_idx={ep.design_idx} reward={reward_extra if reward_extra is not None else 'N/A'}")
                try:
                    sim_t2 = (ep.performance or {}).get('pvt_sim_time_sec', None)
                    if sim_t2 is not None:
                        extra_blocks.append(f"sim_time_sec: {sim_t2}")
                except Exception:
                    pass
                # 不在 extra corner 详情里重复写 real_action
                # 过滤掉 extra_corner 内 performance 中的 pvt_context、extra_corner 自身（一般不会在 perf_extra）
                metrics = get_reporting_metrics(perf_extra)
                scores = get_reporting_scores(perf_extra)
                if metrics:
                    extra_blocks.append("metrics:")
                    pm_key = 'phase_margin (deg)'
                    if pm_key in metrics:
                        extra_blocks.append(f"  {pm_key}: {metrics[pm_key]}")
                    for k in sorted(metrics.keys()):
                        if k == pm_key:
                            continue
                        extra_blocks.append(f"  {k}: {metrics[k]}")
                if scores:
                    extra_blocks.append("scores:")
                    for k in sorted(scores.keys()):
                        extra_blocks.append(f"  {k}: {scores[k]}")
                extra_blocks.append("=" * 80)
        extra_text = "\n".join(extra_blocks) + "\n"
        if agent._extra_details_buffer_start_step is None:
            agent._extra_details_buffer_start_step = step
        agent._extra_details_buffer.append(extra_text)
        if step % 10 == 0:
            os.makedirs(agent._logs_dir, exist_ok=True)
            start_s = agent._extra_details_buffer_start_step
            end_s = step
            out_path = os.path.join(agent._logs_dir, f"extra_corner_details_steps_{start_s}-{end_s}.txt")
            with open(out_path, 'w', encoding='utf-8') as f:
                f.write("".join(agent._extra_details_buffer))
            agent._extra_details_buffer = []
            agent._extra_details_buffer_start_step = None
    except Exception:
        pass


def write_vae_predictions(agent):
    try:
        use_vae = getattr(agent, 'use_vae', getattr(agent, 'use_kan', False))
        diagnostics_enabled = getattr(agent, 'vae_diagnostics_enabled', True)
        vae_pred_records = getattr(agent, '_vae_pred_records', getattr(agent, '_kan_pred_records', []))
        vae_multi_pred_records = getattr(agent, '_vae_multi_pred_records', getattr(agent, '_kan_multi_pred_records', []))

        if use_vae and diagnostics_enabled and len(vae_pred_records) > 0:
            os.makedirs(agent._logs_dir, exist_ok=True)
            out_path = os.path.join(agent._logs_dir, 'vae_predictions.txt')
            with open(out_path, 'w', encoding='utf-8') as f:
                if len(vae_multi_pred_records) > 0:
                    f.write(
                        "step\ttt_constraint_reward\tactual_constraint_reward\tpred_constraint_reward\t"
                        "actual_foml_score\tpred_foml_score\t"
                        "actual_foms_score\tpred_foms_score\t"
                        "actual_active_area_score\tpred_active_area_score\t"
                        "actual_pm_violation\tpred_pm_violation\n"
                    )
                    for record in vae_multi_pred_records:
                        actual = np.asarray(record.get('actual', []), dtype=np.float32)
                        pred = np.asarray(record.get('pred_mean', []), dtype=np.float32)
                        if actual.size < 5 or pred.size < 5:
                            continue
                        f.write(
                            f"{int(record.get('step', 0))}\t"
                            f"{float(record.get('tt_reward', 0.0)):.6f}\t"
                            f"{float(actual[0]):.6f}\t{float(pred[0]):.6f}\t"
                            f"{float(actual[1]):.6f}\t{float(pred[1]):.6f}\t"
                            f"{float(actual[2]):.6f}\t{float(pred[2]):.6f}\t"
                            f"{float(actual[3]):.6f}\t{float(pred[3]):.6f}\t"
                            f"{float(actual[4]):.6f}\t{float(pred[4]):.6f}\n"
                        )
                else:
                    f.write("step\ttt_r\tpvt_r\tpre_r\n")
                    for (st, tr, ar, pr) in vae_pred_records:
                        f.write(f"{st}\t{tr:.6f}\t{ar:.6f}\t{pr:.6f}\n")
            print(f"VAE predictions saved to {out_path}")

            try:
                os.makedirs(agent._training_saves_dir, exist_ok=True)
                multi_plot_path = os.path.join(agent._training_saves_dir, 'vae_multi_vs_actual_final.png')
                if _save_vae_multi_objective_plot(
                    agent,
                    multi_plot_path,
                    title='VAE 5-Objective Prediction vs Actual',
                    dpi=200,
                ):
                    print(f"VAE multi-objective plot saved to {multi_plot_path}")
                per_metric_paths = _save_vae_per_objective_plots(
                    agent,
                    os.path.join(agent._training_saves_dir, 'vae_objective_plots_final'),
                    prefix='vae_vs_actual_final',
                    dpi=200,
                )
                if per_metric_paths:
                    print(
                        "VAE per-objective comparison plots saved to "
                        f"{os.path.dirname(per_metric_paths[0])}"
                    )
            except Exception as exc:
                print(f"[VAE plot] failed to save figure: {exc}")
    except Exception as _e:
        print(f"[VAE txt save] error: {_e}")


def _legacy_write_kan_predictions(agent):
    write_vae_predictions(agent)
    return
    try:
        if agent.use_kan and len(agent._kan_pred_records) > 0:
            os.makedirs(agent._logs_dir, exist_ok=True)
            out_path = os.path.join(agent._logs_dir, 'kan_predictions.txt')
            with open(out_path, 'w', encoding='utf-8') as f:
                f.write("step\ttt_r\tpvt_r\tpre_r\n")
                for (st, tr, ar, pr) in agent._kan_pred_records:
                    f.write(f"{st}\t{tr:.6f}\t{ar:.6f}\t{pr:.6f}\n")
            print(f"✓ KAN predictions saved to {out_path}")

            if agent.kan_actual_worst_history and agent.kan_pred_worst_history:
                try:
                    os.makedirs(agent._training_saves_dir, exist_ok=True)
                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.plot(agent.kan_actual_worst_history, label='Actual PVT Worst Reward', color='tab:blue')
                    ax.plot(agent.kan_pred_worst_history, label='KAN Predicted Reward', color='tab:red')
                    ax.set_xlabel('Inference Steps')
                    ax.set_ylabel('Reward')
                    ax.set_title('KAN Prediction vs Actual PVT Worst Reward')
                    ax.grid(True)
                    ax.legend()
                    plot_path = os.path.join(agent._training_saves_dir, 'kan_actual_vs_pred_final.png')
                    fig.tight_layout()
                    fig.savefig(plot_path, dpi=200)
                    plt.close(fig)
                    print(f"✓ KAN actual-vs-predicted plot saved to {plot_path}")
                except Exception as exc:
                    print(f"[KAN plot] failed to save figure: {exc}")
    except Exception as _e:
        print(f"[KAN txt save] error: {_e}")


def write_kan_predictions(agent):
    write_vae_predictions(agent)


def plot_progress(agent):
    try:
        clear_output(wait=True)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(agent.reward_history, label='Mean Training Reward', color='tab:purple')
        axes[0].set_xlabel('Steps'); axes[0].set_ylabel('Training Reward'); axes[0].set_title('Mean Training Reward (Constrained Utility)'); axes[0].grid(True); axes[0].legend()
        axes[1].plot(agent.loss_history, label='Policy Loss', color='tab:orange')
        axes[1].set_xlabel('Steps'); axes[1].set_ylabel('Loss'); axes[1].set_title('Policy Loss'); axes[1].grid(True); axes[1].legend()
        plt.tight_layout(); plt.show()
    except Exception:
        pass


# === 以下为原 pic.py 中的通用绘图工具，已并入本文件，便于统一维护 ===
from typing import List


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def plot_worst_raw(steps: List[int], worst_raw_history: List[float], outfile: str):
    plt.figure(figsize=(6, 4))
    plt.plot(steps, worst_raw_history, label='Worst Raw Reward', color='red')
    plt.xlabel('Step')
    plt.ylabel('Worst Raw Reward')
    plt.title('Worst Raw Reward per Step')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    ensure_dir(os.path.dirname(outfile) or '.')
    plt.savefig(outfile)
    plt.close()


def plot_loss(steps: List[int], loss_history: List[float], outfile: str):
    plt.figure(figsize=(6, 4))
    plt.plot(steps, loss_history, label='Loss', color='orange')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Policy Loss')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    ensure_dir(os.path.dirname(outfile) or '.')
    plt.savefig(outfile)
    plt.close()


def plot_grad_norm(steps: List[int], grad_norm_history: List[float], outfile: str):
    plt.figure(figsize=(6, 4))
    plt.plot(steps, grad_norm_history, label='Grad Norm', color='purple')
    plt.xlabel('Step')
    plt.ylabel('Grad Norm')
    plt.title('Gradient Norm')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    ensure_dir(os.path.dirname(outfile) or '.')
    plt.savefig(outfile)
    plt.close()


def plot_noise(steps: List[int], noise_history: List[float], outfile: str):
    plt.figure(figsize=(6, 4))
    plt.plot(steps, noise_history, label='Exploration Noise Sigma', color='blue')
    plt.xlabel('Step')
    plt.ylabel('Noise Sigma')
    plt.title('Exploration Noise Decay')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    ensure_dir(os.path.dirname(outfile) or '.')
    plt.savefig(outfile)
    plt.close()


def plot_mean_reward_with_band(
    steps: List[int],
    mean_reward_history: List[float],
    min_history: List[float],
    max_history: List[float],
    outfile: str,
):
    plt.figure(figsize=(6, 4))
    plt.plot(steps, mean_reward_history, label='Mean Training Reward', color='green')
    plt.fill_between(steps, min_history, max_history, color='green', alpha=0.2, label='Min-Max Band')
    plt.xlabel('Step')
    plt.ylabel('Training Reward')
    plt.title('Training Reward with Min-Max Band')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    ensure_dir(os.path.dirname(outfile) or '.')
    plt.savefig(outfile)
    plt.close()


def plot_corner_selection_counts(counts: dict, outfile: str):
    ensure_dir(os.path.dirname(outfile) or '.')
    if not counts:
        # 如果还没有数据，生成一个空图避免报错
        plt.figure(figsize=(6, 4))
        plt.title('Corner Selection Counts (No Data Yet)')
        plt.savefig(outfile)
        plt.close()
        return
    corners = list(counts.keys())
    values = [counts[c] for c in corners]
    plt.figure(figsize=(8, 4))
    plt.bar(corners, values, color='teal', alpha=0.7)
    plt.xlabel('Corner')
    plt.ylabel('Selection Count')
    plt.title('Corner Selection Frequency (Worst-Case Picks)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.grid(axis='y', alpha=0.3)
    plt.savefig(outfile)
    plt.close()
