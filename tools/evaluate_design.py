#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Evaluate ONE design point of a circuit without any training: write the design
variables, run the ngspice testbenches, and report every measured metric with
its target and score. This is the primary entry point for LLM- or human-driven
manual sizing loops.

Usage (run from the repo root):
  python tools/evaluate_design.py --circuit amp_smc
  python tools/evaluate_design.py --circuit amp_smc --vars W_M0=2 L_M0=1 M_M11=64
  python tools/evaluate_design.py --circuit ldo_basic --vars-file point.json --json

--vars values not given fall back to the yaml `init` values. Values outside the
declared ranges are clamped (with a warning). --json prints a machine-readable
result on stdout (used by the MCP server).

Note: the evaluation runs in a scratch copy under <cwd>/simulation_output/, so
do not run it concurrently with training or op-stats generation in the same cwd.
"""
import argparse
import json
import math
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')


def build_params(cfg, overrides):
    """Full design-variable dict: yaml init values + user overrides, clamped to range."""
    params, warnings = {}, []
    for dev, dcfg in cfg['device'].items():
        for p, bounds in dcfg.get('range', {}).items():
            name = '%s_%s' % (p, dev)
            value = dcfg.get('init', {}).get(p, bounds[0])
            if name in overrides:
                value = overrides.pop(name)
            lo, hi = float(bounds[0]), float(bounds[1])
            if not (lo <= float(value) <= hi):
                warnings.append('%s=%s outside range [%g, %g], clamped' % (name, value, lo, hi))
                value = min(max(float(value), lo), hi)
            params[name] = int(round(float(value))) if p == 'M' else float(value)
    if overrides:
        warnings.append('unknown design variables ignored: %s' % sorted(overrides))
    return params, warnings


def to_jsonable(obj):
    import numpy as np
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        obj = obj.item()
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--circuit', required=True, help='circuit config name, e.g. amp_smc')
    ap.add_argument('--vars', nargs='*', default=[], metavar='NAME=VALUE',
                    help='design-variable overrides, e.g. W_M0=2 M_M11=64')
    ap.add_argument('--vars-file', help='JSON file with a {name: value} dict of overrides')
    ap.add_argument('--json', action='store_true', help='machine-readable output')
    args = ap.parse_args()

    overrides = {}
    if args.vars_file:
        with open(args.vars_file) as f:
            overrides.update(json.load(f))
    for tok in args.vars:
        k, v = tok.split('=', 1)
        overrides[k] = float(v)

    from circuit_config_loader import CircuitConfigLoader
    from env_factory import make_env

    cfg = CircuitConfigLoader().get_circuit_config(args.circuit)
    env = make_env(cfg)
    params, warnings = build_params(cfg, dict(overrides))

    # run in a scratch copy so the instance dir's shipped design point stays intact
    sim_dir = os.path.join(env.base_sim_dir, 'evaluate')
    os.makedirs(sim_dir, exist_ok=True)
    files = [os.path.basename(env.path['ACDC_cir_path']),
             os.path.basename(env.path['Tran_cir_path'])]
    env._prepare_action_files(sim_dir, files)
    env._write_vars_file(sim_dir, params)
    observation, info, reward = env.do_simulation(sim_dir, params)

    if args.json:
        print(json.dumps(to_jsonable({
            'circuit': args.circuit,
            'env': type(env).__name__,
            'params': params,
            'warnings': warnings,
            'reward': reward,
            'info': info,
        }), indent=2))
        return

    for w in warnings:
        print('[warn] %s' % w)
    print('circuit: %s   (env: %s)' % (args.circuit, type(env).__name__))
    print('design variables:')
    for k, v in params.items():
        print('  %-10s = %s' % (k, v))
    print()
    try:
        print(env._generate_performance_table(info))
    except Exception:
        for k, v in info.items():
            if isinstance(v, (int, float)):
                print('  %-24s %s' % (k, v))
    print()
    rc = info.get('reward_components', {})
    print('reward = %.4f   (constraint %.4f, FOML %.4f, FOMS %.4f, Area %.4f)'
          % (reward, rc.get('constraint_reward', float('nan')),
             rc.get('FOML_score', float('nan')), rc.get('FOMS_score', float('nan')),
             rc.get('Active_Area_score', float('nan'))))
    print('all targets satisfied' if reward >= 0 else 'targets not yet satisfied (reward < 0)')


if __name__ == '__main__':
    main()
