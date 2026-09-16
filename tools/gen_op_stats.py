#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate <PREFIX>_op_mean_std.json (transistor .OP statistics used to normalize
the GNN observation matrix) for circuit instances, by running the environment's
random-sizing simulations.

Usage:
  python tools/gen_op_stats.py amp_smc amp_raffc ...      # specific circuits
  python tools/gen_op_stats.py --missing                  # every circuit lacking the json
  python tools/gen_op_stats.py --missing --sims 30
"""
import argparse
import os
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('circuits', nargs='*', help='circuit config names')
    ap.add_argument('--missing', action='store_true',
                    help='process every circuit whose op_mean_std json is missing')
    ap.add_argument('--sims', type=int, default=30,
                    help='number of random simulations (default 30)')
    args = ap.parse_args()

    from circuit_config_loader import CircuitConfigLoader
    from env_factory import make_env

    loader = CircuitConfigLoader()
    names = list(args.circuits)
    if args.missing:
        for name in loader.get_available_circuits():
            cfg = loader.get_circuit_config(name)
            if not os.path.exists(cfg['paths']['op_mean_std_path']):
                names.append(name)
    if not names:
        print('nothing to do')
        return

    for name in names:
        cfg = loader.get_circuit_config(name)
        out = cfg['paths']['op_mean_std_path']
        print('=== %s -> %s (%d sims) ===' % (name, os.path.basename(out), args.sims))
        t0 = time.time()
        env = make_env(cfg)
        env._init_random_sim(args.sims)
        print('=== %s done in %.0fs ===' % (name, time.time() - t0))


if __name__ == '__main__':
    main()
