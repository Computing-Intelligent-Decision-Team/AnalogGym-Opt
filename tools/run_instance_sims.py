#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run every ngspice testbench (*.cir) of one or more circuit instances and report
pass/fail. This is the generic entry point for the simulation-only instances
(temperature sensors, voltage references) and doubles as a smoke test for the
RL-integrated ones.

Usage:
  python tools/run_instance_sims.py sensor_ptat_2t ref_three_output
  python tools/run_instance_sims.py --all            # every dir under simulation_files/
"""
import argparse
import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIM_ROOT = os.path.join(REPO_ROOT, 'simulation_files')
SKIP_DIRS = {'sky130_pdk'}


def run_instance(name, timeout=600):
    sim_dir = os.path.join(SIM_ROOT, name)
    cirs = sorted(f for f in os.listdir(sim_dir) if f.lower().endswith('.cir'))
    if not cirs:
        print('  [skip] %s: no .cir testbench' % name)
        return True
    ok = True
    for cir in cirs:
        log = os.path.splitext(cir)[0] + '.log'
        try:
            r = subprocess.run(['ngspice', '-b', '-o', log, cir], cwd=sim_dir,
                               capture_output=True, text=True, timeout=timeout)
            status = 'ok' if r.returncode == 0 else 'FAIL(rc=%d)' % r.returncode
            ok = ok and r.returncode == 0
        except subprocess.TimeoutExpired:
            status = 'FAIL(timeout)'
            ok = False
        print('  [%s] %s / %s' % (status, name, cir))
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('instances', nargs='*')
    ap.add_argument('--all', action='store_true')
    ap.add_argument('--timeout', type=int, default=600, help='seconds per testbench')
    args = ap.parse_args()

    names = list(args.instances)
    if args.all:
        names = [d for d in sorted(os.listdir(SIM_ROOT))
                 if os.path.isdir(os.path.join(SIM_ROOT, d)) and d not in SKIP_DIRS]
    if not names:
        ap.error('give instance names or --all')

    failed = [n for n in names if not run_instance(n, timeout=args.timeout)]
    print('\n%d/%d instances passed' % (len(names) - len(failed), len(names)))
    if failed:
        print('failed: %s' % ', '.join(failed))
        sys.exit(1)


if __name__ == '__main__':
    main()
