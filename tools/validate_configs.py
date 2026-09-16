#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Consistency checks for every circuit config + instance in the repo:

  * yaml loads and all file_paths exist on disk
  * the environment constructs (AmpEnv/LdoEnv via env_factory)
  * env.action_dim matches the yaml action_dim
  * graph: num_nodes == #observation_matrix rows, edge indices in range,
    edge_type length matches, every obs row has num_node_features entries
  * every ckt_hierarchy device has an observation_matrix row
  * vars.spice parameter names match the device section

Run:  python tools/validate_configs.py
"""
import os
import sys
import tempfile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
# AmpEnv wipes <cwd>/simulation_output on construction; run from a scratch dir
# so validation never touches real simulation outputs.
os.chdir(tempfile.mkdtemp(prefix='cfg_validate_'))


def check(name, loader):
    from env_factory import make_env
    problems = []
    cfg = loader.get_circuit_config(name)

    # only input files must exist; the *_results/_dat entries are simulation outputs
    input_keys = ('ACDC_cir_path', 'Tran_cir_path', 'netlist_path', 'vars_path',
                  'dev_params_path', 'op_mean_std_path')
    for key in input_keys:
        path = cfg['paths'].get(key)
        if path and not os.path.exists(path):
            problems.append('missing file for %s: %s' % (key, os.path.basename(path)))

    env = make_env(cfg)
    if env.action_dim != cfg.get('action_dim'):
        problems.append('action_dim mismatch: env=%d yaml=%s'
                        % (env.action_dim, cfg.get('action_dim')))

    g = cfg.get('graph', {})
    obs = g.get('observation_matrix', {})
    n = g.get('num_nodes')
    if n != len(obs):
        problems.append('num_nodes=%s but %d observation rows' % (n, len(obs)))
    ei, et = g.get('edge_index', []), g.get('edge_type', [])
    if len(ei) != len(et):
        problems.append('edge_index/edge_type length mismatch (%d vs %d)'
                        % (len(ei), len(et)))
    bad = [e for e in ei if e[0] >= n or e[1] >= n or e[0] < 0 or e[1] < 0]
    if bad:
        problems.append('%d edges out of node range' % len(bad))
    nf = g.get('num_node_features')
    short = [k for k, row in obs.items() if len(row) != nf]
    if short:
        problems.append('obs rows with wrong width: %s' % short)

    for dev in cfg.get('ckt_hierarchy', []):
        if dev[0] not in obs:
            problems.append('hierarchy device %s missing from observation_matrix' % dev[0])

    # vars.spice params vs device section
    expected = set()
    for dev, dcfg in cfg['device'].items():
        for p in dcfg.get('range', {}):
            expected.add('%s_%s' % (p, dev))
    have = set()
    with open(cfg['paths']['vars_path']) as f:
        for ln in f:
            ln = ln.strip()
            if ln.lower().startswith('.param'):
                have.add(ln.split()[1].split('=')[0])
    if expected != have:
        miss, extra = expected - have, have - expected
        if miss:
            problems.append('vars.spice missing params: %s' % sorted(miss))
        if extra:
            problems.append('vars.spice extra params: %s' % sorted(extra))

    return problems


def main():
    from circuit_config_loader import CircuitConfigLoader
    loader = CircuitConfigLoader(config_dir=os.path.join(REPO_ROOT, 'circuit_configs'))
    names = sorted(loader.get_available_circuits())
    n_bad = 0
    for name in names:
        try:
            problems = check(name, loader)
        except Exception as e:
            problems = ['exception: %r' % e]
        if problems:
            n_bad += 1
            print('[FAIL] %s' % name)
            for p in problems:
                print('    - %s' % p)
        else:
            print('[ok]   %s' % name)
    print('\n%d/%d configs clean' % (len(names) - n_bad, len(names)))
    sys.exit(1 if n_bad else 0)


if __name__ == '__main__':
    main()
